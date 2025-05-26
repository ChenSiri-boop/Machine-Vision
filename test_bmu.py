import argparse
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from src import utils
from src.data.loader_bmu import generate_data, get_transform, BMUDatasetPath
from src.model.bmunet import BMUNet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

parser = argparse.ArgumentParser(description="arg parser")

parser.add_argument("--weight-path", type=str, default="./output/best/pytorch_model.bin")
parser.add_argument("--test-path", type=str, default="./test.csv")


args = parser.parse_args()


def get_dataset():
    test_list = generate_data(args.test_path)
    _, val_transform_us = get_transform(config)
    test_dataset = BMUDatasetPath(
        sample_list=test_list,
        # transforms_mg=val_transform_mg,
        transforms_us=val_transform_us,
    )
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=0, shuffle=False
    )
    return loader


if __name__ == "__main__":
    # Load config
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    accelerator = Accelerator()
    utils.same_seeds(42)

    # Load test data
    test_loader = get_dataset()

    # Load model weight
    model = BMUNet()
    model_weight = torch.load(args.weight_path)
    model.load_state_dict(model_weight)

    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, (us_img, clinic_info, label,_) in enumerate(test_loader):
            logist = model(us_img, clinic_info=clinic_info)  # logits shape: [B, 2]
            scores = F.softmax(logist, dim=-1)  # shape: [B, 2]
            probs = scores[:, 1]  # 取阳性类别的概率

            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算 ROC 曲线坐标 & AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # 计算 ACC
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)

    print(f"Test Accuracy (ACC): {acc:.4f}")
    print(f"Test AUC: {roc_auc:.4f}")

    # 保存 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('./roc.png')
    plt.close()
