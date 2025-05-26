from typing import Tuple, List, Any
import pandas as pd
import torch
import torchvision.datasets
from PIL import Image
from easydict import EasyDict
from torchvision.transforms import Compose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_sampler


class BMUDatasetPath(torch.utils.data.Dataset):
    def __init__(
            self,
            sample_list: List[dict],
            # transforms_mg: torchvision.transforms.Compose,
            transforms_us: torchvision.transforms.Compose,
    ) -> None:
        super(BMUDatasetPath, self).__init__()

        self.sample_list = sample_list
        # self.transforms_mg = transforms_mg
        self.transforms_us = transforms_us
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(
            self, index: int
    ) -> tuple[Any, Any, dict[Any, Any]]:
        sample = self.sample_list[index]

        # 选择一个 us 图像，这里选择 us3
        us_path = sample["us_path"]
        label = sample["label"]
        clinic_info = sample["clinic_info"]

        # 处理 us 图像
        us_image = (
            self.loader(us_path) if us_path != 'N' else Image.new("RGB", (224, 224), (0, 0, 0))
        )
        # 将 us 图像转换为张量
        us_tensor = self.transforms_us(us_image)

        # 原始路径信息，保留只与 us 相关的路径
        ori = {key: value for key, value in sample.items() if key.endswith('_path') or key == 'view'}

        # 只返回 us_tensor 和其他需要的值
        return us_tensor, clinic_info, label, ori

class BMUDataset(BMUDatasetPath):
    def __getitem__(
            self, index: int
    ) -> tuple[Any, Any, Any]:
        # 获取数据时只保留 us3，并且去除 mg1 和 mg2
        us3, clinic_info, label, ori = super().__getitem__(index)

        # 只返回 us3 和其他必要的信息
        return us3, clinic_info, label


def get_dataloader(
        config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.bmu.split == "static_split":
        train_list = generate_data(config.bmu.train_data_dir)
        val_list = generate_data(config.bmu.val_data_dir)
    elif config.bmu.split == "random_split":
        sample_list = generate_data(config.bmu.train_data_dir)
        train_size = int(len(sample_list) * config.bmu.train_ratio)
        train_list, val_list = torch.utils.data.random_split(
            sample_list, [train_size, len(sample_list) - train_size]
        )

    # 只获取 US 图像的 transform
    train_transform_us, val_transform_us = get_transform(config)

    train_dataset = BMUDataset(
        sample_list=train_list,
        transforms_us=train_transform_us,
    )
    val_dataset = BMUDataset(
        sample_list=val_list,
        transforms_us=val_transform_us,
    )

    sampler = get_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config.bmu.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.bmu.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader

def generate_data(data_dir):
    sample_list = []
    df = pd.read_csv(data_dir, skiprows=0)
    for index, row in df.iterrows():
        data = {
            "patient_id": row["patient_id"],
            "exam_id": row["exam_id"],
            # 修改为仅处理一张US图像
            "us_path": row["us_path"],  # 假设你选择了us3_path作为输入
            "label": row["label"],
            "clinic_info": torch.Tensor(
                [
                    row["clinic_info1"],
                    row["clinic_info2"],
                    row["clinic_info3"],
                    row["clinic_info4"],
                    row["clinic_info5"],
                    row["clinic_info6"],
                    row["clinic_info7"],
                    row["clinic_info8"],
                    row["clinic_info9"],
                    row["clinic_info10"],
                ]
            ),
        }
        sample_list.append(data)
    return sample_list


# Image channel to 3
def force_num_chan(data_tensor):
    data_tensor = data_tensor.float()
    existing_chan = data_tensor.size()[0]
    if not existing_chan == 3:
        return data_tensor.expand(3, *data_tensor.size()[1:])
    return data_tensor


def get_transform(config: EasyDict) -> tuple[Compose, Compose]:
    channel_means = [config.bmu.img_mean]
    channel_stds = [config.bmu.img_std]

    # 只需要处理 US 图像的转换
    train_transform_us = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.bmu.us_image_size, config.bmu.us_image_size)
            ),
            torchvision.transforms.RandomAffine(0, (0.1, 0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=10),
            torchvision.transforms.GaussianBlur(kernel_size=3),
            torchvision.transforms.ToTensor(),
        ]
    )

    val_transform_us = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.bmu.us_image_size, config.bmu.us_image_size)
            ),
            torchvision.transforms.ToTensor(),
        ]
    )

    return train_transform_us, val_transform_us

