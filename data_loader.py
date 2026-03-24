"""
data_loader.py - Tải và xử lý dữ liệu CIFAR-10

Tham khảo: dataloader.lua và datasets/cifar10.lua trong code gốc.
Áp dụng Data Augmentation theo đúng bài báo:
  - RandomCrop(32, padding=4): Dịch chuyển ngẫu nhiên
  - RandomHorizontalFlip(): Lật ngang ngẫu nhiên
  - Normalize: Chuẩn hóa màu sắc theo mean/std của CIFAR-10
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


def get_data_loaders(data_dir="./data", batch_size=64, num_workers=2):
    """
    Tải bộ dữ liệu CIFAR-10 và trả về DataLoader cho train và test.

    Args:
        data_dir (str): Thư mục chứa dữ liệu.
        batch_size (int): Kích thước batch. Mặc định 64.
        num_workers (int): Số luồng tải dữ liệu song song.

    Returns:
        trainloader, testloader: DataLoader cho tập huấn luyện và kiểm thử.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Train transform: Có Data Augmentation (Shifting + Mirroring)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # Test transform: Không augmentation, chỉ chuẩn hóa
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = get_data_loaders()
    print(f"Số batch huấn luyện: {len(trainloader)}")
    print(f"Số batch kiểm thử: {len(testloader)}")
