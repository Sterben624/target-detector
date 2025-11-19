"""
ResNet6_CBAM - спрощена версія ResNet10_CBAM для малих датасетів
Архітектура оптимізована для бінарної класифікації звуку дронів-шахедів
"""

import torch
import torch.nn as nn
from typing import Optional

from .cbam import CBAM


class BasicResidualBlock(nn.Module):
    """
    Базовий residual блок з CBAM та dropout
    Структура: Conv -> BN -> ReLU -> Conv -> BN -> CBAM -> Dropout -> Add -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.3,
        reduction_ratio: int = 16
    ):
        """
        Args:
            in_channels: кількість вхідних каналів
            out_channels: кількість вихідних каналів
            stride: крок згортки (для downsampling)
            dropout_rate: ймовірність dropout
            reduction_ratio: коефіцієнт зменшення для CBAM
        """
        super(BasicResidualBlock, self).__init__()

        # Перша згортка
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Друга згортка
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # CBAM модуль уваги
        self.cbam = CBAM(out_channels, reduction_ratio=reduction_ratio)

        # Dropout для регуляризації
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Shortcut connection (якщо розміри не збігаються)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: вхідний тензор [B, C, H, W]
        Returns:
            вихідний тензор після residual блоку [B, C', H', W']
        """
        identity = self.shortcut(x)

        # Перша згортка
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Друга згортка
        out = self.conv2(out)
        out = self.bn2(out)

        # CBAM увага
        out = self.cbam(out)

        # Dropout
        out = self.dropout(out)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out


class ResNet6_CBAM(nn.Module):
    """
    ResNet6_CBAM для бінарної класифікації звуку дронів
    Архітектура: Conv1 -> ResBlock1 -> ResBlock2 -> ResBlock3 -> GAP -> FC
    Всього 6 згорткових шарів (1 початковий + 3 блоки по 2 згортки)
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        reduction_ratio: int = 16
    ):
        """
        Args:
            num_classes: кількість класів (2 для бінарної класифікації)
            in_channels: кількість каналів вхідних даних (1 для MFCC)
            dropout_rate: ймовірність dropout
            reduction_ratio: коефіцієнт зменшення для CBAM
        """
        super(ResNet6_CBAM, self).__init__()

        self.in_channels = 64

        # Початковий згортковий шар
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Три residual блоки (2 + 2 + 2 = 6 згорткових шарів)
        self.layer1 = self._make_layer(
            64, num_blocks=1, stride=1, dropout_rate=dropout_rate, reduction_ratio=reduction_ratio
        )
        self.layer2 = self._make_layer(
            128, num_blocks=1, stride=2, dropout_rate=dropout_rate, reduction_ratio=reduction_ratio
        )
        self.layer3 = self._make_layer(
            256, num_blocks=1, stride=2, dropout_rate=dropout_rate, reduction_ratio=reduction_ratio
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout перед класифікатором
        self.dropout = nn.Dropout(p=dropout_rate)

        # Класифікатор
        self.fc = nn.Linear(256, num_classes)

        # Ініціалізація ваг
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout_rate: float,
        reduction_ratio: int
    ) -> nn.Sequential:
        """
        Створює шар з residual блоків
        """
        layers = []

        # Перший блок може мати stride != 1 для downsampling
        layers.append(
            BasicResidualBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                dropout_rate=dropout_rate,
                reduction_ratio=reduction_ratio
            )
        )
        self.in_channels = out_channels

        # Решта блоків з stride=1
        for _ in range(1, num_blocks):
            layers.append(
                BasicResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout_rate=dropout_rate,
                    reduction_ratio=reduction_ratio
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Ініціалізація ваг мережі за методом He (Kaiming)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: MFCC спектрограми [B, 1, H, W]
        Returns:
            логіти класифікації [B, num_classes]
        """
        # Початковий блок
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual блоки з CBAM
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling та класифікація
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_resnet6_cbam(
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    reduction_ratio: int = 16
) -> ResNet6_CBAM:
    """
    Фабричний метод для створення моделі ResNet6_CBAM

    Args:
        num_classes: кількість класів для класифікації
        dropout_rate: ймовірність dropout
        reduction_ratio: коефіцієнт зменшення для CBAM

    Returns:
        модель ResNet6_CBAM
    """
    return ResNet6_CBAM(
        num_classes=num_classes,
        in_channels=1,
        dropout_rate=dropout_rate,
        reduction_ratio=reduction_ratio
    )


if __name__ == "__main__":
    # Тестування архітектури
    model = create_resnet6_cbam()
    print("ResNet6_CBAM архітектура створена")
    print(f"Загальна кількість параметрів: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Навчаються параметри: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Тест forward pass
    dummy_input = torch.randn(2, 1, 128, 128)  # [batch, channels, height, width]
    output = model(dummy_input)
    print(f"\nВхідна форма: {dummy_input.shape}")
    print(f"Вихідна форма: {output.shape}")
