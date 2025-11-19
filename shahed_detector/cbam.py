"""
Модуль CBAM (Convolutional Block Attention Module)
Реалізація механізму уваги для покращення якості екстракції ознак
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Канальна увага - фокусується на "що" є важливим у вхідних ознаках
    Використовує глобальний average та max pooling з shared MLP
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: кількість вхідних каналів
            reduction_ratio: коефіцієнт зменшення для MLP (за замовчуванням 16)
        """
        super(ChannelAttention, self).__init__()

        # Shared MLP: FC -> ReLU -> FC
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: вхідний тензор [B, C, H, W]
        Returns:
            тензор уваги [B, C, 1, 1]
        """
        batch_size, channels, _, _ = x.size()

        # Global Average Pooling
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_pool)

        # Global Max Pooling
        max_pool = self.max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_pool)

        # Сумуємо та застосовуємо sigmoid
        attention = self.sigmoid(avg_out + max_out)

        return attention.view(batch_size, channels, 1, 1)


class SpatialAttention(nn.Module):
    """
    Просторова увага - фокусується на "де" знаходиться важлива інформація
    Використовує pooling по каналах та згортку 7x7
    """

    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: розмір ядра згортки (за замовчуванням 7x7)
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "Розмір ядра має бути 3 або 7"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: вхідний тензор [B, C, H, W]
        Returns:
            тензор уваги [B, 1, H, W]
        """
        # Average та Max pooling по каналах
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Конкатенуємо по каналах
        concat = torch.cat([avg_out, max_out], dim=1)

        # Застосовуємо згортку та sigmoid
        attention = self.sigmoid(self.conv(concat))

        return attention


class CBAM(nn.Module):
    """
    Повний модуль CBAM - послідовно застосовує канальну та просторову увагу
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        Args:
            in_channels: кількість вхідних каналів
            reduction_ratio: коефіцієнт зменшення для канальної уваги
            kernel_size: розмір ядра для просторової уваги
        """
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: вхідний тензор [B, C, H, W]
        Returns:
            тензор з застосованою увагою [B, C, H, W]
        """
        # Спочатку канальна увага
        x = x * self.channel_attention(x)

        # Потім просторова увага
        x = x * self.spatial_attention(x)

        return x
