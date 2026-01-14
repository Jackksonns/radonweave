# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""提供了一些常量值。"""

# 从 CT 数据中提取的重建目标的最大线性衰减系数（单位：1/cm）
# 对应于约3071 HU（Hounsfield units）
MU_MAX = 3.336  # = 0.0192 * 3071/1000 + 0.0192