import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),  # 恢复通道
            nn.Sigmoid()  # 激活函数，生成通道权重
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 压缩空间维度
        y = self.fc(y).view(b, c, 1, 1, 1)  # 恢复形状
        return x * y  # 通道注意力加权


class IR_SE_CNN(nn.Module):
    def __init__(self, input_channels=3):
        super(IR_SE_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(16),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )

    def forward(self, x):
        return self.features(x)

# class FusionNet(nn.Module):
#     def __init__(self):
#         super(FusionNet, self).__init__()
#         self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(64)
#         self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         return x


class FusionNet(nn.Module):
    def __init__(self, feature_channels=64):
        super(FusionNet, self).__init__()
        self.rgb_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.ir_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels)
        )

        # 通道注意力Refinement，提升融合后特征质量
        self.channel_refine = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, feature_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_ir):
        # Gating: 输出每个模态的权重 [B, 1, 1, 1, 1]
        gate_rgb = self.rgb_gate(x_rgb)
        gate_ir = self.ir_gate(x_ir)
        # 归一化门控权重，确保两个模态的贡献之和为1
        gate_sum = gate_rgb + gate_ir + 1e-8
        gate_rgb = gate_rgb / gate_sum
        gate_ir = gate_ir / gate_sum

        fused = gate_rgb * x_rgb + gate_ir * x_ir

        # 残差连接 + 通道注意力精炼
        fused_out = self.fuse_conv(fused)
        channel_att = self.channel_refine(fused_out)
        fused_out = fused_out * channel_att
        fused_out = fused_out + fused  # 残差连接
        return fused_out


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            SEBlock(16)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            SEBlock(32)
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )

        self.rppg_branch = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0),
        )

        self.rr_branch = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

# ===== SpO2 多尺度特征提取 =====
        # 路径1：高层时空特征（从编码器最终输出提取全局模式）
        self.spo2_rgb_high = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 1, 1)),
            nn.Flatten(),
            nn.Linear(64 * 4, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.spo2_ir_high = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 1, 1)),
            nn.Flatten(),
            nn.Linear(64 * 4, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )

        # 路径2：通道统计特征（捕获AC/DC比率——SpO2的物理基础）
        # 对每个通道计算mean(DC分量)和std(AC分量)，形成2*64=128维统计描述
        self.spo2_rgb_stats = nn.Sequential(
            nn.Linear(128, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.spo2_ir_stats = nn.Sequential(
            nn.Linear(128, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )

        # 路径3：RGB-IR交叉注意力（学习跨模态光学比率关系）
        self.spo2_cross_attn = nn.Sequential(
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Linear(48, 32),
        )

        # 路径4：时序动态特征（捕获融合后脉搏波AC分量的时域模式——SpO2物理基础）
        self.spo2_temporal_conv = nn.Sequential(
            nn.Conv1d(64, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # SpO2预测头：48+48+48+48+32+32 = 256
        # 深度残差结构：消除瓶颈，释放预测范围
        self.spo2_pre = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.spo2_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # 残差精炼块：深层网络保持梯度流通
        self.spo2_refine = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.03),
            nn.Linear(64, 64),
        )
        self.spo2_final = nn.Linear(64, 1)
        # 残差快捷连接：直接从原始特征预测SpO2偏移
        self.spo2_shortcut = nn.Linear(256, 1)

        # 可学习中心偏置（数据分布均值94.4作为初始锚点）
        self.spo2_center = nn.Parameter(
            torch.tensor(94.4, dtype=torch.float32))

        self.ir_encoder = IR_SE_CNN()
        self.fusion_net = FusionNet()

        # SpO2相关层的Kaiming初始化
        self._init_spo2_weights()

    def _compute_channel_stats(self, x):
        """计算通道级统计特征：mean(DC分量) + std(AC分量)

        Args:
            x: [B, C, T, H, W] 编码器特征
        Returns:
            stats: [B, 2*C] 每通道的均值和标准差
        """
        b, c = x.shape[0], x.shape[1]
        x_flat = x.view(b, c, -1)  # [B, C, T*H*W]
        ch_mean = x_flat.mean(dim=2)  # [B, C] DC分量
        ch_std = x_flat.std(dim=2) + 1e-6  # [B, C] AC分量
        return torch.cat([ch_mean, ch_std], dim=1)  # [B, 2C]

    def _init_spo2_weights(self):
        """对SpO2相关模块进行Kaiming初始化，提升训练起点质量"""
        spo2_modules = [
            self.spo2_rgb_high, self.spo2_ir_high,
            self.spo2_rgb_stats, self.spo2_ir_stats,
            self.spo2_cross_attn, self.spo2_temporal_conv,
            self.spo2_pre, self.spo2_head, self.spo2_refine,
        ]
        for module_list in spo2_modules:
            for m in module_list.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        # 快捷连接和最终层用小权重初始化（让残差起步时接近0）
        nn.init.normal_(self.spo2_shortcut.weight, std=0.01)
        nn.init.constant_(self.spo2_shortcut.bias, 0)
        nn.init.normal_(self.spo2_final.weight, std=0.05)
        nn.init.constant_(self.spo2_final.bias, 0)

    def forward(self, x1, x2=None):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, width, height] = x1.shape
        x2_ir_feat = None  # 保存IR编码器原始特征，供SpO2专用路径使用
        x1_visual = None   # 保存RGB编码器预融合特征，供SpO2光学比率路径使用
        if x2 is not None:
            x1_visual = self.share_m(x1)
            x2_ir_feat = self.ir_encoder(x2)

            x = self.fusion_net(x1_visual, x2_ir_feat)
            # x = self.encode_video_x(x)

            # x = self.fusion_net(x1, x2)
            # x = self.encode_video_x(x)
            # x = self.encode_video(x)
        else:
            # x = self.share_m(x1)
            # rPPG = self.encode_video_rppg(x)
            # rr = self.encode_video_rr(x)
            x = self.encode_video(x1)
        # print(f"x.shape: {x.shape}")

        rPPG = self.rppg_branch(x)
        rPPG = rPPG.view(batch, length)
        rr = self.rr_branch(x)
        rr = rr.view(batch, length)

# ===== SpO2 多尺度预测 =====
        if x1_visual is not None and x2_ir_feat is not None:
            # 路径1：高层时空特征
            rgb_high = self.spo2_rgb_high(x1_visual)      # [B, 48]
            ir_high = self.spo2_ir_high(x2_ir_feat)       # [B, 48]

            # 路径2：通道统计特征（AC/DC比率信息）
            rgb_stats = self._compute_channel_stats(x1_visual)  # [B, 128]
            rgb_stats = self.spo2_rgb_stats(rgb_stats)          # [B, 48]
            ir_stats = self._compute_channel_stats(x2_ir_feat)  # [B, 128]
            ir_stats = self.spo2_ir_stats(ir_stats)             # [B, 48]

            # 路径3：交叉注意力（模拟R值 = AC_red/DC_red / AC_ir/DC_ir）
            cross_in = torch.cat([rgb_high * ir_high,
                                  rgb_high - ir_high], dim=1)  # [B, 96]
            cross_feat = self.spo2_cross_attn(cross_in)        # [B, 32]
        else:
            rgb_high = self.spo2_rgb_high(x)                   # [B, 48]
            ir_high = torch.zeros(batch, 48, device=x.device)
            rgb_stats = self._compute_channel_stats(x)
            rgb_stats = self.spo2_rgb_stats(rgb_stats)         # [B, 48]
            ir_stats = torch.zeros(batch, 48, device=x.device)
            cross_feat = torch.zeros(batch, 32, device=x.device)

        # 路径4：时序动态特征（从融合特征提取时域脉搏波模式）
        fused_temporal = x.mean(dim=(-2, -1))  # [B, 64, T] 空间池化保留时序
        temporal_feat = self.spo2_temporal_conv(fused_temporal)  # [B, 32]

        # 拼接所有特征: 48+48+48+48+32+32 = 256
        spo2_feat = torch.cat(
            [rgb_high, ir_high, rgb_stats, ir_stats, cross_feat, temporal_feat], dim=1)

        # 深度残差预测头
        spo2_pre = self.spo2_pre(spo2_feat)              # [B, 128]
        spo2_h = self.spo2_head(spo2_pre)                # [B, 64]
        spo2_r = self.spo2_refine(spo2_h) + spo2_h      # [B, 64] 残差连接
        spo2_main = self.spo2_final(spo2_r)              # [B, 1]
        spo2_skip = self.spo2_shortcut(spo2_feat)        # [B, 1]
        spo2_offset = spo2_main + 0.2 * spo2_skip        # [B, 1]

        # ★ 线性输出 + 软边界（彻底消除tanh压缩，释放全部预测范围）
        # 直接 center + offset 保持线性，不压缩梯度
        spo2 = self.spo2_center + spo2_offset.view(batch, 1)
        # 软边界：仅在超出生理范围时平滑截断 [75, 100]
        spo2 = 75.0 + F.softplus(spo2 - 75.0)
        spo2 = 100.0 - F.softplus(100.0 - spo2)

        return rPPG, spo2, rr

    def encode_video(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        return x

    def share_m(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = nn.AdaptiveAvgPool3d((64, 18, 18))(x)   # [8, 64, 64, 18, 18]
        return x

    def encode_video_x(self, x):
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        return x
