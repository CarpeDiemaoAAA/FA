#!/usr/bin/env python3
"""
LADH数据集全面分析脚本
=====================
分析维度：
  1. 数据集结构统计（被试数、实验数、文件完整性）
  2. SpO2标签分析（分布、方差、异常值、每被试覆盖范围）
  3. HR标签分析（分布、方差、动态范围）
  4. RR原始信号分析（采样率、幅值范围、信号质量）
  5. BVP信号分析（采样率、信号质量）
  6. 视频帧时间戳分析（帧率稳定性、RGB-IR同步性）
  7. 信号时间对齐分析（各信号时间范围是否匹配）
  8. 训练相关分析（chunk级标签分布、标签方差热图）
  9. 异常样本检测（SpO2常值片段、极端值、信号质量差的片段）
  10. 模型优化建议（基于分析结果自动生成）

使用方法：
  python analyze_dataset.py --data_path /gpfs/home/zhangaofeng/EMXZ/datasets_double
  python analyze_dataset.py --data_path /gpfs/home/zhangaofeng/EMXZ/datasets_double --output_dir ./analysis_results
"""

import os
import sys
import glob
import argparse
import warnings
import json
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal as scipy_signal

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
CHUNK_LENGTH = 256
VIDEO_FPS = 30  # 目标帧率
SPO2_NORMAL_RANGE = (85, 100)
HR_NORMAL_RANGE = (40, 180)
RR_BPM_NORMAL_RANGE = (5, 60)

# ============================================================
# 工具函数
# ============================================================


def safe_read_csv(filepath, required_cols=None):
    """安全读取CSV，返回DataFrame或None"""
    try:
        df = pd.read_csv(filepath)
        if required_cols:
            for col in required_cols:
                if col not in df.columns:
                    return None
        return df
    except Exception:
        return None


def compute_sampling_rate(timestamps):
    """从时间戳序列计算采样率"""
    if len(timestamps) < 2:
        return 0.0
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    return 1.0 / np.median(diffs)


def compute_signal_snr(sig, fs, freq_low, freq_high):
    """通过频谱计算信号的信噪比(dB)"""
    try:
        if len(sig) < 64:
            return np.nan
        sig = sig - np.mean(sig)
        f, pxx = scipy_signal.periodogram(sig, fs=fs)
        band_mask = (f >= freq_low) & (f <= freq_high)
        noise_mask = ~band_mask & (f > 0)
        sig_power = np.sum(pxx[band_mask])
        noise_power = np.sum(pxx[noise_mask])
        if noise_power == 0:
            return np.nan
        return 10 * np.log10(sig_power / noise_power)
    except Exception:
        return np.nan


# ============================================================
# 核心分析类
# ============================================================

class DatasetAnalyzer:
    def __init__(self, data_path, output_dir="./analysis_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 存储分析结果
        self.sessions = []  # 每个session的信息
        self.spo2_all = []  # 所有SpO2值
        self.hr_all = []
        self.rr_bpm_all = []
        self.bvp_snr_all = []
        self.session_spo2_stats = []  # 每个session的SpO2统计
        self.chunk_spo2_labels = []  # chunk级SpO2标签
        self.issues = []  # 发现的问题
        self.per_subject_spo2 = defaultdict(list)  # 每被试SpO2
        self.per_subject_hr = defaultdict(list)

    def discover_sessions(self):
        """发现所有数据session"""
        print("=" * 70)
        print("  [1/10] 扫描数据集结构")
        print("=" * 70)

        date_dirs = sorted(glob.glob(os.path.join(self.data_path, "*")))
        date_dirs = [d for d in date_dirs if os.path.isdir(d)]

        total_sessions = 0
        date_info = {}

        for date_dir in date_dirs:
            date_name = os.path.basename(date_dir)
            subject_dirs = sorted(glob.glob(os.path.join(date_dir, "p_*")))
            date_subjects = 0
            date_sessions = 0

            for subj_dir in subject_dirs:
                subj_name = os.path.basename(subj_dir)
                exp_dirs = sorted(glob.glob(os.path.join(subj_dir, "v*")))

                for exp_dir in exp_dirs:
                    exp_name = os.path.basename(exp_dir)
                    session_id = f"{date_name}/{subj_name}/{exp_name}"

                    # 检查必要文件
                    required_files = {
                        'spo2': os.path.join(exp_dir, 'SpO2.csv'),
                        'hr': os.path.join(exp_dir, 'HR.csv'),
                        'bvp': os.path.join(exp_dir, 'BVP.csv'),
                        'rr': os.path.join(exp_dir, 'RR.csv'),
                        'ts_rgb': os.path.join(exp_dir, 'frames_timestamp_RGB.csv'),
                        'ts_ir': os.path.join(exp_dir, 'frames_timestamp_IR.csv'),
                        'info': os.path.join(exp_dir, 'info.txt'),
                    }

                    missing = [k for k, v in required_files.items()
                               if not os.path.exists(v)]

                    self.sessions.append({
                        'id': session_id,
                        'path': exp_dir,
                        'date': date_name,
                        'subject': subj_name,
                        'experiment': exp_name,
                        'files': required_files,
                        'missing': missing,
                    })
                    date_sessions += 1
                    total_sessions += 1

                date_subjects += 1

            date_info[date_name] = {
                'subjects': date_subjects, 'sessions': date_sessions}

        # 统计打印
        all_subjects = set(s['subject'] for s in self.sessions)
        print(f"  数据集路径: {self.data_path}")
        print(f"  采集日期数: {len(date_info)}")
        print(f"  总被试数:   {len(all_subjects)}")
        print(f"  总session数: {total_sessions}")
        print()

        for date, info in sorted(date_info.items()):
            print(
                f"    {date}: {info['subjects']} 被试, {info['sessions']} sessions")

        # 文件完整性
        incomplete = [s for s in self.sessions if s['missing']]
        if incomplete:
            print(f"\n  ⚠ 文件不完整的session: {len(incomplete)}/{total_sessions}")
            for s in incomplete[:5]:
                print(f"    {s['id']}: 缺少 {s['missing']}")
            if len(incomplete) > 5:
                print(f"    ... 还有 {len(incomplete)-5} 个")
            self.issues.append(f"文件不完整的session: {len(incomplete)}")
        else:
            print(f"\n  ✓ 所有session文件完整")

        return self

    def analyze_spo2(self):
        """分析SpO2标签"""
        print("\n" + "=" * 70)
        print("  [2/10] SpO2标签分析（重点）")
        print("=" * 70)

        spo2_values_all = []
        session_stats = []
        constant_sessions = []  # SpO2全程不变的session
        low_variance_sessions = []
        unique_value_counter = Counter()

        for sess in self.sessions:
            if 'spo2' in sess['missing']:
                continue
            df = safe_read_csv(sess['files']['spo2'], ['timestamp', 'spo2'])
            if df is None or len(df) == 0:
                continue

            vals = df['spo2'].values.astype(float)
            timestamps = df['timestamp'].values

            spo2_values_all.extend(vals.tolist())
            self.per_subject_spo2[sess['subject']].extend(vals.tolist())

            for v in vals:
                unique_value_counter[int(v)] += 1

            stat = {
                'session': sess['id'],
                'subject': sess['subject'],
                'n_samples': len(vals),
                'duration_s': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                'sampling_rate': compute_sampling_rate(timestamps),
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'range': np.max(vals) - np.min(vals),
                'unique_count': len(np.unique(vals)),
                'pct_below_90': np.mean(vals < 90) * 100,
                'pct_below_95': np.mean(vals < 95) * 100,
            }
            session_stats.append(stat)

            if stat['range'] == 0:
                constant_sessions.append(sess['id'])
            if stat['std'] < 0.5:
                low_variance_sessions.append(sess['id'])

        self.spo2_all = np.array(spo2_values_all)
        self.session_spo2_stats = session_stats

        if len(spo2_values_all) == 0:
            print("  ⚠ 未找到SpO2数据！")
            return self

        # 全局统计
        arr = self.spo2_all
        print(f"\n  --- 全局SpO2分布 ---")
        print(f"  样本总数:     {len(arr)}")
        print(f"  均值±标准差:  {np.mean(arr):.2f} ± {np.std(arr):.2f}")
        print(f"  中位数:       {np.median(arr):.1f}")
        print(f"  最小值:       {np.min(arr):.0f}")
        print(f"  最大值:       {np.max(arr):.0f}")
        print(f"  范围:         {np.max(arr) - np.min(arr):.0f}")

        # 分位数
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{pct:02d}: {np.percentile(arr, pct):.1f}", end="  ")
            if pct in [10, 50, 90]:
                print()
        print()

        # 值分布（直方图文本形式）
        print(f"\n  --- SpO2值分布 ---")
        total = len(arr)
        for val in sorted(unique_value_counter.keys()):
            cnt = unique_value_counter[val]
            pct = cnt / total * 100
            bar = "█" * int(pct * 2)
            print(f"    SpO2={val:3d}: {cnt:6d} ({pct:5.1f}%) {bar}")

        # 问题检测
        print(f"\n  --- 问题检测 ---")
        print(
            f"  SpO2全程不变的session:    {len(constant_sessions)}/{len(session_stats)}")
        print(
            f"  SpO2低方差(std<0.5)的session: {len(low_variance_sessions)}/{len(session_stats)}")
        print(
            f"  SpO2<90%的采样点:         {np.sum(arr < 90)} ({np.mean(arr < 90)*100:.2f}%)")
        print(
            f"  SpO2<95%的采样点:         {np.sum(arr < 95)} ({np.mean(arr < 95)*100:.2f}%)")

        if len(constant_sessions) > 0:
            self.issues.append(
                f"SpO2全程不变: {len(constant_sessions)} sessions (模型无法从中学习变化)")
            print(f"  ⚠ 以下session的SpO2全程恒定（无法提供训练梯度）:")
            for s in constant_sessions[:10]:
                stat = next(x for x in session_stats if x['session'] == s)
                print(
                    f"    {s}: SpO2恒定={stat['mean']:.0f}%, 时长={stat['duration_s']:.0f}s")

        if len(low_variance_sessions) > len(session_stats) * 0.5:
            self.issues.append(f"超过50%的session SpO2方差<0.5 → 模型倾向输出均值")

        # 每被试SpO2范围
        print(f"\n  --- 每被试SpO2覆盖范围 ---")
        print(
            f"  {'被试ID':<25s} {'样本数':>6s} {'均值':>6s} {'标准差':>6s} {'范围':>10s} {'唯一值数':>6s}")
        print(f"  {'-'*65}")
        for subj in sorted(self.per_subject_spo2.keys()):
            vals = np.array(self.per_subject_spo2[subj])
            rng = f"{np.min(vals):.0f}-{np.max(vals):.0f}"
            print(
                f"  {subj:<25s} {len(vals):6d} {np.mean(vals):6.1f} {np.std(vals):6.2f} {rng:>10s} {len(np.unique(vals)):6d}")

        return self

    def analyze_hr(self):
        """分析HR标签"""
        print("\n" + "=" * 70)
        print("  [3/10] HR标签分析")
        print("=" * 70)

        hr_all = []
        for sess in self.sessions:
            if 'hr' in sess['missing']:
                continue
            df = safe_read_csv(sess['files']['hr'], ['timestamp', 'hr'])
            if df is None or len(df) == 0:
                continue
            vals = df['hr'].values.astype(float)
            hr_all.extend(vals.tolist())
            self.per_subject_hr[sess['subject']].extend(vals.tolist())

        self.hr_all = np.array(hr_all)

        if len(hr_all) == 0:
            print("  ⚠ 未找到HR数据")
            return self

        arr = self.hr_all
        print(f"  样本总数:     {len(arr)}")
        print(f"  均值±标准差:  {np.mean(arr):.1f} ± {np.std(arr):.1f} bpm")
        print(f"  范围:         {np.min(arr):.0f} - {np.max(arr):.0f} bpm")

        # HR分区统计
        zones = [(40, 60, "静息低"), (60, 80, "正常"), (80, 100, "偏高"),
                 (100, 120, "高"), (120, 180, "极高")]
        for lo, hi, name in zones:
            cnt = np.sum((arr >= lo) & (arr < hi))
            pct = cnt / len(arr) * 100
            print(f"    {name}({lo}-{hi}): {cnt} ({pct:.1f}%)")

        outliers = np.sum((arr < 40) | (arr > 180))
        if outliers > 0:
            self.issues.append(f"HR异常值(超出40-180): {outliers}个")
            print(f"  ⚠ HR异常值: {outliers}个")

        return self

    def analyze_rr(self):
        """分析RR信号"""
        print("\n" + "=" * 70)
        print("  [4/10] RR信号分析")
        print("=" * 70)

        rr_stats = []
        for sess in self.sessions[:50]:  # 采样分析（全量太慢）
            if 'rr' in sess['missing']:
                continue
            df = safe_read_csv(sess['files']['rr'], ['timestamp', 'rr'])
            if df is None or len(df) == 0:
                continue

            vals = df['rr'].values.astype(float)
            timestamps = df['timestamp'].values
            fs = compute_sampling_rate(timestamps)

            rr_stats.append({
                'session': sess['id'],
                'n_samples': len(vals),
                'sampling_rate': fs,
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
            })

        if not rr_stats:
            print("  ⚠ 未找到RR数据")
            return self

        fs_arr = [s['sampling_rate'] for s in rr_stats]
        print(
            f"  采样率:  {np.mean(fs_arr):.1f} ± {np.std(fs_arr):.1f} Hz (中位数 {np.median(fs_arr):.1f})")
        print(
            f"  RR值范围: {np.min([s['min'] for s in rr_stats]):.0f} - {np.max([s['max'] for s in rr_stats]):.0f}")
        print(f"  RR均值:   {np.mean([s['mean'] for s in rr_stats]):.1f}")
        print(f"  注意: RR.csv包含的是原始呼吸波形（非呼吸率），需要通过FFT/峰值检测转换为呼吸率(bpm)")

        return self

    def analyze_bvp(self):
        """分析BVP信号"""
        print("\n" + "=" * 70)
        print("  [5/10] BVP信号分析")
        print("=" * 70)

        bvp_stats = []
        for sess in self.sessions[:50]:
            if 'bvp' in sess['missing']:
                continue
            df = safe_read_csv(sess['files']['bvp'], ['timestamp', 'bvp'])
            if df is None or len(df) == 0:
                continue

            vals = df['bvp'].values.astype(float)
            timestamps = df['timestamp'].values
            fs = compute_sampling_rate(timestamps)

            # 信号质量：计算心率频段的SNR
            snr = compute_signal_snr(vals, fs, 0.75, 2.5)
            self.bvp_snr_all.append(snr)

            bvp_stats.append({
                'session': sess['id'],
                'n_samples': len(vals),
                'sampling_rate': fs,
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'snr': snr,
            })

        if not bvp_stats:
            print("  ⚠ 未找到BVP数据")
            return self

        fs_arr = [s['sampling_rate'] for s in bvp_stats]
        snr_arr = [s['snr'] for s in bvp_stats if not np.isnan(s['snr'])]
        print(f"  采样率:   {np.mean(fs_arr):.1f} ± {np.std(fs_arr):.1f} Hz")
        print(
            f"  BVP值范围: {np.min([s['min'] for s in bvp_stats]):.0f} - {np.max([s['max'] for s in bvp_stats]):.0f}")
        if snr_arr:
            print(
                f"  信号SNR:  {np.mean(snr_arr):.1f} ± {np.std(snr_arr):.1f} dB")

        return self

    def analyze_timestamps(self):
        """分析帧时间戳和RGB-IR同步性"""
        print("\n" + "=" * 70)
        print("  [6/10] 视频帧时间戳分析")
        print("=" * 70)

        fps_rgb_all = []
        fps_ir_all = []
        sync_errors = []
        frame_count_diffs = []

        for sess in self.sessions[:50]:
            if 'ts_rgb' in sess['missing'] or 'ts_ir' in sess['missing']:
                continue

            df_rgb = safe_read_csv(sess['files']['ts_rgb'], [
                                   'frame', 'timestamp'])
            df_ir = safe_read_csv(sess['files']['ts_ir'], [
                                  'frame', 'timestamp'])
            if df_rgb is None or df_ir is None:
                continue

            ts_rgb = df_rgb['timestamp'].values
            ts_ir = df_ir['timestamp'].values

            fps_rgb = compute_sampling_rate(ts_rgb)
            fps_ir = compute_sampling_rate(ts_ir)
            fps_rgb_all.append(fps_rgb)
            fps_ir_all.append(fps_ir)

            # 帧数差异
            frame_diff = abs(len(ts_rgb) - len(ts_ir))
            frame_count_diffs.append(frame_diff)

            # 时间同步误差（对齐帧的时间差）
            min_len = min(len(ts_rgb), len(ts_ir))
            if min_len > 0:
                sync_err = np.abs(ts_rgb[:min_len] - ts_ir[:min_len])
                sync_errors.append(np.mean(sync_err))

        if fps_rgb_all:
            print(
                f"  RGB帧率: {np.mean(fps_rgb_all):.1f} ± {np.std(fps_rgb_all):.1f} fps")
            print(
                f"  IR帧率:  {np.mean(fps_ir_all):.1f} ± {np.std(fps_ir_all):.1f} fps")
            print(
                f"  RGB-IR帧数差异: 均值 {np.mean(frame_count_diffs):.1f}, 最大 {np.max(frame_count_diffs):.0f}")

            if sync_errors:
                print(
                    f"  RGB-IR同步误差: 均值 {np.mean(sync_errors)*1000:.1f}ms, 最大 {np.max(sync_errors)*1000:.1f}ms")

                if np.max(sync_errors) > 0.1:  # >100ms
                    self.issues.append(
                        f"RGB-IR同步误差最大达 {np.max(sync_errors)*1000:.0f}ms")

        return self

    def analyze_signal_alignment(self):
        """分析各信号的时间对齐情况"""
        print("\n" + "=" * 70)
        print("  [7/10] 信号时间对齐分析")
        print("=" * 70)

        alignment_issues = 0
        total_checked = 0

        for sess in self.sessions[:30]:
            if any(k in sess['missing'] for k in ['spo2', 'bvp', 'rr', 'ts_rgb']):
                continue

            df_spo2 = safe_read_csv(sess['files']['spo2'], ['timestamp'])
            df_bvp = safe_read_csv(sess['files']['bvp'], ['timestamp'])
            df_rr = safe_read_csv(sess['files']['rr'], ['timestamp'])
            df_rgb = safe_read_csv(sess['files']['ts_rgb'], ['timestamp'])

            if any(x is None for x in [df_spo2, df_bvp, df_rr, df_rgb]):
                continue

            total_checked += 1

            # 各信号的时间范围
            ranges = {
                'SpO2': (df_spo2['timestamp'].min(), df_spo2['timestamp'].max()),
                'BVP':  (df_bvp['timestamp'].min(),  df_bvp['timestamp'].max()),
                'RR':   (df_rr['timestamp'].min(),   df_rr['timestamp'].max()),
                'RGB':  (df_rgb['timestamp'].min(),  df_rgb['timestamp'].max()),
            }

            # 检查是否有信号在视频之外
            rgb_start, rgb_end = ranges['RGB']
            for sig_name, (sig_start, sig_end) in ranges.items():
                if sig_name == 'RGB':
                    continue
                # 信号应该覆盖视频的大部分时间范围
                video_duration = rgb_end - rgb_start
                coverage = min(sig_end, rgb_end) - max(sig_start, rgb_start)
                coverage_pct = coverage / video_duration * 100 if video_duration > 0 else 0

                if coverage_pct < 80:
                    alignment_issues += 1

        print(f"  检查session数: {total_checked}")
        print(f"  信号覆盖不足(<80%)的session: {alignment_issues}")

        if alignment_issues > 0:
            self.issues.append(f"信号时间覆盖不足的session: {alignment_issues}")

        return self

    def analyze_chunk_labels(self):
        """模拟训练流程的chunk分割，分析chunk级SpO2标签"""
        print("\n" + "=" * 70)
        print("  [8/10] Chunk级标签分析（模拟训练数据）")
        print("=" * 70)

        chunk_spo2_means = []
        chunk_spo2_stds = []
        chunk_spo2_ranges = []
        constant_chunks = 0
        total_chunks = 0

        for sess in self.sessions:
            if 'spo2' in sess['missing'] or 'ts_rgb' in sess['missing']:
                continue

            df_spo2 = safe_read_csv(sess['files']['spo2'], [
                                    'timestamp', 'spo2'])
            df_rgb = safe_read_csv(sess['files']['ts_rgb'], ['timestamp'])
            if df_spo2 is None or df_rgb is None:
                continue

            spo2_ts = df_spo2['timestamp'].values
            spo2_vals = df_spo2['spo2'].values.astype(float)
            frame_ts = df_rgb['timestamp'].values

            if len(spo2_ts) < 2 or len(frame_ts) < CHUNK_LENGTH:
                continue

            # 重采样SpO2到视频帧率（与训练流程一致）
            try:
                interpolator = interp1d(
                    spo2_ts, spo2_vals, bounds_error=False, fill_value="extrapolate")
                resampled_spo2 = interpolator(frame_ts)
            except Exception:
                continue

            # 按chunk分割
            n_chunks = len(resampled_spo2) // CHUNK_LENGTH
            for i in range(n_chunks):
                chunk = resampled_spo2[i *
                                       CHUNK_LENGTH: (i + 1) * CHUNK_LENGTH]
                chunk_mean = np.mean(chunk)
                chunk_std = np.std(chunk)
                chunk_range = np.max(chunk) - np.min(chunk)

                chunk_spo2_means.append(chunk_mean)
                chunk_spo2_stds.append(chunk_std)
                chunk_spo2_ranges.append(chunk_range)
                total_chunks += 1

                if chunk_range < 0.5:  # chunk内SpO2变化<0.5%
                    constant_chunks += 1

        self.chunk_spo2_labels = np.array(chunk_spo2_means)

        if total_chunks == 0:
            print("  ⚠ 无法生成chunk数据")
            return self

        means = np.array(chunk_spo2_means)
        stds = np.array(chunk_spo2_stds)
        ranges = np.array(chunk_spo2_ranges)

        print(f"  总chunk数: {total_chunks}")
        print(f"  Chunk长度: {CHUNK_LENGTH}帧 ≈ {CHUNK_LENGTH/VIDEO_FPS:.1f}秒")
        print(f"\n  --- Chunk级SpO2标签均值分布 ---")
        print(f"  均值±标准差: {np.mean(means):.2f} ± {np.std(means):.2f}")
        print(f"  范围: {np.min(means):.1f} - {np.max(means):.1f}")

        # 直方图
        bins = np.arange(int(np.min(means)), int(np.max(means)) + 2, 0.5)
        hist, edges = np.histogram(means, bins=bins)
        print(f"\n  Chunk SpO2标签分布直方图:")
        max_bar = 50
        max_count = max(hist) if max(hist) > 0 else 1
        for i, count in enumerate(hist):
            if count > 0:
                bar_len = int(count / max_count * max_bar)
                print(
                    f"    [{edges[i]:5.1f},{edges[i+1]:5.1f}): {count:5d} {'█' * bar_len}")

        # 标签方差分析
        print(f"\n  --- Chunk内SpO2方差分析 ---")
        print(f"  Chunk内标准差均值:   {np.mean(stds):.3f}")
        print(f"  Chunk内标准差中位数: {np.median(stds):.3f}")
        print(
            f"  Chunk内SpO2恒定(变化<0.5%): {constant_chunks}/{total_chunks} ({constant_chunks/total_chunks*100:.1f}%)")

        if constant_chunks / total_chunks > 0.5:
            self.issues.append(
                f"超过{constant_chunks/total_chunks*100:.0f}%的chunk内SpO2恒定——模型从中学不到任何变化信息")

        # Batch方差模拟
        print(f"\n  --- 模拟Batch方差（batch_size=16）---")
        batch_size = 16
        n_batches = min(1000, total_chunks // batch_size)
        batch_vars = []
        for _ in range(n_batches):
            batch_idx = np.random.choice(
                total_chunks, batch_size, replace=False)
            batch_labels = means[batch_idx]
            batch_vars.append(np.var(batch_labels))

        batch_vars = np.array(batch_vars)
        print(f"  模拟{n_batches}个batch的标签方差:")
        print(f"    方差均值:   {np.mean(batch_vars):.4f}")
        print(f"    方差中位数: {np.median(batch_vars):.4f}")
        print(f"    方差<0.5的比例: {np.mean(batch_vars < 0.5)*100:.1f}%")
        print(f"    方差<1.0的比例: {np.mean(batch_vars < 1.0)*100:.1f}%")

        if np.mean(batch_vars < 1.0) > 0.7:
            self.issues.append("超过70%的batch标签方差<1.0 → 排序损失/CCC损失可能不稳定")

        return self

    def analyze_outliers(self):
        """异常样本检测"""
        print("\n" + "=" * 70)
        print("  [9/10] 异常样本检测")
        print("=" * 70)

        anomalies = {
            'spo2_extreme': [],
            'spo2_constant_long': [],
            'hr_extreme': [],
            'timestamp_gap': [],
        }

        for sess in self.sessions:
            # SpO2极端值
            if 'spo2' not in sess['missing']:
                df = safe_read_csv(sess['files']['spo2'], [
                                   'timestamp', 'spo2'])
                if df is not None:
                    vals = df['spo2'].values.astype(float)
                    if np.any(vals < 80) or np.any(vals > 100):
                        anomalies['spo2_extreme'].append(
                            f"{sess['id']}: min={np.min(vals)}, max={np.max(vals)}")
                    if np.std(vals) == 0 and len(vals) > 30:
                        anomalies['spo2_constant_long'].append(
                            f"{sess['id']}: 恒定值={vals[0]}, 时长={len(vals)}s")

            # HR极端值
            if 'hr' not in sess['missing']:
                df = safe_read_csv(sess['files']['hr'], ['timestamp', 'hr'])
                if df is not None:
                    vals = df['hr'].values.astype(float)
                    if np.any(vals < 40) or np.any(vals > 180):
                        anomalies['hr_extreme'].append(
                            f"{sess['id']}: min={np.min(vals)}, max={np.max(vals)}")

            # 帧时间戳间隙
            if 'ts_rgb' not in sess['missing']:
                df = safe_read_csv(sess['files']['ts_rgb'], ['timestamp'])
                if df is not None:
                    ts = df['timestamp'].values
                    if len(ts) > 1:
                        gaps = np.diff(ts)
                        max_gap = np.max(gaps)
                        if max_gap > 0.5:  # > 500ms间隙
                            anomalies['timestamp_gap'].append(
                                f"{sess['id']}: 最大间隙={max_gap*1000:.0f}ms")

        print(f"  SpO2超出80-100范围:  {len(anomalies['spo2_extreme'])}个session")
        for s in anomalies['spo2_extreme'][:5]:
            print(f"    {s}")

        print(
            f"  SpO2长时间恒定:      {len(anomalies['spo2_constant_long'])}个session")
        for s in anomalies['spo2_constant_long'][:5]:
            print(f"    {s}")

        print(f"  HR超出40-180范围:    {len(anomalies['hr_extreme'])}个session")
        for s in anomalies['hr_extreme'][:5]:
            print(f"    {s}")

        print(f"  帧时间戳大间隙(>500ms): {len(anomalies['timestamp_gap'])}个session")
        for s in anomalies['timestamp_gap'][:5]:
            print(f"    {s}")

        return self

    def generate_optimization_suggestions(self):
        """基于分析结果生成模型优化建议"""
        print("\n" + "=" * 70)
        print("  [10/10] 模型优化建议")
        print("=" * 70)

        suggestions = []

        # 基于SpO2分析
        if len(self.spo2_all) > 0:
            spo2_std = np.std(self.spo2_all)
            spo2_range = np.max(self.spo2_all) - np.min(self.spo2_all)
            spo2_mean = np.mean(self.spo2_all)

            if spo2_std < 2.0:
                suggestions.append({
                    'priority': 'P0',
                    'category': '数据分布',
                    'issue': f'SpO2标签方差极小 (std={spo2_std:.2f}), 集中在{spo2_mean:.0f}%附近',
                    'suggestion': '1) 考虑标签增强：对chunk标签加入同session内的局部方差信息\n'
                    '    2) 损失函数必须对微小差异敏感（排序损失优于MSE）\n'
                    '    3) 使用相对误差而非绝对误差作为优化目标',
                })

            # 检查SpO2是否以整数分布
            int_vals = self.spo2_all.astype(int)
            if np.allclose(self.spo2_all, int_vals):
                suggestions.append({
                    'priority': 'P1',
                    'category': '标签精度',
                    'issue': 'SpO2标签全部为整数值，精度仅1%',
                    'suggestion': '1) CMS50E脉搏血氧仪分辨率限制，标签天然离散\n'
                    '    2) 回归目标的有效精度仅±1%，MAE理论下界≈0.5%\n'
                    '    3) 当前MAE=1.32已经接近设备精度极限\n'
                    '    4) 考虑将SpO2建模为有序分类(ordinal regression)而非连续回归',
                })

        # 基于chunk分析
        if len(self.chunk_spo2_labels) > 0:
            chunk_std = np.std(self.chunk_spo2_labels)

            if chunk_std < 1.5:
                suggestions.append({
                    'priority': 'P0',
                    'category': '训练策略',
                    'issue': f'Chunk级SpO2标签方差极小 (std={chunk_std:.2f})',
                    'suggestion': '1) 采样策略优化：过采样SpO2偏离均值的chunk（加权采样器）\n'
                    '    2) 增大batch_size（32或更大）以包含更多SpO2变化\n'
                    '    3) 按SpO2值分层采样，确保每个batch覆盖完整范围',
                })

        # 基于HR分析
        if len(self.hr_all) > 0:
            hr_std = np.std(self.hr_all)
            if hr_std > 15:
                suggestions.append({
                    'priority': 'P2',
                    'category': '多任务',
                    'issue': f'HR动态范围大 (std={hr_std:.1f})，而SpO2范围小',
                    'suggestion': '多任务学习中BVP/HR梯度可能主导训练\n'
                    '    考虑: 先单独训练SpO2分支几个epoch再联合训练\n'
                    '    或: 使用梯度归一化(GradNorm)自动平衡多任务权重',
                })

        # 通用建议
        suggestions.append({
            'priority': 'P1',
            'category': '模型架构',
            'issue': '当前tanh映射范围[85,103]可能不够精确',
            'suggestion': '根据实际数据分布调整映射中心和范围:\n'
            f'    数据均值={np.mean(self.spo2_all):.1f}, '
            f'建议: tanh(x)*{max(3, spo2_range/2):.0f} + {spo2_mean:.0f}',
        })

        # 打印建议
        for i, s in enumerate(sorted(suggestions, key=lambda x: x['priority'])):
            print(f"\n  [{s['priority']}] {s['category']}: {s['issue']}")
            print(f"    建议: {s['suggestion']}")

        return suggestions

    def save_results(self):
        """保存分析结果到文件"""
        print("\n" + "=" * 70)
        print("  保存分析结果")
        print("=" * 70)

        # 保存SpO2 session统计
        if self.session_spo2_stats:
            df = pd.DataFrame(self.session_spo2_stats)
            path = os.path.join(self.output_dir, 'spo2_session_stats.csv')
            df.to_csv(path, index=False)
            print(f"  ✓ SpO2 session统计 → {path}")

        # 保存chunk级标签分布
        if len(self.chunk_spo2_labels) > 0:
            path = os.path.join(self.output_dir, 'chunk_spo2_labels.npy')
            np.save(path, self.chunk_spo2_labels)
            print(f"  ✓ Chunk SpO2标签 → {path}")

        # 保存每被试统计
        subj_stats = []
        for subj in sorted(self.per_subject_spo2.keys()):
            vals = np.array(self.per_subject_spo2[subj])
            hr_vals = np.array(self.per_subject_hr.get(subj, []))
            subj_stats.append({
                'subject': subj,
                'n_spo2_samples': len(vals),
                'spo2_mean': np.mean(vals),
                'spo2_std': np.std(vals),
                'spo2_min': np.min(vals),
                'spo2_max': np.max(vals),
                'spo2_range': np.max(vals) - np.min(vals),
                'spo2_unique': len(np.unique(vals)),
                'hr_mean': np.mean(hr_vals) if len(hr_vals) > 0 else np.nan,
                'hr_std': np.std(hr_vals) if len(hr_vals) > 0 else np.nan,
            })
        if subj_stats:
            df = pd.DataFrame(subj_stats)
            path = os.path.join(self.output_dir, 'per_subject_stats.csv')
            df.to_csv(path, index=False)
            print(f"  ✓ 每被试统计 → {path}")

        # 保存问题汇总
        path = os.path.join(self.output_dir, 'issues_summary.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write("数据集分析发现的问题\n")
            f.write("=" * 50 + "\n\n")
            for i, issue in enumerate(self.issues, 1):
                f.write(f"{i}. {issue}\n")
        print(f"  ✓ 问题汇总 → {path}")

        print(f"\n  所有结果已保存到: {self.output_dir}/")

    def try_generate_plots(self):
        """尝试生成可视化图表（matplotlib可用时）"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            print("\n" + "=" * 70)
            print("  生成可视化图表")
            print("=" * 70)
        except ImportError:
            print("\n  matplotlib不可用，跳过图表生成")
            return self

        # 1. SpO2全局分布
        if len(self.spo2_all) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('SpO2 Data Analysis', fontsize=14)

            # 1a. 全局直方图
            ax = axes[0, 0]
            ax.hist(self.spo2_all, bins=np.arange(self.spo2_all.min()-0.5, self.spo2_all.max()+1.5, 1),
                    edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('SpO2 (%)')
            ax.set_ylabel('Count')
            ax.set_title('SpO2 Label Distribution (All Samples)')
            ax.axvline(np.mean(self.spo2_all), color='red',
                       linestyle='--', label=f'Mean={np.mean(self.spo2_all):.1f}')
            ax.legend()

            # 1b. Chunk级标签分布
            ax = axes[0, 1]
            if len(self.chunk_spo2_labels) > 0:
                ax.hist(self.chunk_spo2_labels, bins=50,
                        edgecolor='black', alpha=0.7, color='darkorange')
                ax.set_xlabel('SpO2 Chunk Mean (%)')
                ax.set_ylabel('Count')
                ax.set_title(
                    f'Chunk-level SpO2 Labels (n={len(self.chunk_spo2_labels)})')
                ax.axvline(np.mean(self.chunk_spo2_labels), color='red', linestyle='--',
                           label=f'Mean={np.mean(self.chunk_spo2_labels):.2f}')
                ax.legend()

            # 1c. 每被试SpO2箱线图
            ax = axes[1, 0]
            subjects = sorted(self.per_subject_spo2.keys())
            if len(subjects) <= 40:
                data_for_box = [self.per_subject_spo2[s] for s in subjects]
                bp = ax.boxplot(data_for_box, vert=True, patch_artist=True)
                ax.set_xticklabels([s.split('_')[-1]
                                   for s in subjects], rotation=90, fontsize=6)
                ax.set_ylabel('SpO2 (%)')
                ax.set_title('SpO2 Range per Subject')
            else:
                # 太多被试用散点图
                subj_means = [np.mean(self.per_subject_spo2[s])
                              for s in subjects]
                subj_stds = [np.std(self.per_subject_spo2[s])
                             for s in subjects]
                ax.errorbar(range(len(subjects)), subj_means,
                            yerr=subj_stds, fmt='o', markersize=3, alpha=0.6)
                ax.set_xlabel('Subject Index')
                ax.set_ylabel('SpO2 (%) Mean ± Std')
                ax.set_title(f'SpO2 per Subject (n={len(subjects)})')

            # 1d. HR vs SpO2 散点图
            ax = axes[1, 1]
            common_subjects = set(self.per_subject_spo2.keys()) & set(
                self.per_subject_hr.keys())
            if common_subjects:
                spo2_means = [np.mean(self.per_subject_spo2[s])
                              for s in common_subjects]
                hr_means = [np.mean(self.per_subject_hr[s])
                            for s in common_subjects]
                ax.scatter(hr_means, spo2_means, alpha=0.6, s=20)
                ax.set_xlabel('HR (bpm)')
                ax.set_ylabel('SpO2 (%)')
                ax.set_title('HR vs SpO2 per Subject')

            plt.tight_layout()
            path = os.path.join(self.output_dir, 'spo2_analysis.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  ✓ SpO2分析图表 → {path}")

        # 2. HR分布
        if len(self.hr_all) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(self.hr_all, bins=50, edgecolor='black',
                    alpha=0.7, color='coral')
            ax.set_xlabel('HR (bpm)')
            ax.set_ylabel('Count')
            ax.set_title('HR Distribution')
            ax.axvline(np.mean(self.hr_all), color='red', linestyle='--',
                       label=f'Mean={np.mean(self.hr_all):.1f}')
            ax.legend()
            plt.tight_layout()
            path = os.path.join(self.output_dir, 'hr_distribution.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  ✓ HR分布图表 → {path}")

        return self

    def run_full_analysis(self):
        """执行完整分析流程"""
        print()
        print("╔" + "═" * 68 + "╗")
        print("║" + "   LADH数据集全面分析报告".center(62) + "║")
        print("╚" + "═" * 68 + "╝")

        self.discover_sessions()
        self.analyze_spo2()
        self.analyze_hr()
        self.analyze_rr()
        self.analyze_bvp()
        self.analyze_timestamps()
        self.analyze_signal_alignment()
        self.analyze_chunk_labels()
        self.analyze_outliers()

        suggestions = self.generate_optimization_suggestions()

        self.try_generate_plots()
        self.save_results()

        # 最终汇总
        print("\n" + "=" * 70)
        print("  分析完成 — 发现的关键问题汇总")
        print("=" * 70)
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. ⚠ {issue}")
        else:
            print("  ✓ 未发现关键问题")

        print("\n" + "=" * 70)
        print(f"  分析结果已保存至: {self.output_dir}")
        print("=" * 70)

        return suggestions


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LADH数据集全面分析')
    parser.add_argument('--data_path', type=str,
                        default='/gpfs/home/zhangaofeng/EMXZ/datasets_double',
                        help='数据集根目录路径')
    parser.add_argument('--output_dir', type=str,
                        default='./analysis_results',
                        help='分析结果输出目录')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"错误: 数据路径不存在: {args.data_path}")
        print("请用 --data_path 指定正确路径")
        sys.exit(1)

    analyzer = DatasetAnalyzer(args.data_path, args.output_dir)
    suggestions = analyzer.run_full_analysis()

    return suggestions


if __name__ == '__main__':
    main()
