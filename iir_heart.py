# IIR 巴特沃斯带通滤波器通带截止频率0.8Hz,阻带截止频率2 Hz 过滤心跳信号
import numpy as np
from scipy.signal import butter, sosfilt


def iir_heart(n: int, phase: np.ndarray,fs,filt_sos):
    fs = 20  # 采样率为20Hz，即0.05秒的采样间隔
    f1 = 0.8 / (fs/2)  # 归一化通带截止频率
    f2 = 2 / (fs/2)  # 归一化阻带截止频率
    # 设计巴特沃斯IIR滤波器
    # b = [0.0029,   -0.0133,    0.0283,   -0.0359,    0.0283,   -0.0133,    0.0029]
    # a = [1.0000,    2.6015,    3.3373,    2.4564,    1.0833,    0.2662,    0.0283]
    # sos = butter(n, [f1, f2], btype='bandpass', output='sos')
    res = sosfilt(filt_sos, phase)
    return res