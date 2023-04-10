import pandas
import numpy
import os
import pywt
from scipy.fft import fft
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft
"""
1、三四分位去除无效值
2、hampel去除异常值(耗时过长)、三四分位去除异常值(可实时)
3、小波阈值去噪
"""


class Quantile34:
    """
    三四分位去除无效值
    用单个传感器信号计算无效值截断标准
    """

    def __init__(self, signal):
        self.signal = signal
        self.upper_limit = 0
        self.lower_limit = 0
        self.index = 0
        self.inverse_index = -1

    def get_limit(self, signal_1):
        Q1 = numpy.percentile(signal_1, 25)
        Q3 = numpy.percentile(signal_1, 75)
        IQR = Q3 - Q1
        self.upper_limit = Q3
        self.lower_limit = Q1

    def get_index(self, signal_2):
        for i in range(len(signal_2)):
            if signal_2[i] >= self.upper_limit:
                self.index = i
                break
        for j in range(len(signal_2) - 1, -1, -1):
            if signal_2[j] >= self.upper_limit:
                self.inverse_index = j + 1
                break

    def get_quantile_signal(self):
        signal_ = self.signal[:, 2]
        self.get_index(signal_)
        self.get_limit(signal_)
        processed_signal = self.signal[self.index:self.inverse_index, :]
        return processed_signal

class QuantileFilter:
    """
    三四分位滤波
    """
    def __init__(self,signal):
        self.signal=signal

    def signal_filter(self,signal_origin):
        average = numpy.mean(signal_origin)
        signal_copy = signal_origin.copy()
        Percentile = numpy.percentile(signal_copy, [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        for i in range(len(signal_copy)):
            if signal_copy[i] > UpLimit or signal_copy[i] < DownLimit:
                signal_copy[i] = average
        return signal_copy

    def get_filtered_signal(self):
        signal_filtered=[]
        for i in range(7):
            signal_filtered_i=self.signal_filter(self.signal[:,i])
            signal_filtered.append(signal_filtered_i)
        signal_filtered=numpy.array(signal_filtered)
        return signal_filtered.transpose()


class Hampel:
    """
    hampel去除异常值
    耗时过长 无法用于实时处理
    """

    def __init__(self, signal, k=4000):
        self.signal = signal
        self.k = k

    def get_hampel_signal(self):
        length = self.signal.shape[0] - 1
        n = 3
        signal_pad = numpy.pad(self.signal, ((self.k, self.k), (0, 0)), 'edge')
        index_lower = numpy.array([i - self.k for i in range(self.k, length + self.k + 1)])
        index_upper = numpy.array([i + self.k for i in range(self.k, length + self.k + 1)])
        signal_replace = []
        for j in range(7):
            local_medians = []
            local_stds = []
            for i in range(length + 1):
                window = signal_pad[index_lower[i]:index_upper[i] + 1, j]
                local_median = numpy.median(window)
                local_std = numpy.std(window)
                local_medians.append(local_median)
                local_stds.append(local_std)
            local_medians = numpy.array(local_medians)
            local_stds = numpy.array(local_stds)
            xi = ~(numpy.abs(self.signal[:, j] - local_medians) <= n * local_stds)
            signal_temp = self.signal[:, j].copy()
            signal_temp[xi] = local_medians[xi]
            signal_replace.append(signal_temp)
        signal_replace = numpy.array(signal_replace)
        return signal_replace.transpose()


class WaveletDenoise:
    """
    小波阈值去噪
    """

    def __init__(self, signal, wave='sym8', level=5, mode='periodization'):
        self.signal = signal
        self.wave = wave
        self.level = level
        self.mode = mode

    def freq_analysis(self, signal_, sample_rate):
        # 计算傅里叶变换
        fft_data = fft(signal_)
        # 计算频域参数
        freqs = numpy.fft.fftfreq(len(fft_data), 1 / sample_rate)
        amplitude = numpy.abs(fft_data)
        phase = numpy.angle(fft_data)
        # 提取正频部分
        abs_f = freqs[freqs >= 0]
        abs_a = amplitude[freqs >= 0]
        return abs_f, abs_a

    def signal_denoise(self, signal_):
        coeffs = pywt.wavedec(signal_, wavelet=self.wave, level=self.level, mode=self.mode)
        threshold = numpy.sqrt(2 * numpy.log(len(signal_))) * numpy.mean(numpy.abs(coeffs[0])) / 0.6745
        new_coeffs = [coeffs[0]]
        for coeff in coeffs[1:]:
            new_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
        signal_denoise = pywt.waverec(new_coeffs, self.wave, self.mode)
        return signal_denoise

    def get_denoise_signal(self):
        signal_copy = self.signal.copy()
        for i in range(3, 6):
            x = self.signal_denoise(signal_copy[:, i])
            if len(x) > len(signal_copy[:,i]):
                pad_len = len(x) - len(signal_copy[:, i])
                x = x[: -pad_len]
            signal_copy[:, i] = x
        return signal_copy


class Process:
    """
    预处理整合，默认不使用hampel滤波去除异常值
    """
    def __init__(self, signal, k=4000, wave='sym8', level=5, mode='periodization', quantile=True, filter=True, hampel=False,
                 denoise=True):
        self.signal = signal
        self.k = k
        self.wave = wave
        self.level = level
        self.mode = mode
        self.is_quantile = quantile  # 是否去除无效值
        self.is_filter=filter  # 是否用三四分位去异常
        self.is_hampel = hampel  # 是否hampel去异常
        self.is_denoise = denoise  # 是否去噪

    @property
    def processed_signal(self):
        # 三四分位
        x = self.signal
        if self.is_quantile:
            quantile = Quantile34(x)
            x = quantile.get_quantile_signal()
        # 三四分位去除异常
        if self.is_filter:
            filter=QuantileFilter(x)
            x=filter.get_filtered_signal()
        # hampel去除异常
        if self.is_hampel:
            hampel = Hampel(x, k=self.k)
            x = hampel.get_hampel_signal()
        # 小波去噪
        if self.is_denoise:
            denoise = WaveletDenoise(x, wave=self.wave, level=self.level, mode=self.mode)
            x = denoise.get_denoise_signal()
        return x


if __name__ == '__main__':
    data = pandas.read_csv('E:/pycharm/PHM2010/c1/c1/c_1_001.csv').values
    # quantile=Quantile34(data)
    # a=quantile.get_quantile_signal()

    # filter=QuantileFilter(data)
    # b=filter.get_filtered_signal()

    # hampel = Hampel(data, 4000)
    # c = hampel.get_hampel_signal()

    # wavedenoise = WaveletDenoise(data)
    # d = wavedenoise.get_denoise_signal()

    process=Process(data,hampel=False)
    processed_data=process.processed_signal
    print(processed_data)
