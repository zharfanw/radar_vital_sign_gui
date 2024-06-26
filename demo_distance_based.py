# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DistanceAlgo import *
from iir_heart import iir_heart
from IIR_Breath import iir_breath
from PeakHeart import peakheart
from PeakBreath import peakbreath

import threading
import time
import datetime

from scipy.signal import butter, sosfilt

# -------------------------------------------------
# Presentation
# -------------------------------------------------
class Draw:
    # Draws plots for data - each antenna is in separated plot

    def __init__(self, max_range_m, num_ant, num_samples):
        # max_range_m:  maximum supported range
        # num_ant:      number of available antennas

        self._num_ant = num_ant
        self._pln = []

        plt.ion()

        self._fig, self._axs = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        self._fig.canvas.manager.set_window_title("Range FFT")
        self._fig.set_size_inches(17 / 3 * num_ant, 4)

        self._dist_points = np.linspace(
            0,
            max_range_m,
            num_samples)

        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas):
        # Create common plots as well scale it in same way
        # data_all_antennas: array of raw data for each antenna
        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            # This is a workaround: If there is only one plot then self._axs is
            # an object of the blass type AxesSubplot. However, if multiple
            # axes is available (if more than one RX antenna is activated),
            # then self._axs is an numpy.ndarray of AxesSubplot.
            # The code above gets in both cases the axis.
            if type(self._axs) == np.ndarray:
                ax = self._axs[i_ant]
            else:
                ax = self._axs

            data = data_all_antennas[i_ant]
            pln, = ax.plot(self._dist_points, data)
            ax.set_ylim(minmin, 1.05 * maxmax)
            self._pln.append(pln)

            ax.set_xlabel("distance (m)")
            ax.set_ylabel("FFT magnitude")
            ax.set_title("Antenna #" + str(i_ant))
        self._fig.tight_layout()

    def _draw_next_time(self, data_all_antennas):
        # data_all_antennas: array of raw data for each antenna

        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._pln[i_ant].set_ydata(data)

    def draw(self, data_all_antennas):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        if self._is_window_open:
            if len(self._pln) == 0:  # handle the first run
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open

# -------------------------------------------------
# Presentation
# -------------------------------------------------
class VitalDraw:
    # Draws plots for data - each antenna is in separated plot

    def __init__(self,num_ant, num_samples):
        # max_range_m:  maximum supported range
        # num_ant:      number of available antennas

        self._num_ant = num_ant
        self._pln = []

        plt.ion()

        self._fig, self._axs = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        self._fig.canvas.manager.set_window_title("Range FFT")
        self._fig.set_size_inches(17 / 3 * num_ant, 4)

        self._dist_points = np.arange(0,num_samples,1)

        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas):
        # Create common plots as well scale it in same way
        # data_all_antennas: array of raw data for each antenna
        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            # This is a workaround: If there is only one plot then self._axs is
            # an object of the blass type AxesSubplot. However, if multiple
            # axes is available (if more than one RX antenna is activated),
            # then self._axs is an numpy.ndarray of AxesSubplot.
            # The code above gets in both cases the axis.
            if type(self._axs) == np.ndarray:
                ax = self._axs[i_ant]
            else:
                ax = self._axs

            data = data_all_antennas[i_ant]
            pln, = ax.plot(self._dist_points, data)
            ax.set_ylim(0, 0.04)
            self._pln.append(pln)

            ax.set_xlabel("Time")
            ax.set_ylabel("Selected magnitude")
            ax.set_title("Antenna #" + str(i_ant))
        self._fig.tight_layout()

    def _draw_next_time(self, data_all_antennas):
        # data_all_antennas: array of raw data for each antenna
        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._pln[i_ant].set_ydata(data)
            # self._axs[i_ant].set_ylim(0,maxmax)

    def draw(self, data_all_antennas):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        if self._is_window_open:
            if len(self._pln) == 0:  # handle the first run
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open

class VitalSignDraw:
    # Draws plots for data - each antenna is in separated plot

    def __init__(self, num_samples, name,fs):
        # max_range_m:  maximum supported range

        self._pln = []

        plt.ion()
        self._name = name

        self._fig, self._axs = plt.subplots(nrows=2, ncols=1, figsize=((1 + 1) // 2, 2))
        self._fig.canvas.manager.set_window_title(self._name)
        self._fig.set_size_inches(17 / 3, 4)

        self._dist_points = np.arange(0,num_samples,1)

        dum_points = round(num_samples/2)
        dum_single_point = fs/(2*dum_points)
        self._dist_points_fft = np.arange(0,(fs/2),dum_single_point)

        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True
        self._is_first = True

    def _draw_first_time(self, heart_data,dat_fft):
        # Create common plots as well scale it in same way
        # data_all_antennas: array of raw data for each antenna
        # minmin = min(heart_data)
        # maxmax = max(heart_data)
        minmin = -0.002
        maxmax = 0.002

        ax = self._axs[0]
        ax_fft = self._axs[1]

        data = heart_data
        pln, = ax.plot(self._dist_points, data)
        ax.set_ylim(minmin, maxmax)
        self._pln = pln
        ax.set_xlabel("Time")
        ax.set_ylabel("Selected magnitude")
        ax.set_title(self._name + " Rate")

        # data_fft = dat_fft
        minmin = min(dat_fft)
        maxmax = max(dat_fft)
        pln_fft, = ax_fft.plot(self._dist_points_fft,dat_fft)
        ax_fft.set_ylim(minmin, maxmax)
        self._pln_fft = pln_fft
        ax_fft.set_xlabel("Freq")
        ax_fft.set_ylabel("Magnitude")
        ax_fft.set_title(self._name + " Rate Spectrum")


        self._fig.tight_layout()



    def _draw_next_time(self, heart_data,dat_fft,str_bpm):
        # data_all_antennas: array of raw data for each antenna
        minmin = min(heart_data)
        maxmax = max(heart_data)
        ax = self._axs[0]
        
        data = heart_data
        self._pln.set_ydata(data)
        ax.set_ylim(minmin,maxmax)

        minmin = min(dat_fft)
        maxmax = max(dat_fft)
        ax_fft = self._axs[1]
        self._pln_fft.set_ydata(dat_fft)
        ax_fft.set_ylim(minmin,maxmax)
        ax_fft.set_title(self._name + " Rate Spectrum|BPM : "+str_bpm)



    def draw(self, datanya,fftnya,str_bpm):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        if self._is_window_open:
            if self._is_first:  # handle the first run
                self._draw_first_time(datanya,fftnya)
                self._is_first = False
            else:
                self._draw_next_time(datanya,fftnya,str_bpm)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open


class Filter_moving_avg:
    def __init__(self,windows_length):
        self.windows_length = windows_length
        self._filt_window = []
        self._current_sum = 0
    
    def filter_dat(self,input_dat):
        if(len(self._filt_window)<self.windows_length):
            self._current_sum = self._current_sum + input_dat
            self._filt_window.append(input_dat)
            
        else:            
            self._current_sum = self._current_sum - self._filt_window[0]
            self._filt_window.pop(0)

            self._current_sum = self._current_sum + input_dat
            self._filt_window.append(input_dat)
        
        output = self._current_sum/self.windows_length
        
        return output



# -------------------------------------------------
# Helpers
# -------------------------------------------------
def parse_program_arguments(description, def_nframes, def_frate):
    # Parse all program attributes
    # description:   describes program
    # def_nframes:   default number of frames
    # def_frate:     default frame rate in Hz

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-n', '--nframes', type=int,
                        default=def_nframes, help="number of frames, default " + str(def_nframes))
    parser.add_argument('-f', '--frate', type=int, default=def_frate,
                        help="frame rate in Hz, default " + str(def_frate))

    return parser.parse_args()

import json

# Opening JSON file
f = open("ifx_vitalsign_conf_64c_200Hz_2.json")

# returns JSON object as
# a dictionary
data_sample_config = json.load(f)

# Iterating through the json
# list
# for i in data['emp_details']:
#     print(i)

# Closing file
f.close()
pretty_json = json.dumps(data_sample_config, indent=2)
print(pretty_json)

# def filter_moving_average(dat_arr):
#     len_dat = len(dat_arr)
#     for

is_radar_running = False
windows_length = 200
global distance_data_all_antennas_length
distance_data_all_antennas_length = 0
num_rx_antennas = 3
windows = np.zeros([num_rx_antennas,windows_length])
filtered_ma_windows = np.zeros([num_rx_antennas,windows_length])
filtered_10_ma_windows = np.zeros([num_rx_antennas,windows_length])
heart_signal =np.zeros(windows_length)
global breath_signal
breath_signal =np.zeros(windows_length)

current_time = datetime.datetime.now()

def radar_running(device,algo,filternya,num_rx_antennas):
    print("Begin Radar Running")

    while(is_radar_running):
        current_time = datetime.datetime.now()
        frame_contents = device.get_next_frame()
        frame_data = frame_contents[0]

        for i_ant in range(0, num_rx_antennas):  # for each antenna
            antenna_samples = frame_data[i_ant, :, :]
            distance_peak_m, distance_data = algo.compute_distance(antenna_samples)

            # Step 4 - peak search and distance calculation
            skip = 8
            distance_peak = np.argmax(distance_data[skip:])
            magnitude = distance_data[distance_peak + skip]

            distance_data_all_antennas[i_ant] = distance_data

            # Without Filter
            # windows = windows[1:]
            # windows = np.append(windows,[magnitude])
            # windows_all.append(windows)

            # With Filter
            # windows = windows[1:]
            # datanya = filternya.filter_dat(magnitude)
            # windows = np.append(windows,[datanya])
            # windows_all.append(windows)
            windows[i_ant] = np.roll(windows[i_ant],-1)
            filtered_ma_windows[i_ant] = np.roll(filtered_ma_windows[i_ant],-1)
            datanya = filternya.filter_dat(magnitude)

            # windows[i_ant][-1] = datanya
            filtered_ma_windows[i_ant][-1] = datanya
            windows[i_ant][-1] = magnitude
        
            # print("Distance antenna # " + str(i_ant) + ": " + format(distance_peak_m, "^05.3f") + "m Magnitude: "+format(magnitude, "^05.9f"))
        delta_time = datetime.datetime.now() - current_time
        # print(delta_time)
        
    print("Ending Radar Running")

def heart_signal_running():
    # HEart Filter
    fs = 5  # 采样率为20Hz，即0.05秒的采样间隔
    _heartf1 = 0.8 / (fs/2)  # 归一化通带截止频率
    _heartf2 = 2 / (fs/2)  # 归一化阻带截止频率
    _breathf1 = 0.1 / (fs/2)  # 归一化通带截止频率
    _breathf2 = 0.6 / (fs/2)  # 归一化阻带截止频率
    sos_heart = butter(2, [_heartf1, _heartf2], btype='bandpass', output='sos')
    sos_breath = butter(3, [_breathf1, _breathf2], btype='bandpass', output='sos')
    heartDraw = VitalSignDraw(windows_length,"Heart",fs)
    while(is_radar_running):
        breath_wave = iir_breath(4, windows[0],5,sos_breath)
        heart_wave = iir_heart(8, windows[0],5,sos_heart)
        # heart_signal = np.transpose(heart_wave)
        heart_signal = heart_wave
        breath_signal = breath_wave
        
        

        
        # minmin = min(heart_signal)
        # maxmax = max(heart_signal)
        # print(f"Max Heart：{maxmax}, Min Heart：{minmin}")
        breath_fre = np.abs(np.fft.fftshift(np.fft.fft(breath_wave)))
        heart_fre = np.abs(np.fft.fftshift(np.fft.fft(heart_wave)))
        heart_fre = np.split(heart_fre,2)
        heart_fre = heart_fre[1]
        # print("Size Hearth LEngth")
        # print(heart_wave.size)
        # breath_rate, maxIndexBreathSpect = peakbreath(breath_fre)
        # heart_rate = peakheart(heart_fre, maxIndexBreathSpect)
        dum_points = round(len(heart_fre)/2)
        dum_single_point = fs/(2*dum_points)
        hz2bpm = 60
        hz_val = heart_fre.argmax() *dum_single_point
        bpm_val = hz_val*hz2bpm
        if(hz_val >0.8):
            str_bpm = str(bpm_val) + "bpm"
        else:
            bpm_val = 0.2*hz2bpm
            str_bpm = "<"+str(bpm_val) + "bpm"
        # print(f"Breath Rate：{breath_rate}, Heart Rate：{heart_rate}")
        heartDraw.draw(heart_wave,heart_fre, str_bpm)
    heartDraw.close()
    print("Ending Radar Running")

def breath_signal_running():
    # HEart Filter
    fs = 5  # 采样率为20Hz，即0.05秒的采样间隔
    _breathf1 = 0.01 / (fs/2)  # 归一化通带截止频率
    _breathf2 = 0.6 / (fs/2)  # 归一化阻带截止频率
    sos_breath = butter(8, [_breathf1, _breathf2], btype='bandpass', output='sos')
    Breathdraw = VitalSignDraw(windows_length,"Breath",fs)
    while(is_radar_running):
        # breath_wave = iir_breath(4, filtered_ma_windows[0],5,sos_breath)
        breath_wave = filtered_ma_windows[0]
        

        
        # minmin = min(heart_signal)
        # maxmax = max(heart_signal)
        # print(f"Max Heart：{maxmax}, Min Heart：{minmin}")
        breath_fre = np.abs(np.fft.fftshift(np.fft.fft(breath_wave)))
        breath_fre = np.split(breath_fre,2)
        breath_fre = breath_fre[1]
        breath_fre[0]=0
        breath_fre[1]=0
        # print("Size Hearth LEngth")
        # print(heart_wave.size)
        # breath_rate, maxIndexBreathSpect = peakbreath(breath_fre)
        # heart_rate = peakheart(heart_fre, maxIndexBreathSpect)
        dum_points = round(len(breath_fre)/2)
        dum_single_point = fs/(2*dum_points)
        hz2bpm = 60
        hz_val = breath_fre.argmax() *dum_single_point
        bpm_val = hz_val*hz2bpm
        if(hz_val >0.2):
            str_bpm = str(bpm_val) + "bpm"
        else:
            bpm_val = 0.2*hz2bpm
            str_bpm = "<"+str(bpm_val) + "bpm"
        # locate_max =
        # print(f"Breath Rate：{locate_max}")
        Breathdraw.draw(breath_wave,breath_fre,str_bpm)
    Breathdraw.close()
    print("Ending Breath Signal Running")



# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    args = parse_program_arguments(
        '''Displays distance plot from Radar Data''',
        def_nframes=50,
        def_frate=5)

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        # use all available RX antennas
        # num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]
        

        metrics = FmcwMetrics(
            range_resolution_m=0.05,
            max_range_m=1.6,
            max_speed_m_s=3,
            speed_resolution_m_s=0.2,
            center_frequency_Hz=60_750_000_000,
        )

        config = FmcwSimpleSequenceConfig(
            frame_repetition_time_s=data_sample_config["device_config"]["fmcw_single_shape"]["frame_repetition_time_s"],  # Frame repetition time 0.15s (frame rate of 6.667Hz)
            chirp_repetition_time_s=data_sample_config["device_config"]["fmcw_single_shape"]["chirp_repetition_time_s"],  # Chirp repetition time (or pulse repetition time) of 0.5ms
            num_chirps=data_sample_config["device_config"]["fmcw_single_shape"]["num_chirps_per_frame"],  # chirps per frame
            tdm_mimo=False,  # MIMO disabled
            chirp=FmcwSequenceChirp(
                start_frequency_Hz=data_sample_config["device_config"]["fmcw_single_shape"]["start_frequency_Hz"],  # start frequency: 60 GHz
                end_frequency_Hz=data_sample_config["device_config"]["fmcw_single_shape"]["end_frequency_Hz"],  # end frequency: 61.5 GHz
                sample_rate_Hz=data_sample_config["device_config"]["fmcw_single_shape"]["sample_rate_Hz"],  # ADC sample rate of 1MHz
                num_samples=data_sample_config["device_config"]["fmcw_single_shape"]["num_samples_per_chirp"],  # 64 samples per chirp
                rx_mask=7,  # RX antennas 1 and 3 activated
                tx_mask=1,  # TX antenna 1 activated
                tx_power_level=data_sample_config["device_config"]["fmcw_single_shape"]["tx_power_level"],  # TX power level of 31
                lp_cutoff_Hz=data_sample_config["device_config"]["fmcw_single_shape"]["aaf_cutoff_Hz"],  # Anti-aliasing cutoff frequency of 500kHz
                hp_cutoff_Hz=data_sample_config["device_config"]["fmcw_single_shape"]["hp_cutoff_Hz"],  # 80kHz cutoff frequency for high-pass filter
                if_gain_dB=data_sample_config["device_config"]["fmcw_single_shape"]["if_gain_dB"],  # 33dB if gain
            )
        )

        # create acquisition sequence based on metrics parameters
        sequence = device.create_simple_sequence(config)
        sequence.loop.repetition_time_s = 1 / args.frate  # set frame repetition time

        # convert metrics into chirp loop parameters
        chirp_loop = sequence.loop.sub_sequence.contents
        device.sequence_from_metrics(metrics, chirp_loop)
        metrics = device.metrics_from_sequence(chirp_loop)

        # set remaining chirp parameters which are not derived from metrics
        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        # chirp.sample_rate_Hz = 1_000_000
        # chirp.rx_mask = (1 << num_rx_antennas) - 1
        # chirp.tx_mask = 1
        # chirp.tx_power_level = 31
        # chirp.if_gain_dB = 33
        # chirp.lp_cutoff_Hz = 500000
        # chirp.hp_cutoff_Hz = 80000

        device.set_acquisition_sequence(sequence)
        
        distance_data_all_antennas_length = chirp.num_samples
        distance_data_all_antennas = np.zeros([num_rx_antennas,distance_data_all_antennas_length])
        
        algo = DistanceAlgo(chirp, chirp_loop.loop.num_repetitions)
        # draw = Draw(metrics.max_range_m, num_rx_antennas, chirp.num_samples)
        vitaldraw = VitalDraw(num_rx_antennas, windows_length)

        filternya = Filter_moving_avg(10)
        filternya_10 = Filter_moving_avg(10)
        is_radar_running = True
        radar_thread = threading.Thread(target=radar_running,args=(device,algo,filternya,num_rx_antennas))
        radar_thread.start()
        time.sleep(2)
        
        vital_thread = threading.Thread(target=heart_signal_running)
        vital_thread.start()
        time.sleep(2)

        breath_thread = threading.Thread(target=breath_signal_running)
        breath_thread.start()
        time.sleep(2)

        # for frame_number in range(args.nframes):  # for each frame
        while vitaldraw.is_open():
            if not vitaldraw.is_open():                
                # radar_thread.
                break

            # Without Thread
            # frame_contents = device.get_next_frame()
            # frame_data = frame_contents[0]

            # distance_data_all_antennas = []
            # distance_peak_m_4_all_ant = []
            

            # for i_ant in range(0, num_rx_antennas):  # for each antenna
            #     antenna_samples = frame_data[i_ant, :, :]
            #     distance_peak_m, distance_data = algo.compute_distance(antenna_samples)

            #     # Step 4 - peak search and distance calculation
            #     skip = 8
            #     distance_peak = np.argmax(distance_data[skip:])
            #     magnitude = distance_data[distance_peak + skip]

            #     # Without Filter
            #     # windows = windows[1:]
            #     # windows = np.append(windows,[magnitude])
            #     # windows_all.append(windows)

            #     # With Filter
            #     # windows = windows[1:]
            #     # datanya = filternya.filter_dat(magnitude)
            #     # windows = np.append(windows,[datanya])
            #     # windows_all.append(windows)
            #     windows[i_ant] = np.roll(windows[i_ant],-1)
            #     datanya = filternya.filter_dat(magnitude)
            #     windows[i_ant][-1] = datanya
                
            #     # windows_all.append(windows[i_ant])

            #     # distance_data_all_antennas.append(distance_data)
            #     # distance_peak_m_4_all_ant.append(distance_peak_m)
                
            #     print("Distance antenna # " + str(i_ant) + ": " +
            #           format(distance_peak_m, "^05.3f") + "m Magnitude: "+format(magnitude, "^05.9f"))
            
            # draw.draw(distance_data_all_antennas)
            # vitaldraw.draw(windows_all)

            # draw.draw(distance_data_all_antennas)
            vitaldraw.draw(windows)
            # for index in range(0,len(breath_signal)):
            #     print(breath_signal[index])
            
            

        # draw.close()
        vitaldraw.close()
        # radar_thread.
        # VitalSignDraw.close()
        is_radar_running = False
