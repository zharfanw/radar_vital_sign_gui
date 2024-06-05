import argparse
import numpy as np
from scipy import signal
from scipy import constants
from collections import deque
from ifxAvian import Avian
import matplotlib.pyplot as plt
from blit_manager import BlitManager

# from fft_spectrum import fft_spectrum
# from Peakcure import peakcure
# from Diffphase import diffphase
# from IIR_Heart import iir_heart
# from IIR_Breath import iir_breath
# from PeakBreath import peakbreath
# from PeakHeart import peakheart
import matplotlib.pyplot as plt



# sourcery skip: for-index-underscore
if __name__ == '__main__':

    # print(f"Radar SDK Version: {Avian.get_version()}")

    config = Avian.DeviceConfig(
        sample_rate_Hz=2e6,  # ADC sample rate of 2MHz
        rx_mask=1,  # RX antenna 1 activated
        tx_mask=1,  # TX antenna 1 activated
        tx_power_level=31,  # TX power level of 31
        if_gain_dB=33,  # 33dB if gain
        start_frequency_Hz=58e9,  # start frequency: 58.0 GHz
        end_frequency_Hz=63.5e9,  # end frequency: 63.5 GHz
        num_samples_per_chirp=256,  # 256 samples per chirp
        num_chirps_per_frame=1,  # 32 chirps per frame
        chirp_repetition_time_s=0.150,  # Chirp repetition time (or pulse repetition time) of 150us
        # frame_repetition_time_s=1 / args.frate,  # Frame repetition time default 0.005s (frame rate of 200Hz)
        frame_repetition_time_s=0.005,  # Frame repetition time default 0.005s (frame rate of 200Hz)
        mimo_mode="off")  # MIMO disabled

    # connect to an Avian radar sensor
    with Avian.Device() as device:

        # metrics = device.metrics_from_config(config)

        # set device config
        device.set_config(config)
        q = deque()
        plt.ion()
        _fig, _axs = plt.subplots(nrows=1, ncols=1)
        first_time = True
        _pln = []
        axbackground = _fig.canvas.copy_from_bbox(_axs.bbox)

        
        bm = BlitManager(_fig.canvas,_pln)
        plt.show(block=False)
        plt.pause(.1)
        # atasnya = _axs[0].plot(np.arange(256),np.arange(256))
        # bawahnya = _axs[1].plot([3,4])
        while True:
            frame = device.get_next_frame()
            frame = frame[0, 0, :]
            # frame = frame[0]
            # print(frame)
            
            # _fig.canvas.restore_region(axbackground)

            if(first_time):
                _pln.append(_axs.plot(frame))
                first_time = False
            else:
                _pln[0][0].set_ydata(frame)
                # _axs.draw_artist(_axs.patch)
                # _axs.draw_artist(_pln[0][0])
                # _fig.canvas.blit(_axs.bbox)
                bm.update()
            # _fig.canvas.draw()
            # _fig.canvas.flush_events()
