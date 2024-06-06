import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import pprint
import matplotlib.pyplot as plt
import numpy as np

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from helpers.DigitalBeamForming import *
from helpers.DopplerAlgo import *

import threading

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def num_rx_antennas_from_rx_mask(rx_mask):
    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


class LivePlot:
    def __init__(self, max_angle_degrees: float, max_range_m: float,fignya,axnya):
        # max_angle_degrees: maximum supported speed
        # max_range_m:   maximum supported range
        self.h = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m

        plt.ion()

        # self.fig, self._ax = plt.subplots(nrows=1, ncols=1)
        self.fig = fignya
        self._ax = axnya

        # self.fig.canvas.manager.set_window_title("Range-Angle-Map using Digital Beam Forming")
        self.fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data: np.ndarray):
        # First time draw

        minmin = -60
        maxmax = 0

        self.h = self._ax.imshow(
            data,
            vmin=minmin, vmax=maxmax,
            cmap='viridis',
            extent=(-self.max_angle_degrees,
                    self.max_angle_degrees,
                    0,
                    self.max_range_m),
            origin='lower')

        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")

        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self.fig.colorbar(self.h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data: np.ndarray):
        # Update data for each antenna

        self.h.set_data(data)

    def draw(self, data: np.ndarray, title: str):
        if self._is_window_open:
            if self.h:
                self._draw_next_time(data)
            else:
                self._draw_first_time(data)
            self._ax.set_title(title)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self.fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open

num_beams = 27  # number of beams
max_angle_degrees = 40  # maximum angle, angle ranges from -40 to +40 degrees
# test
config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=0.15,  # Frame repetition time 0.15s (frame rate of 6.667Hz)
    chirp_repetition_time_s=0.0005,  # Chirp repetition time (or pulse repetition time) of 0.5ms
    num_chirps=128,  # chirps per frame
    tdm_mimo=False,  # MIMO disabled
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=60e9,  # start frequency: 60 GHz
        end_frequency_Hz=61.5e9,  # end frequency: 61.5 GHz
        sample_rate_Hz=1e6,  # ADC sample rate of 1MHz
        num_samples=64,  # 64 samples per chirp
        rx_mask=5,  # RX antennas 1 and 3 activated
        tx_mask=1,  # TX antenna 1 activated
        tx_power_level=31,  # TX power level of 31
        lp_cutoff_Hz=500000,  # Anti-aliasing cutoff frequency of 500kHz
        hp_cutoff_Hz=80000,  # 80kHz cutoff frequency for high-pass filter
        if_gain_dB=33,  # 33dB if gain
    )
)



fignya = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
axnya = fignya.subplots(nrows=1, ncols=1)

plot = LivePlot(100,2,fignya,axnya)





# in a real program it's best to use after callbacks instead of
# sleeping in a thread, this is just an example
def acq_loop():
    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        # configure device
        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        # get metrics and print them
        chirp_loop = sequence.loop.sub_sequence.contents
        metrics = device.metrics_from_sequence(chirp_loop)
        pprint.pprint(metrics)

        # get maximum range
        max_range_m = metrics.max_range_m

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        num_rx_antennas = num_rx_antennas_from_rx_mask(chirp.rx_mask)

        # Create objects for Range-Doppler, Digital Beam Forming, and plotting.
        doppler = DopplerAlgo(config.chirp.num_samples, config.num_chirps, num_rx_antennas)
        dbf = DigitalBeamForming(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)
        plot = LivePlot(max_angle_degrees, max_range_m,fignya,axnya)
        close_btn = tk.Button(frm_buttons, text="close", command=plot.close)
        close_btn.grid(row=2, column=0, sticky="ew", padx=5)
        while not plot.is_closed():
            # frame has dimension num_rx_antennas x num_chirps_per_frame x num_samples_per_chirp
            frame_contents = device.get_next_frame()
            frame = frame_contents[0]

            rd_spectrum = np.zeros((config.chirp.num_samples, 2 * config.num_chirps, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros((config.chirp.num_samples, num_beams))

            for i_ant in range(num_rx_antennas):  # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                mat = frame[i_ant, :, :]

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]
                beam_range_energy[:, i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            # Rescale map to better capture the peak The rescaling is done in a
            # way such that the maximum always has the same value, independent
            # on the original input peak. A proper peak search can greatly
            # improve this algorithm.
            scale = 150
            beam_range_energy = scale * (beam_range_energy / max_energy - 1)

            # Find dominant angle of target
            _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[idx]

            # And plot...
            plot.draw(beam_range_energy, f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees")


def start_acq():
    thread = threading.Thread(target=acq_loop)
    thread.daemon = True
    thread.start()


window = tk.Tk()
window.title("Simple Text Editor")

window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

plot_frm = tk.Frame(window)
frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=3)
acq_btn = tk.Button(frm_buttons, text="Acquisition Data",command=start_acq)
acq_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
set_btn = tk.Button(frm_buttons, text="Set Params", )
set_btn.grid(row=1, column=0, sticky="ew", padx=5)

# Radar Parameter Control
lbl_frame_repetition_time_s = tk.Label(frm_buttons,text="Frame Repetition Times(s)")
lbl_frame_repetition_time_s.grid(row=3,column=0, sticky="ew",padx=5, pady=5)
txt_frame_repetition_time_s = tk.Spinbox(frm_buttons)
txt_frame_repetition_time_s.grid(row=4,column=0, sticky="ew",padx=5, pady=5)

lbl_chirp_repetition_time_s = tk.Label(frm_buttons,text="Chirp Repetition Time(s)")
lbl_chirp_repetition_time_s.grid(row=5,column=0, sticky="ew",padx=5, pady=5)
txt_chirp_repetition_time_s = tk.Spinbox(frm_buttons)
txt_chirp_repetition_time_s.grid(row=6,column=0, sticky="ew",padx=5, pady=5)

lbl_num_chirps = tk.Label(frm_buttons,text="Num Chirps")
lbl_num_chirps.grid(row=7,column=0, sticky="ew",padx=5, pady=5)
txt_num_chirps = tk.Spinbox(frm_buttons)
txt_num_chirps.grid(row=8,column=0, sticky="ew",padx=5, pady=5)

lbl_tdm_mimo = tk.Label(frm_buttons,text="TDM MIMO")
lbl_tdm_mimo.grid(row=9,column=0, sticky="ew",padx=5, pady=5)
txt_tdm_mimo = tk.Checkbutton(frm_buttons)
txt_tdm_mimo.grid(row=10,column=0, sticky="ew",padx=5, pady=5)

lbl_start_frequency_Hz = tk.Label(frm_buttons,text="Start_frequency_Hz")
lbl_start_frequency_Hz.grid(row=11,column=0, sticky="ew",padx=5, pady=5)
txt_start_frequency_Hz = tk.Spinbox(frm_buttons)
txt_start_frequency_Hz.grid(row=12,column=0, sticky="ew",padx=5, pady=5)

lbl_end_frequency_Hz = tk.Label(frm_buttons,text="end_frequency_Hz")
lbl_end_frequency_Hz.grid(row=13,column=0, sticky="ew",padx=5, pady=5)
txt_end_frequency_Hz = tk.Spinbox(frm_buttons)
txt_end_frequency_Hz.grid(row=14,column=0, sticky="ew",padx=5, pady=5)

lbl_sample_rate_Hz = tk.Label(frm_buttons,text="sample_rate_Hz")
lbl_sample_rate_Hz.grid(row=15,column=0, sticky="ew",padx=5, pady=5)
txt_sample_rate_Hz = tk.Spinbox(frm_buttons)
txt_sample_rate_Hz.grid(row=16,column=0, sticky="ew",padx=5, pady=5)

lbl_rx_mask = tk.Label(frm_buttons,text="rx_mask")
lbl_rx_mask.grid(row=17,column=0, sticky="ew",padx=5, pady=5)
txt_rx_mask = tk.Spinbox(frm_buttons)
txt_rx_mask.grid(row=18,column=0, sticky="ew",padx=5, pady=5)

lbl_tx_mask = tk.Label(frm_buttons,text="tx_mask")
lbl_tx_mask.grid(row=19,column=0, sticky="ew",padx=5, pady=5)
txt_tx_mask = tk.Spinbox(frm_buttons)
txt_tx_mask.grid(row=20,column=0, sticky="ew",padx=5, pady=5)

lbl_lp_cutoff_Hz = tk.Label(frm_buttons,text="lp_cutoff_Hz")
lbl_lp_cutoff_Hz.grid(row=21,column=0, sticky="ew",padx=5, pady=5)
txt_lp_cutoff_Hz = tk.Spinbox(frm_buttons)
txt_lp_cutoff_Hz.grid(row=22,column=0, sticky="ew",padx=5, pady=5)

lbl_hp_cutoff_Hz = tk.Label(frm_buttons,text="hp_cutoff_Hz")
lbl_hp_cutoff_Hz.grid(row=23,column=0, sticky="ew",padx=5, pady=5)
txt_hp_cutoff_Hz = tk.Spinbox(frm_buttons)
txt_hp_cutoff_Hz.grid(row=24,column=0, sticky="ew",padx=5, pady=5)

lbl_if_gain_dB = tk.Label(frm_buttons,text="if_gain_dB")
lbl_if_gain_dB.grid(row=25,column=0, sticky="ew",padx=5, pady=5)
txt_if_gain_dB = tk.Spinbox(frm_buttons)
txt_if_gain_dB.grid(row=26,column=0, sticky="ew",padx=5, pady=5)




# txt_frame_repetition_time_s.pack()


# frame_repetition_time_s=0.15,  # Frame repetition time 0.15s (frame rate of 6.667Hz)
#     chirp_repetition_time_s=0.0005,  # Chirp repetition time (or pulse repetition time) of 0.5ms
#     num_chirps=128,  # chirps per frame
#     tdm_mimo=False,  # MIMO disabled
#     chirp=FmcwSequenceChirp(
#         start_frequency_Hz=60e9,  # start frequency: 60 GHz
#         end_frequency_Hz=61.5e9,  # end frequency: 61.5 GHz
#         sample_rate_Hz=1e6,  # ADC sample rate of 1MHz
#         num_samples=64,  # 64 samples per chirp
#         rx_mask=5,  # RX antennas 1 and 3 activated
#         tx_mask=1,  # TX antenna 1 activated
#         tx_power_level=31,  # TX power level of 31
#         lp_cutoff_Hz=500000,  # Anti-aliasing cutoff frequency of 500kHz
#         hp_cutoff_Hz=80000,  # 80kHz cutoff frequency for high-pass filter
#         if_gain_dB=33,  # 33dB if gain
#     )





frm_buttons.grid(row=0, column=0, sticky="ns")
plot_frm.grid(row=0, column=1, sticky="nsew")

# plot_lbl = tk.Label(plot_frm,text="Plot")
# plot_lbl.pack()


canvas = FigureCanvasTkAgg(plot.fig, master=plot_frm)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, plot_frm)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

window.mainloop()