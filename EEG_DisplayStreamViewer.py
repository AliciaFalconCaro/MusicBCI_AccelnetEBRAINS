import time
import numpy as np
import pylsl
import csv
import matplotlib.pyplot as plt
import datetime
from scipy.signal import butter, iirnotch, sosfiltfilt, welch

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def notch_harmonics(data, fs, line_freq=50.0, num_harmonics=3):
    """Applies notch filtering at line frequency and its harmonics."""
    clean_data = np.copy(data)
    nyq = 0.5 * fs
    for i in range(1, num_harmonics + 1):
        freq = line_freq * i
        if freq < nyq:
            # Quality factor Q controls the width of the notch
            b, a = iirnotch(freq, Q=30.0, fs=fs)
            # Apply along axis 0 (time axis)
            clean_data = sosfiltfilt(butter(2, [freq-1, freq+1], btype='bandstop', output='sos', fs=fs), clean_data, axis=0)
    return clean_data

def main():
    print("Looking for an EEG stream...")
    streams1 = pylsl.resolve_byprop('name', 'X.on-102801-0035')
    streams2 = pylsl.resolve_byprop('name', 'X.on-102106-0071')

    if not streams1 or not streams2:
        print("No EEG stream found. Make sure your LSL device is broadcasting.")
        return

    stream_info1 = streams1[0]
    stream_info2 = streams2[0]
    print(f"Connecting to: {stream_info1.name()} and {stream_info2.name()}...")
    inlet1 = pylsl.StreamInlet(stream_info1)
    inlet2 = pylsl.StreamInlet(stream_info2)

    srate = int(stream_info1.nominal_srate())
    n_channels = stream_info1.channel_count()
    n_channelsEEG = n_channels - 3

    Window_Seg = 2
    WindowLength = int(Window_Seg * srate)

    # Pre-build filters for maximum computational speed
    bp_sos = butter_bandpass(1.0, 40.0, srate, order=4)

    # Initialize fixed-size FIFO buffer (shape: samples x kept channels)
    eeg_buffer = np.zeros((WindowLength, n_channelsEEG), dtype=np.float32)
    timestamp_buffer = np.zeros(WindowLength, dtype=np.float64)
    buffer_index = 0
    Num_samples = 0

    # Data saving
    currentDate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = f"eeg1_data_{currentDate}.csv"
    filename2 = f"eeg2_data_{currentDate}.csv"

    # Real-time plotting setup
    y_offset = 80
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    lines = []
    for ch in range(n_channelsEEG):
         line, = ax.plot([], [], lw=1)
         lines.append(line)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Preprocessed EEG Stream Viewer (2s Windows)")
    ax.set_xlim(0, Window_Seg)
    last_plot_time = time.time()

    with open(filename1, mode='w', newline='') as file1, \
         open(filename2, mode='w', newline='') as file2:

        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)

        try:
            while True:
                sample1, timestamp1 = inlet1.pull_sample(timeout=0.1)
                sample2, timestamp2 = inlet2.pull_sample(timeout=0.1)
                if sample1 is None or sample2 is None:
                    continue

                # Write raw data immediately to disk
                sample1_np = np.asarray(sample1, dtype=np.float32)
                writer1.writerow([timestamp1] + list(sample1))
                sample2_np = np.asarray(sample2, dtype=np.float32)
                writer2.writerow([timestamp2] + list(sample2))

                # Update FIFO Buffer with the dropped-channel Stream 1 data
                eeg_buffer[buffer_index] = sample1_np[:-3]
                timestamp_buffer[buffer_index] = timestamp1
                buffer_index = (buffer_index + 1) % WindowLength
                Num_samples += 1

                # Frame-rate limiter (30 FPS max updates)
                if time.time() - last_plot_time < 1/30:
                     continue
                last_plot_time = time.time()

                if Num_samples < WindowLength:
                    continue

                # Reconstruct sorted chronological arrays
                raw_window = np.concatenate((eeg_buffer[buffer_index:], eeg_buffer[:buffer_index]), axis=0)
                time_window = np.concatenate((timestamp_buffer[buffer_index:], timestamp_buffer[:buffer_index]))
                time_window = time_window - time_window[0]

                # =========================================================
                # PREPROCESSING PIPELINE
                # =========================================================
                # 1. DC Removal (Zero-mean the time series)
                proc_window = raw_window - np.mean(raw_window, axis=0)

                # 2. Notch Filter (50Hz + Harmonics at 100Hz, 150Hz)
                proc_window = notch_harmonics(proc_window, fs=srate, line_freq=50.0, num_harmonics=2)

                # 3. Bandpass Filter (1-40 Hz) using zero-phase forward/backward filter
                proc_window = sosfiltfilt(bp_sos, proc_window, axis=0)

                # 4. Common Average Reference (CAR)
                spatial_mean = np.mean(proc_window, axis=1, keepdims=True)
                proc_window = proc_window - spatial_mean

                # =========================================================
                # REAL-TIME FEATURE ANALYSIS: POWER BANDS & CORRELATION
                # =========================================================
                # Compute Power Spectral Density (PSD) via Welch method
                freqs, psd = welch(proc_window, fs=srate, axis=0, nperseg=min(WindowLength, 256))

                # Extract Alpha (8-12 Hz) band power per electrode
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
                alpha_power = np.mean(psd[alpha_idx, :], axis=0)

                # Spatial Pearson Correlation matrix between electrodes
                # np.corrcoef expects channels as rows (shape: channels x samples)
                corr_matrix = np.corrcoef(proc_window.T)

                # Print feature debug update to terminal
                print(f"Mean Electrode Correlation: {np.mean(corr_matrix):.3f} | Alpha Power (Ch0): {alpha_power[0]:.2f}", end="\r")

                # =========================================================
                # UPDATE STREAMVIEWER PLOT
                # =========================================================
                for ch in range(n_channelsEEG):
                    lines[ch].set_data(time_window, proc_window[:, ch] + ch * y_offset)

                ax.set_ylim(
                    np.min(proc_window) - 20,
                    np.max(proc_window) + y_offset * n_channelsEEG + 20
                )

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        except KeyboardInterrupt:
            print("\nStopping...")

if __name__ == '__main__':
    main()