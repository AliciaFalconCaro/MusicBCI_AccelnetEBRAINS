import time
import numpy as np
import pylsl
import csv
import matplotlib.pyplot as plt
import datetime


def main():
    print("Looking for an EEG stream...")
    streams1 = pylsl.resolve_byprop('name', 'X.on-102801-0071')
    streams2 = pylsl.resolve_byprop('name', 'X.on-102106-0035')

    if not streams1 or not streams2:
        print("No EEG stream found. Make sure your LSL device is broadcasting.")
        return

    stream_info1 = streams1[0]
    stream_info2 = streams2[0]
    print(f"Connecting to stream: {stream_info1.name()} and {stream_info2.name()}...")
    inlet1 = pylsl.StreamInlet(stream_info1)
    inlet2 = pylsl.StreamInlet(stream_info2)

    print("Reading data. Press Ctrl+C to stop.")
    srate = int(stream_info1.nominal_srate())
    n_channels = stream_info1.channel_count()

    Window_Seg = 2  # Window size in seconds
    WindowLength = int(Window_Seg * srate)

    n_channelsEEG = n_channels - 3
    total_combined_channels = n_channelsEEG * 2  # 2 subjects combined

    # Initialize separate buffers
    eeg_buffer1 = np.zeros((WindowLength, n_channelsEEG), dtype=np.float32)
    timestamp_buffer1 = np.zeros(WindowLength, dtype=np.float64)

    eeg_buffer2 = np.zeros((WindowLength, n_channelsEEG), dtype=np.float32)
    timestamp_buffer2 = np.zeros(WindowLength, dtype=np.float64)

    buffer_index = 0
    Num_samples = 0

    # Data saving setup
    currentDate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = f"eeg1_data_{currentDate}.csv"
    filename2 = f"eeg2_data_{currentDate}.csv"

    # --- SETUP THE GRAPH ONCE OUTSIDE THE LOOP (NO ION) ---
    y_offset = 100
    fig, ax = plt.subplots(figsize=(12, 8))
    lines = []
    for ch in range(total_combined_channels):
        line, = ax.plot([], [], lw=1)
        lines.append(line)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + Offset")
    ax.set_title("Dual-Subject EEG Live Preview (2s Windows)")
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

                # Save raw samples to CSV
                sample1_np = np.asarray(sample1, dtype=np.float32)
                writer1.writerow([timestamp1] + list(sample1))

                sample2_np = np.asarray(sample2, dtype=np.float32)
                writer2.writerow([timestamp2] + list(sample2))

                # ---Map the correct samples to the correct buffers ---
                eeg_buffer1[buffer_index] = sample1_np[:-3]
                timestamp_buffer1[buffer_index] = timestamp1

                eeg_buffer2[buffer_index] = sample2_np[:-3]  # Fixed from sample1 to sample2
                timestamp_buffer2[buffer_index] = timestamp2

                buffer_index = (buffer_index + 1) % WindowLength

                # --- Only increment once per incoming parallel time-step ---
                Num_samples += 1

                # Limit plot refresh to ~30 FPS
                if time.time() - last_plot_time < 1 / 30:
                    continue
                last_plot_time = time.time()

                # Don't plot until a full 2-second window is gathered
                if Num_samples < WindowLength:
                    continue

                # Reconstruct Subject 1 chronologically
                window_subj1 = np.concatenate((eeg_buffer1[buffer_index:], eeg_buffer1[:buffer_index]), axis=0)
                # Reconstruct Subject 2 chronologically
                window_subj2 = np.concatenate((eeg_buffer2[buffer_index:], eeg_buffer2[:buffer_index]), axis=0)

                # --- Combine side-by-side (Axis 1) ---
                combined_window = np.concatenate((window_subj1, window_subj2), axis=1)

                # Reconstruct continuous timeline (using subject 1 as master clock alignment)
                time_window = np.concatenate(
                    (timestamp_buffer1[buffer_index:], timestamp_buffer1[:buffer_index])).flatten()
                time_window = time_window - time_window[0]

                #Analysis on inter-subject data


                # --- Plot combined_window data dynamically ---
                for ch in range(total_combined_channels):
                    lines[ch].set_data(time_window, combined_window[:, ch] + ch * y_offset)

                ax.set_ylim(
                    np.min(combined_window) - 20,
                    np.max(combined_window) + y_offset * total_combined_channels + 20
                )

                # Repaint without blocking the stream
                plt.pause(0.001)

        except KeyboardInterrupt:
            print("\nStopping...")


if __name__ == '__main__':
    main()