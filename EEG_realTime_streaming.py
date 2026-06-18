import time
import numpy as np
import pylsl
import csv
import matplotlib.pyplot as plt
from collections import deque

from matplotlib.pyplot import figure


#from pythonosc import udp_client


def main():
    # pd = udp_client.SimpleUDPClient("localhost", 3000)
    #
    # pd.send_message("/mrt2/model", ["/Users/acaillon/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small.mlxfn"])
    # pd.send_message("/mrt2/reset", [])
    # pd.send_message("/mrt2/buffersize", [16384])

    print("Looking for an EEG stream...")
    #streams = pylsl.resolve_byprop('type', 'EEG')
    streams = pylsl.resolve_byprop('name', 'X.on-102801-0071')

    if not streams:
        print(
            "No EEG stream found. Make sure your LSL device is broadcasting.")
        return

    # Create a new inlet to read from the stream
    stream_info = streams[0]
    print(f"Connecting to stream: {stream_info.name()}...")
    inlet = pylsl.StreamInlet(stream_info)

    print("Reading data. Press Ctrl+C to stop.")
    mean = 0
    std = 0
    M2 = 0.0
    num_readings = 0
    srate = int(stream_info.nominal_srate())
    n_channels = stream_info.channel_count()

    Window_Seg = 2  # Window size in seconds
    WindowLength = int(Window_Seg * srate)

    # Initialize fixed-size FIFO buffer (shape: samples x channels)
    eeg_buffer = np.zeros((WindowLength, n_channels), dtype=np.float32)
    timestamp_buffer = np.zeros(WindowLength, dtype=np.float64)
    buffer_index = 0
    Num_samples = 0

    #data saving
    filename = f"eeg_data_{int(time.time())}.csv"

    #real-time plotting in interactive mode
    y_offset = 100 #channel separation for visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    lines=[]

    for ch in range(n_channels):
        line, =ax.plot([],[],lw=1)
        lines.append(line)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("EEG Data Preview (2s windows)")
    last_plot_time = time.time()


    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header: timestamp followed by channel indices
       # writer.writerow(['timestamp'] + [f'ch_{i}' for i in range(n_channels)])
        print(range(n_channels))
        try:
            while True:
                # Get a new sample (and timestamp) from the inlet.
                # This is a blocking call.
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample is None:
                    continue

                sample = np.asarray(sample, dtype=np.float32)
                writer.writerow([timestamp] + sample)
                #print(sample)

                eeg_buffer[buffer_index] = sample
                timestamp_buffer[buffer_index] = timestamp
                buffer_index = (buffer_index + 1) % WindowLength
                Num_samples += 1

                # Window Analysis
                STEP = 1*srate #update every 250 samples (50% overlap)
                #print(Num_samples)
                if Num_samples >= WindowLength and Num_samples % STEP == 0:
                    #print("window in progress")
                    window = np.concatenate((
                        eeg_buffer[buffer_index:],
                        eeg_buffer[:buffer_index]
                    ))

                    time_window = np.concatenate((
                        timestamp_buffer[buffer_index:],
                        timestamp_buffer[:buffer_index]
                    )).flatten()

                    time_window = time_window-[0]

                    #Preprocessing: notch, filtering, re-referencing and DC removal
                    window=window -np.mean(window) #DC removal


                    # Plot each 2 seconds
                    #plt.figure(figsize=(12, 6))
                    for ch in range(n_channels):
                        plt.plot(time_window, window[:, ch] + ch * 50)

                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude + offset")
                    plt.title("EEG Data Preview")
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    plt.show()

        except KeyboardInterrupt:
            print("\nStopping...")


if __name__ == '__main__':
    main()
