import time
import numpy as np
import pylsl
import csv
import matplotlib.pyplot as plt
import datetime
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
    streams1 = pylsl.resolve_byprop('name', 'X.on-102801-0035') #it was 71: clara
    streams2 = pylsl.resolve_byprop('name', 'X.on-102106-0071') #it was 35: lianne on thursday

#on friday, phoebe on 35, Alicia on 71
    if not streams1 or not streams2:
        print(
            "No EEG stream found. Make sure your LSL device is broadcasting.")
        return

    # Create a new inlet to read from the stream
    stream_info1 = streams1[0]
    stream_info2 = streams2[0]
    print(f"Connecting to stream: {stream_info1.name()} and {stream_info2.name()}...")
    inlet1 = pylsl.StreamInlet(stream_info1)
    inlet2 = pylsl.StreamInlet(stream_info2)

    print("Reading data. Press Ctrl+C to stop.")
    num_readings = 0
    srate = int(stream_info1.nominal_srate())
    n_channels = stream_info1.channel_count()

    Window_Seg = 2  # Window size in seconds
    WindowLength = int(Window_Seg * srate)

    # Initialize fixed-size FIFO buffer (shape: samples x channels)
    n_channelsEEG=n_channels -3
    eeg_buffer = np.zeros((WindowLength, n_channelsEEG), dtype=np.float32)
    timestamp_buffer = np.zeros(WindowLength, dtype=np.float64)
    buffer_index = 0
    Num_samples = 0

    #data saving
    currentDate=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = f"eeg1_data_{currentDate}.csv"
    filename2 = f"eeg2_data_{currentDate}.csv"

    #real-time plotting in interactive mode
    y_offset = 100 #channel separation for visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    lines=[]
    #
    for ch in range(n_channels):
         line, =ax.plot([],[],lw=1)
         lines.append(line)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("EEG Data Preview (2s windows)")
    ax.set_xlim(0, Window_Seg)
    last_plot_time = time.time()


    with open(filename1, mode='w', newline='') as file1, \
         open (filename2, mode='w', newline='') as file2:

        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)

        # Write header: timestamp followed by channel indices
       # writer.writerow(['timestamp'] + [f'ch_{i}' for i in range(n_channels)])
        print(range(n_channels))
        try:
            while True:
                # Get a new sample (and timestamp) from the inlet.
                # This is a blocking call.
                sample1, timestamp1 = inlet1.pull_sample(timeout=0.1)
                sample2, timestamp2 = inlet2.pull_sample(timeout=0.1)
                if sample1 is None or sample2 is None:
                    continue

                sample1 = np.asarray(sample1, dtype=np.float32)
                writer1.writerow([timestamp1] + list(sample1))

                sample2 = np.asarray(sample2, dtype=np.float32)
                writer2.writerow([timestamp2] + list(sample2))
                #print(sample1)
                #print(sample2)

                eeg_buffer[buffer_index] = sample1[:-3]
                timestamp_buffer[buffer_index] = timestamp1
                buffer_index = (buffer_index + 1) % WindowLength
                Num_samples += 1

                # Window Analysis
                STEP = 1*srate #update every 250 samples (50% overlap)
                print(Num_samples)

                if time.time() - last_plot_time < 1/30: #plotting at 30FPS
                     continue

                last_plot_time = time.time()

                if Num_samples < WindowLength:
                    continue

                window = np.concatenate((
                        eeg_buffer[buffer_index:],
                        eeg_buffer[:buffer_index]
                ))

                time_window = np.concatenate((
                        timestamp_buffer[buffer_index:],
                        timestamp_buffer[:buffer_index]
                ))

                time_window = time_window - time_window[0]

                # Update graph data structures
                for ch in range(n_channelsEEG):
                    lines[ch].set_data(time_window, window[:, ch] + ch * y_offset)

                # Set Y dynamic range based on incoming signal variations
                ax.set_ylim(
                    np.min(window) - 20,
                    np.max(window) + y_offset * n_channelsEEG + 20
                )

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)  # Yields execution momentarily to let OS refresh the GUI window

                #using multiple plotting instead of iterative
                    #print("window in progress")
                    # window = np.concatenate((
                    #     eeg_buffer[buffer_index:],
                    #     eeg_buffer[:buffer_index]
                    # ))
                    #
                    # time_window = np.concatenate((
                    #     timestamp_buffer[buffer_index:],
                    #     timestamp_buffer[:buffer_index]
                    # )).flatten()
                    #
                    # time_window = time_window-[0]

                    #Preprocessing: notch, filtering, re-referencing and DC removal
                    #window=window -np.mean(window) #DC removal

                    # Plot each 2 seconds
                    #plt.figure(figsize=(12, 6))
                    # for ch in range(n_channels):
                    #     plt.plot(time_window, window[:, ch] + ch * 50)
                    #
                    # plt.xlabel("Time (s)")
                    # plt.ylabel("Amplitude + offset")
                    # plt.title("EEG Data Preview")
                    # plt.legend(loc="upper right")
                    # plt.tight_layout()
                    # plt.show()

        except KeyboardInterrupt:
            print("\nStopping...")


if __name__ == '__main__':
    main()
