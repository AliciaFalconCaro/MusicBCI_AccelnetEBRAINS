import time
import numpy as np
import pylsl
from pythonosc import udp_client


def main():
    pd = udp_client.SimpleUDPClient("localhost", 3000)


    
    print("Looking for an EEG stream...")
    streams = pylsl.resolve_byprop('type', 'EEG')

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
    try:
        while True:
            # Get a new sample (and timestamp) from the inlet.
            # This is a blocking call.
            sample, timestamp = inlet.pull_sample()
            sample = np.asarray(sample, dtype=float)

            num_readings += 1

            # Welford's online algorithm for calculating mean and variance
            delta = sample - mean
            mean += delta / num_readings
            delta2 = sample - mean

            M2 += delta * delta2

            if num_readings > 1:
                variance = M2 / num_readings  # Use (num_readings - 1) for sample variance
                std = np.sqrt(variance)

                # Safely divide to avoid zero-division errors on early samples or flatlines
                sample = np.divide(
                    sample - mean,
                    std,
                    out=np.zeros_like(sample),
                    where=std != 0,
                )
            else:
                # Cannot standardize a single data point; mean is the point itself
                sample = np.zeros_like(sample)

            pd.send_message("/samples", [float(s) for s in sample])
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == '__main__':
    main()
