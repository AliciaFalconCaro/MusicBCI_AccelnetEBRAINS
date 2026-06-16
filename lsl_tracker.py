import numpy as np
import pylsl

class LSLTracker:
    """
    A class to acquire data from an LSL (Lab Streaming Layer) stream
    and return it as a generator, similar to HandTracker.
    """
    
    def __init__(self, stream_type="EEG", stream_name=None):
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.inlet = None
        self._connect()

    def _connect(self):
        print(f"Looking for {self.stream_type} stream...")
        
        # Resolve stream by name or type
        if self.stream_name:
            streams = pylsl.resolve_byprop('name', self.stream_name)
        else:
            streams = pylsl.resolve_byprop('type', self.stream_type)
        
        if not streams:
            raise RuntimeError(f"No LSL stream found (type='{self.stream_type}', name='{self.stream_name}').")
            
        # Create inlet from the first found stream
        self.inlet = pylsl.StreamInlet(streams[0])
        print(f"Connected to stream: {self.inlet.info().name()}")

    def get_inputs(self):
        """
        Generator that yields dict(timestamp: float, sample: np.ndarray)
        """
        if not self.inlet:
            self._connect()
            
        print("Starting LSL input generator. Press Ctrl+C to stop.")
        try:
            while True:
                # pull_sample is a blocking call
                sample, timestamp = self.inlet.pull_sample()
                
                yield {
                    "timestamp": timestamp,
                    "sample": np.array(sample, dtype=np.float32)
                }
        except KeyboardInterrupt:
            print("\nGenerator stopped by user.")
        finally:
            self.close()

    def close(self):
        # LSL inlets do not have an explicit close method, 
        # they are garbage collected.
        self.inlet = None
        print("LSL tracker resources cleared.")
