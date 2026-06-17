import pylsl
import sounddevice as sd
import pathlib
import time
from absl import flags, app
import random
import subprocess
import pandas as pd
import numpy as np
import tqdm

import os

_REST_DURATION = flags.DEFINE_float("rest_duration",
                                    default=20,
                                    help="Rest duration (s)")
_NUM_REPEATS = flags.DEFINE_integer("num_repeats",
                                    default=3,
                                    help="Num repeats")
_NAME = flags.DEFINE_string(
    "name",
    default=None,
    required=True,
    help="Experiment name",
)


class DummyDevice:

    def pull_sample(self, *args, **kwargs):
        time.sleep(.1)
        return [1, 2, 3], 0.


def apply_fade(audio, samplerate=48000, fade_duration=2):
    num_samples_fade = int(fade_duration * samplerate)

    # Create fade-in envelope
    fade_in = np.linspace(0, 1, num_samples_fade)

    # Create fade-out envelope
    fade_out = np.linspace(1, 0, num_samples_fade)

    # Apply fade-in
    audio[:num_samples_fade] *= fade_in[:, np.newaxis]

    # Apply fade-out
    audio[-num_samples_fade:] *= fade_out[:, np.newaxis]

    return audio


def load_audio(path):
    audio = subprocess.run(
        f"ffmpeg -i {path} -ac 2 -ar 48000 -f f32le -".split(" "),
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(audio, np.float32).reshape(-1, 2).copy()


def main(argv):
    del argv
    os.makedirs("recordings", exist_ok=True)
    wavs = sorted(list(map(str, pathlib.Path("audio").glob("*.wav"))))

    streams = pylsl.resolve_streams()
    if not streams:
        print("Using dummy device")
        time.sleep(1)
        lsl_device = DummyDevice()

    if len(streams) == 1:
        stream_info = streams[0]
        lsl_device = pylsl.StreamInlet(stream_info)
        print(stream_info)
    else:
        for s in streams:
            print(s)

        index = int(input("select device: "))
        stream_info = streams[index]
        lsl_device = pylsl.StreamInlet(stream_info)

    audio_duration = 15
    num_audio_samples = audio_duration * 48000
    output_file = pathlib.Path(f"recordings/{_NAME.value}.csv")
    first_write = not output_file.exists()
    print("starting !")
    pbar = tqdm.tqdm()

    try:
        for repeat in range(_NUM_REPEATS.value):
            for audio_path in wavs:
                time.sleep(_REST_DURATION.value)
                audio = load_audio(audio_path)

                start = random.randint(a=0, b=len(audio) - num_audio_samples)
                audio = audio[start:start + num_audio_samples]
                audio = apply_fade(audio)

                start = time.time()
                sd.play(audio, samplerate=48000)
                current_measures = []
                while time.time() - start < audio_duration:
                    sample, timestamp = lsl_device.pull_sample(timeout=1.)
                    pbar.update()
                    current_measures.append(
                        dict(sample=sample,
                             audio_path=audio_path,
                             repeat=repeat,
                             timestamp=timestamp), )
                sd.wait()
                pd.DataFrame(current_measures).to_csv(output_file,
                                                      mode='a',
                                                      header=first_write,
                                                      index=False)
                first_write = False
    except Exception as e:
        print("Something went wrong", e)


if __name__ == "__main__":
    app.run(main)
