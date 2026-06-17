import pylsl
import sounddevice as sd
import pathlib
import time
from absl import flags, app
import random
import subprocess
import pandas as pd
import numpy as np

_REST_DURATION = flags.DEFINE_float("rest_duration",
                                    default=30,
                                    help="Rest duration (s)")
_NUM_REPEATS = flags.DEFINE_integer("num_repeats",
                                    default=3,
                                    help="Num repeats")


def load_audio(path):
    audio = subprocess.run(
        f"ffmpeg -i {path} -ac 2 -ar 48000 -f f32le -".split(" "),
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(audio, np.float32).reshape(-1, 2)


def main(argv):
    del argv
    wavs = sorted(list(map(str, pathlib.Path("audio").glob("*.wav"))))
    lsl_device = pylsl.StreamInlet(pylsl.resolve_streams()[0])
    audio_duration = 15
    num_audio_samples = audio_duration * 48000
    print("starting !")
    measures = []

    try:
        for repeat in range(_NUM_REPEATS.value):
            for audio_path in wavs:
                time.sleep(_REST_DURATION.value)
                audio = load_audio(audio_path)
                start = random.randint(a=0, b=len(audio) - num_audio_samples)
                audio = audio[start:start + num_audio_samples]
                start = time.time()
                sd.play(audio, samplerate=48000)
                while time.time() - start < audio_duration:
                    sample, timestamp = lsl_device.pull_sample(timeout=1.)
                    measures.append(
                        dict(sample=sample,
                             audio_path=audio_path,
                             repeat=repeat,
                             timestamp=timestamp))
                sd.wait()
    except Exception as e:
        print("Something went wrong", e)

    pd.DataFrame(measures).to_csv("measures.csv")


if __name__ == "__main__":
    app.run(main)
