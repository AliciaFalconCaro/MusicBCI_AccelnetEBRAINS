import pandas as pd
import pylsl
import tqdm
from absl import app, flags

_NAME = flags.DEFINE_string(
    "name",
    default=None,
    help="Name of the recording",
    required=True,
)


def main(argv):
    del argv
    streams = pylsl.resolve_streams()

    if not streams:
        print(
            "No EEG stream found. Make sure your LSL device is broadcasting.")
        exit()

    if len(streams) == 1:
        stream_info = streams[0]
        print(stream_info)
    else:
        for s in streams:
            print(s)

    inlet = pylsl.StreamInlet(stream_info)

    inputs = []

    pbar = tqdm.tqdm()

    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            inputs.append(dict(values=sample, timestamp=timestamp))
            pbar.update()
    except:
        print("stopping...")

    pd.DataFrame(inputs).to_csv(_NAME.value + ".csv")


if __name__ == "__main__":
    app.run(main)
