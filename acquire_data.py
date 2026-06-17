import os
import pandas as pd
import pylsl
import tqdm
from absl import app, flags
import pathlib

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

        index = int(input("select device: "))
        stream_info = streams[index]

    inlet = pylsl.StreamInlet(stream_info)

    recordings_dir = pathlib.Path("recordings")
    recordings_dir.mkdir(exist_ok=True)
    output_file = recordings_dir / (_NAME.value + ".csv")

    first_write = not output_file.exists()
    pbar = tqdm.tqdm()

    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            pd.DataFrame([dict(values=sample, timestamp=timestamp)
                          ]).to_csv(output_file,
                                    mode='a',
                                    header=first_write,
                                    index=False)
            first_write = False
            pbar.update()
    except:
        print("stopping...")


if __name__ == "__main__":
    app.run(main)
