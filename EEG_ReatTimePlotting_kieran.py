import numpy as np
import matplotlib
#matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from pylsl import resolve_byprop, StreamInlet
import time

# --- Config ---
BANDS = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30), "gamma": (30,48)}
BAND_NAMES = list(BANDS.keys())
WINDOW_SEC = 2.0
REFRESH_SEC = 0.2
NAME_1 = 'X.on-102106-0035'
NAME_2 = 'X.on-102801-0071'
EEG_CH = None
PLV_BAND = "alpha"

# --- Connect to BOTH ---
print("Resolving streams...")
s1 = resolve_byprop('name', NAME_1, timeout=10)
s2 = resolve_byprop('name', NAME_2, timeout=10)
if not s1 or not s2:
    raise RuntimeError("Could not find both EEG streams.")
inlet1, inlet2 = StreamInlet(s1[0]), StreamInlet(s2[0])
info = inlet1.info()
FS = int(info.nominal_srate())
N_CH = info.channel_count()
WIN = int(WINDOW_SEC * FS)
if EEG_CH is None:
    EEG_CH = list(range(N_CH))
NC = len(EEG_CH)

ch_labels = []
try:
    ch = info.desc().child("channels").child("channel")
    for i in range(N_CH):
        ch_labels.append(ch.child_value("label") or f"ch{i}")
        ch = ch.next_sibling()
except Exception:
    ch_labels = [f"ch{i}" for i in range(N_CH)]
eeg_labels = [ch_labels[i] for i in EEG_CH]
print(f"Connected @ {FS} Hz | using EEG channels: {eeg_labels}")

# --- Processing ---
def bandpass(data, lo, hi, fs, order=4):
    sos = signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)

def pull_aligned():
    max_s = int(WINDOW_SEC * FS * 1.5)
    a1, t1 = inlet1.pull_chunk(timeout=0.2, max_samples=max_s)
    a2, t2 = inlet2.pull_chunk(timeout=0.2, max_samples=max_s)
    return (np.asarray(a1), np.asarray(t1)), (np.asarray(a2), np.asarray(t2))

def align(buf1, t1, buf2, t2):
    """Align two buffers onto a common time grid. Returns (d1, d2) or (None, None)."""
    if len(t1) < 2 or len(t2) < 2:
        return None, None
    try:
        t1 = t1 + inlet1.time_correction()
        t2 = t2 + inlet2.time_correction()
    except Exception:
        pass
    ts, te = max(t1[0], t2[0]), min(t1[-1], t2[-1])
    if te - ts < WINDOW_SEC * 0.5:
        return None, None
    n = int((te - ts) * FS)
    grid = np.linspace(ts, te, n)
    d1 = np.stack([np.interp(grid, t1, buf1[:, c]) for c in EEG_CH], axis=1)
    d2 = np.stack([np.interp(grid, t2, buf2[:, c]) for c in EEG_CH], axis=1)
    return d1, d2

def cross_plv(d1, d2, band):
    lo, hi = band
    b1 = bandpass(d1, lo, hi, FS); b2 = bandpass(d2, lo, hi, FS)
    p1 = np.angle(hilbert(b1, axis=0)); p2 = np.angle(hilbert(b2, axis=0))
    M = np.zeros((NC, NC))
    for i in range(NC):
        for j in range(NC):
            M[i, j] = np.abs(np.mean(np.exp(1j * (p1[:, i] - p2[:, j]))))
    return M

# --- Figure layout: raw1 | raw2 | PLV heatmap | band bar ---
plt.ion()
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
ax_raw1 = fig.add_subplot(gs[0, 0])
ax_raw2 = fig.add_subplot(gs[0, 1])
ax_plv  = fig.add_subplot(gs[0, 2])
ax_bar  = fig.add_subplot(gs[1, :])

# Raw EEG: stacked offset traces, one per channel
OFFSET = 100  # vertical spacing between channels (tune to your signal amplitude)
raw1_lines = [ax_raw1.plot([], [])[0] for _ in range(NC)]
raw2_lines = [ax_raw2.plot([], [])[0] for _ in range(NC)]
for ax, title in [(ax_raw1, "Device 1 — raw EEG"), (ax_raw2, "Device 2 — raw EEG")]:
    ax.set_title(title); ax.set_yticks([i*OFFSET for i in range(NC)])
    ax.set_yticklabels(eeg_labels); ax.set_xticks([])

# PLV heatmap
plv_im = ax_plv.imshow(np.zeros((NC, NC)), vmin=0, vmax=1, cmap='viridis', aspect='auto')
ax_plv.set_title(f"Cross-brain PLV ({PLV_BAND})")
ax_plv.set_xlabel("Dev2 ch"); ax_plv.set_ylabel("Dev1 ch")
ax_plv.set_xticks(range(NC)); ax_plv.set_xticklabels(eeg_labels, rotation=90, fontsize=7)
ax_plv.set_yticks(range(NC)); ax_plv.set_yticklabels(eeg_labels, fontsize=7)
fig.colorbar(plv_im, ax=ax_plv, label="PLV")

# Per-band sync bar
bars = ax_bar.bar(BAND_NAMES, [0]*len(BANDS))
ax_bar.set_ylim(0, 1); ax_bar.set_title("Per-band homologous sync strength")
fig.tight_layout()
plt.show(block=False)

# --- Buffers ---
buf1 = np.zeros((0, N_CH), dtype=np.float32); tbuf1 = np.zeros(0)
buf2 = np.zeros((0, N_CH), dtype=np.float32); tbuf2 = np.zeros(0)

print("Running — close the window to stop.")
while plt.fignum_exists(fig.number):
    (a1, t1), (a2, t2) = pull_aligned()
    if a1.shape[0]:
        buf1 = np.concatenate([buf1, a1], axis=0); tbuf1 = np.concatenate([tbuf1, t1])
        buf1, tbuf1 = buf1[-WIN:], tbuf1[-WIN:]
    if a2.shape[0]:
        buf2 = np.concatenate([buf2, a2], axis=0); tbuf2 = np.concatenate([tbuf2, t2])
        buf2, tbuf2 = buf2[-WIN:], tbuf2[-WIN:]
    if buf1.shape[0] < WIN // 2 or buf2.shape[0] < WIN // 2:
        plt.pause(REFRESH_SEC); continue

    d1, d2 = align(buf1, tbuf1, buf2, tbuf2)
    if d1 is None:
        plt.pause(REFRESH_SEC); continue

    # Raw traces (band-passed for display), stacked with offsets
    disp1 = bandpass(d1, 1.0, 48.0, FS); disp2 = bandpass(d2, 1.0, 48.0, FS)
    x = np.arange(disp1.shape[0])
    for i in range(NC):
        raw1_lines[i].set_data(x, disp1[:, i] + i*OFFSET)
        raw2_lines[i].set_data(x, disp2[:, i] + i*OFFSET)
    ax_raw1.set_xlim(0, len(x)); ax_raw1.set_ylim(-OFFSET, NC*OFFSET)
    ax_raw2.set_xlim(0, len(x)); ax_raw2.set_ylim(-OFFSET, NC*OFFSET)

    # PLV per band: heatmap for one, strengths for the bar
    strengths = []
    for name, band in BANDS.items():
        M = cross_plv(d1, d2, band)
        strengths.append(float(np.mean(np.diag(M))))
        if name == PLV_BAND:
            plv_im.set_data(M)
    for b, h in zip(bars, strengths):
        b.set_height(h)

    fig.canvas.draw_idle()
    plt.pause(REFRESH_SEC)

print("Window closed, exiting.")