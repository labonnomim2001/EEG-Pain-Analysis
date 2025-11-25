# Keep RAW & CLEAN viewers on screen until you close them
import os, mne

FILE = "ID11.gdf"                       # change to any ID*.gdf
DATA = "/Users/labonnomim/Downloads/pain"
path = os.path.join(DATA, FILE)

print("Loading:", path)
raw = mne.io.read_raw_gdf(path, preload=True)

# light prep to make raw readable (still “raw” for demo)
raw.set_eeg_reference("average")
mont = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(mont, match_case=False, on_missing="ignore")
if raw.times[-1] > 10:
    raw.crop(tmin=5.0)

def clean_copy(r):
    rc = r.copy()
    rc.filter(1., 40., fir_design="firwin")
    try: rc.notch_filter(60.)
    except Exception: pass
    return rc

def split_eo_ec(r):
    EO, EC = None, None
    events, event_id = mne.events_from_annotations(r, verbose=False)
    labels = {v:k for k,v in event_id.items()}
    if len(events) and any(("EO" in labels[e[2]] or "EC" in labels[e[2]]) for e in events):
        sf = r.info["sfreq"]
        eo_on = [e[0]/sf for e in events if "EO" in labels[e[2]]]
        ec_on = [e[0]/sf for e in events if "EC" in labels[e[2]]]
        if eo_on: EO = r.copy().crop(eo_on[0], min(eo_on[0]+300, r.times[-1]))
        if ec_on: EC = r.copy().crop(ec_on[0], min(ec_on[0]+300, r.times[-1]))
    else:
        T = r.times[-1]
        EO = r.copy().crop(0, min(300, T))
        if T > 300: EC = r.copy().crop(300, min(600, T))
    return EO, EC

EO, EC = split_eo_ec(raw)

browsers = []

def open_pair(segment, tag):
    if segment is None:
        print(f"No {tag} segment.")
        return
    raw_view   = segment.copy()
    clean_view = clean_copy(segment)
    b1 = raw_view.plot(  duration=8, n_channels=10, scalings="auto",
                         title=f"{FILE} • {tag} • RAW", block=False)
    b2 = clean_view.plot(duration=8, n_channels=10, scalings="auto",
                         title=f"{FILE} • {tag} • CLEANED (1–40 Hz + notch)", block=False)
    browsers.extend([b1, b2])

open_pair(EO, "Eyes Open (EO)")
open_pair(EC, "Eyes Closed (EC)")

# Keep the windows up until you close them
try:
    from mne.viz.backends.qt import _qt_app_exec
    print("\nRAW and CLEAN viewers are OPEN.")
    print("Arrange them side-by-side (macOS: Window ▸ Tile Window).")
    print("Close all viewer windows when you’re done.")
    _qt_app_exec()  # blocks until all Qt windows are closed
except Exception as e:
    print("Qt event loop not available, polling until windows close…", e)
    import time
    while any(getattr(b, "figure", None) is not None for b in browsers):
        time.sleep(0.2)

print("All viewers closed. Bye!")

