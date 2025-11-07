import os
from glob import glob
from tqdm import tqdm
from mouse_tracker import MouseTracker

IN_DIR = "/data/input"
OUT_DIR = "/data/output"

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def run_batch():
    ensure_dirs()
    vids = [p for p in glob(os.path.join(IN_DIR, "*")) if p.endswith(VIDEO_EXT)]
    if not vids:
        print(f"[INFO] 入力動画が見つかりませんでした: {IN_DIR}")
        return

    tracker = MouseTracker()

    for vp in tqdm(vids, desc="processing"):
        base = os.path.splitext(os.path.basename(vp))[0]
        csv_out  = os.path.join(OUT_DIR, f"{base}_track.csv")
        vid_out  = os.path.join(OUT_DIR, f"{base}_annotated.mp4")

        try:
            tracker.process_video(vp, csv_out, vid_out, annotate=True)
            print(f"[OK] {os.path.basename(vp)} -> CSV:{os.path.basename(csv_out)}, MP4:{os.path.basename(vid_out)}")
        except Exception as e:
            print(f"[ERR] {os.path.basename(vp)} -> {e}")

if __name__ == "__main__":
    run_batch()python src/main.py
    