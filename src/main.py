# src/main.py
from pathlib import Path
from mouse_tracker import MouseTracker

def run_batch(input_dir="data/input", output_dir="data/output"):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".mp4", ".mov", ".avi", ".mkv")
    videos = [p for p in in_dir.glob("*") if p.suffix.lower() in exts]
    if not videos:
        print(f"[INFO] 入力動画がありません: {in_dir.resolve()}")
        return

    # 初期パラメータ（あとで調整可能）
    tracker = MouseTracker(
        min_area=250, max_area=80000,
        history=300, var_threshold=25,
        dilate_iter=2, erode_iter=1,
        select="largest"  # "centermost" にすると中央に近い対象を優先
    )

    for v in videos:
        print(f"[RUN] {v.name} を解析中...")
        res = tracker.track_video(v, out_dir)
        if res.get("ok"):
            print(f"  -> CSV:  {res['csv']}")
            print(f"  -> 動画: {res['video']}")
            print(f"  -> Frames: {res['frames']} @ {res['fps']:.1f} fps")
        else:
            print(f"[WARN] {res.get('msg','unknown error')}")

if __name__ == "__main__":
    run_batch()