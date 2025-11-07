# src/mouse_tracker.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

class MouseTracker:
    def __init__(
        self,
        min_area: int = 200,       # ノイズと区別する面積（ピクセル）
        max_area: int = 50000,     # 過大な領域を除外
        history: int = 300,        # 背景学習フレーム数
        var_threshold: int = 25,   # 前景判定のしきい値
        dilate_iter: int = 2,      # 膨張回数（穴埋め）
        erode_iter: int = 1,       # 収縮回数（細ノイズ除去）
        select: str = "largest"    # "largest" or "centermost"
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.history = history
        self.var_threshold = var_threshold
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        self.select = select

        # 適応型背景差分（＝簡易なAI的前景抽出）
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=True
        )

    def _pick_target(self, contours, frame_shape):
        if not contours:
            return None
        if self.select == "centermost":
            h, w = frame_shape[:2]
            cx, cy = w/2, h/2
            best = None
            best_d = 1e12
            for c in contours:
                area = cv2.contourArea(c)
                if area <= 0: 
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                x = M["m10"]/M["m00"]
                y = M["m01"]/M["m00"]
                d = (x-cx)**2+(y-cy)**2
                if d < best_d:
                    best_d = d
                    best = c
            return best
        else:
            # 最大面積を選択
            return max(contours, key=cv2.contourArea)

    def track_video(self, in_path: Path, out_dir: Path) -> dict:
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            return {"ok": False, "msg": f"Could not open {in_path.name}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 出力パス
        stem = in_path.stem
        csv_path = out_dir / f"{stem}_track.csv"
        mp4_path = out_dir / f"{stem}_annotated.mp4"

        # 動画ライタ（mp4）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))

        rows = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            fg = self.bg.apply(frame)
            # 影(=127)を除外
            _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

            # モルフォロジー
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=self.erode_iter)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)

            # 輪郭抽出
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if self.min_area <= cv2.contourArea(c) <= self.max_area]

            cx = cy = area = np.nan
            target = self._pick_target(cnts, frame.shape)

            if target is not None:
                area = cv2.contourArea(target)
                M = cv2.moments(target)
                if M["m00"] != 0:
                    cx = M["m10"]/M["m00"]
                    cy = M["m01"]/M["m00"]
                    # 可視化
                    x,y,w,h = cv2.boundingRect(target)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)
                    cv2.putText(frame, f"({int(cx)}, {int(cy)})", (x, max(0,y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            writer.write(frame)

            rows.append({
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "x": cx, "y": cy, "area": area, "id": 1
            })
            frame_idx += 1

        cap.release()
        writer.release()

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return {"ok": True, "csv": str(csv_path), "video": str(mp4_path), "frames": frame_idx, "fps": fps}