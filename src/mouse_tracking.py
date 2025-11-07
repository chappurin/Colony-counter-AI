import cv2
import numpy as np
import pandas as pd
from collections import deque

class MouseTracker:
    """
    単一マウス前提のシンプル自動トラッキング。
    - 背景差分 + 形態学的処理で最大輪郭をマウス候補に
    - 外れ値ガードと指数移動平均(EMA)で位置を平滑化
    - CSV出力: frame, x, y, area, speed_px_per_s
    - 注釈付き動画出力（軌跡・重心・面積）
    """

    def __init__(
        self,
        min_area=100,             # ノイズ除去用の最小面積（px^2）
        max_area=200000,          # 上限（万一のノイズ）
        ema_alpha=0.2,            # 位置平滑化の係数
        trail_len=120,            # 何点分の軌跡を表示するか
        kernel_open=5,            # オープニングのカーネルサイズ
        kernel_close=7,           # クロージングのカーネルサイズ
        fg_history=500,           # 背景学習フレーム数
        fg_thresh=16,             # 背景差分の閾値
        detect_shadows=True       # 影検出（有効推奨）
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.ema_alpha = ema_alpha
        self.trail_len = trail_len

        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=fg_history, varThreshold=fg_thresh, detectShadows=detect_shadows
        )

        self.last_ema = None
        self.trail = deque(maxlen=trail_len)
        self.prev_xy = None

    def _ema(self, pt):
        if self.last_ema is None:
            self.last_ema = pt
        else:
            ax = self.ema_alpha
            self.last_ema = (ax*pt[0] + (1-ax)*self.last_ema[0], ax*pt[1] + (1-ax)*self.last_ema[1])
        return self.last_ema

    def _mask_postprocess(self, fgmask):
        # 影(= 127)は0に、前景(= 255)のみ残す
        fgmask = np.where(fgmask == 255, 255, 0).astype(np.uint8)
        # ノイズ抑制：オープン -> クローズ
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.open_kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.close_kernel, iterations=1)
        return fgmask

    def _largest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in cnts:
            a = cv2.contourArea(c)
            if self.min_area <= a <= self.max_area and a > best_area:
                best = c
                best_area = a
        return best, best_area

    def _centroid(self, contour):
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        return (cx, cy)

    def process_video(self, input_path, csv_path, out_video_path, annotate=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # フォールバック
            fps = 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 出力動画
        writer = None
        if annotate:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # 結果レコード
        rows = []
        frame_idx = 0
        self.last_ema = None
        self.trail.clear()
        self.prev_xy = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fg = self.bg.apply(gray)
            fg = self._mask_postprocess(fg)

            cnt, area = self._largest_contour(fg)
            cx, cy = (None, None)
            spd = None

            if cnt is not None:
                ctr = self._centroid(cnt)
                if ctr is not None:
                    cx, cy = self._ema(ctr)
                    self.trail.append((int(cx), int(cy)))

                    if self.prev_xy is not None:
                        dx = cx - self.prev_xy[0]
                        dy = cy - self.prev_xy[1]
                        # px/秒
                        spd = ( (dx**2 + dy**2) ** 0.5 ) * fps
                    self.prev_xy = (cx, cy)

            rows.append({
                "frame": frame_idx,
                "x": None if cx is None else float(cx),
                "y": None if cy is None else float(cy),
                "area": float(area) if cnt is not None else None,
                "speed_px_per_s": None if spd is None else float(spd)
            })

            if annotate:
                vis = frame.copy()
                # マスク輪郭
                if cnt is not None:
                    cv2.drawContours(vis, [cnt], -1, (0, 255, 255), 2)
                # 重心
                if cx is not None and cy is not None:
                    cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                # 軌跡
                for i in range(1, len(self.trail)):
                    cv2.line(vis, self.trail[i-1], self.trail[i], (255, 0, 0), 2)
                # テキスト
                info = []
                info.append(f"frame: {frame_idx}")
                if cx is not None:
                    info.append(f"x,y: ({int(cx)},{int(cy)})")
                if area is not None:
                    info.append(f"area: {int(area)}")
                if spd is not None:
                    info.append(f"speed(px/s): {int(spd)}")
                y0 = 24
                for t in info:
                    cv2.putText(vis, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 50), 2, cv2.LINE_AA)
                    y0 += 24

                writer.write(vis)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        pd.DataFrame(rows).to_csv(csv_path, index=False)