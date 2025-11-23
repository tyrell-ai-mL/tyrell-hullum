import os
import glob
import cv2

# ---------- CONFIG ----------
IMG_DIR = "frames_rainynight"       # your frames
LBL_DIR = "labels"                  # YOLO txt labels
OUT_VIDEO = "overlay_rainynight.mp4"

# ↓ Slower playback: use 10–15 FPS instead of 30
FPS = 8

# Class mapping for this project:
# 0 = car, 1 = person, 2 = bus
NAMES = {
    0: "car",
    1: "person",
    2: "bus",
}

# Colors (BGR): car=purple, bus=orange, person=green
COLORS = {
    0: (255, 0, 255),   # car
    1: (0, 255, 0),     # person
    2: (0, 165, 255),   # bus
}

# Box + text style
BOX_THICKNESS = 1       # thinner boxes
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
# -----------------------------


def main():
    image_files = sorted(
        f for f in glob.glob(os.path.join(IMG_DIR, "*"))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Frames found: {len(image_files)}")
    if not image_files:
        print("❌ No frames found. Check IMG_DIR.")
        return

    first = cv2.imread(image_files[0])
    if first is None:
        print(f"❌ Could not read first frame: {image_files[0]}")
        return
    h, w = first.shape[:2]
    print(f"Frame size: {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (w, h))

    frames_with_boxes = 0

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, base + ".txt")
        has_box = False

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    # YOLO: class x_center y_center width height (0–1)
                    cls = int(float(parts[0]))
                    x_c, y_c, bw, bh = map(float, parts[1:])

                    # normalized -> pixel coords
                    x_c *= w
                    y_c *= h
                    bw  *= w
                    bh  *= h

                    x1 = int(x_c - bw / 2)
                    y1 = int(y_c - bh / 2)
                    x2 = int(x_c + bw / 2)
                    y2 = int(y_c + bh / 2)

                    # clip inside frame
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    color = COLORS.get(cls, (0, 255, 255))
                    label = NAMES.get(cls, str(cls))

                    # thin box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

                    # simple label text above box
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 2, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        TEXT_SCALE,
                        color,
                        TEXT_THICKNESS,
                        cv2.LINE_AA,
                    )
                    has_box = True

        if has_box:
            frames_with_boxes += 1

        writer.write(frame)

    writer.release()
    print(f"✅ Frames with at least one box: {frames_with_boxes}")
    print(f"✅ Saved video as: {OUT_VIDEO}")


if __name__ == "__main__":
    main()




