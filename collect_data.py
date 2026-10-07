"""
collect_data.py — Collect hand-gesture landmark samples from your webcam
========================================================================
    python collect_data.py                    # 150 samples of every gesture
    python collect_data.py --fill 100         # top up every class to at least 100
    python collect_data.py --gestures I B     # only these gestures
    python collect_data.py --samples 50 --gestures "Hello"

Controls (in the camera window)
    SPACE   save one sample
    A       toggle auto-capture (saves every few frames while a hand is visible —
            move your hand around a little for variety)
    N       skip to the next gesture
    Q       save and quit

Samples are appended to data/gesture_data.json and saved after every gesture,
so quitting early never loses work. Run `python train.py` afterwards.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signlang.config import DATA_PATH, GESTURES  # noqa: E402
from signlang.landmarks import HandDetector, draw_hand  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gestures", nargs="+", default=None,
                   help="gestures to collect (default: all in signlang/config.py)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--samples", type=int, default=150, help="samples to add per gesture")
    g.add_argument("--fill", type=int, default=None,
                   help="collect only what's missing so each gesture reaches this many")
    p.add_argument("--auto-every", type=int, default=3,
                   help="in auto-capture mode, save every Nth frame (default 3)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--data", default=str(DATA_PATH))
    return p.parse_args()


def save(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def overlay(frame, gesture, count, target, total_for_gesture, auto, hand_ok):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 84), (60, 31, 13), -1)
    cv2.addWeighted(bar, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, f"Gesture: {gesture}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 195, 79), 2)
    cv2.putText(frame, f"This session: {count}/{target}   (dataset total: {total_for_gesture})",
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, "SPACE=save  A=auto  N=next  Q=quit", (w - 330, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 143, 91), 1)
    if auto:
        cv2.circle(frame, (w - 25, 28), 9, (0, 0, 255), -1)
        cv2.putText(frame, "AUTO", (w - 85, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    pct = int(count / max(target, 1) * (w - 30))
    cv2.rectangle(frame, (15, h - 40), (w - 15, h - 25), (80, 50, 30), -1)
    cv2.rectangle(frame, (15, h - 40), (15 + pct, h - 25), (247, 195, 79), -1)
    msg, col = ("Hand detected", (80, 175, 76)) if hand_ok else ("No hand detected", (54, 67, 244))
    cv2.putText(frame, msg, (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def main():
    args = parse_args()
    data_path = Path(args.data)
    all_data = json.loads(data_path.read_text()) if data_path.exists() else []
    counts = Counter(d["label"] for d in all_data)

    gestures = args.gestures or GESTURES
    if args.fill is not None:
        plan = {g: max(0, args.fill - counts[g]) for g in gestures}
    else:
        plan = {g: args.samples for g in gestures}
    plan = {g: n for g, n in plan.items() if n > 0}

    print(f"Dataset: {data_path} ({len(all_data)} samples)")
    for g in gestures:
        print(f"  {g:<10} have {counts[g]:>4}   collect {plan.get(g, 0):>4}")
    if not plan:
        print("\nNothing to collect — every gesture already meets the target.")
        return

    detector = HandDetector(max_hands=1)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit("Could not open the webcam.")

    quit_all = False
    try:
        for idx, (gesture, target) in enumerate(plan.items(), 1):
            print(f"\n[{idx}/{len(plan)}] Show '{gesture}' — need {target} samples")
            count, frame_i, auto = 0, 0, False
            while count < target:
                ok, frame = cap.read()
                if not ok:
                    quit_all = True
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = detector.detect(rgb)
                hand = hands[0] if hands else None
                frame_i += 1

                if hand is not None:
                    draw_hand(frame, hand, line_color=(214, 91, 91), point_color=(94, 197, 34))
                    if auto and frame_i % args.auto_every == 0:
                        all_data.append({"label": gesture, "landmarks": hand.features.tolist()})
                        count += 1
                overlay(frame, gesture, count, target, counts[gesture] + count, auto, hand is not None)
                cv2.imshow("SignLang AI - Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and hand is not None:
                    all_data.append({"label": gesture, "landmarks": hand.features.tolist()})
                    count += 1
                elif key == ord("a"):
                    auto = not auto
                elif key == ord("n"):
                    break
                elif key == ord("q"):
                    quit_all = True
                    break
            counts[gesture] += count
            save(data_path, all_data)
            print(f"  saved {count} -> '{gesture}' now has {counts[gesture]}")
            if quit_all:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        save(data_path, all_data)

    print(f"\nDone. {len(all_data)} samples in {data_path}")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
