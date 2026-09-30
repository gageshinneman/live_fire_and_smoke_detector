#local live inference, grabs frames from a webcam (or a video file as a fallback
#demo source), runs the trained yolov8 model on each frame, draws boxes live, and
#reports fps
#usage: python3 scripts/live_infer.py                     #webcam (device 0)
#       python3 scripts/live_infer.py --source path.mp4   #looped video file
#       python3 scripts/live_infer.py --source 1          #webcam device 1
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["smoke", "fire"]
COLORS = {0: (0, 255, 255), 1: (0, 0, 255)}  # BGR: smoke=yellow, fire=red


#figures out whether source is a webcam index or a video file path, and opens it
#input: (source - a string, either a webcam index like "0" or a path to a video file)
#returns: (cap, source, is_file), the opened VideoCapture, the source value (int if
#it was a webcam index), and whether it's a file
def open_source(source):
    #tries to interpret as a webcam index, otherwise treats it as a file path
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    is_file = isinstance(source, str)
    return cap, source, is_file


#draws boxes and labels for every detection above the confidence threshold, in place
#input: (frame - the bgr frame to draw on) (result - a single ultralytics predict()
#result) (conf_thres - minimum confidence to draw a box for)
#returns: the same frame, with boxes drawn on it
def draw_detections(frame, result, conf_thres):
    for box in result.boxes:
        conf = float(box.conf.item())
        if conf < conf_thres:
            continue
        cls = int(box.cls.item())
        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        #filled label background behind the text so it's readable on any frame
        cv2.rectangle(frame, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
        cv2.putText(frame, label, (x0 + 2, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


#entry point, loads the model, opens the source, and loops reading/predicting/
#drawing/displaying frames until the source ends or 'q' is pressed
#input: none, reads args from the command line
#returns: nothing, opens a live opencv window as a side effect
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "model/best.pt"))
    parser.add_argument("--source", default="0", help="webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--loop", action="store_true", default=True, help="loop video file sources")
    args = parser.parse_args()

    print(f"Loading model from {args.weights} ...")
    model = YOLO(args.weights)

    cap, source, is_file = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    print(f"Source: {'file ' + args.source if is_file else 'webcam device ' + args.source}")
    print("Press 'q' to quit.")

    fps_smoothed = 0.0
    alpha = 0.1  #ema smoothing factor for the fps readout, so it doesn't jump around every frame

    while True:
        ok, frame = cap.read()
        if not ok:
            #video files hit end of stream, loop back to the start instead of quitting
            if is_file and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("End of stream / cannot read frame.")
            break

        t0 = time.perf_counter()
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        t1 = time.perf_counter()

        frame = draw_detections(frame, result, args.conf)

        inst_fps = 1.0 / max(t1 - t0, 1e-6)
        fps_smoothed = inst_fps if fps_smoothed == 0 else (alpha * inst_fps + (1 - alpha) * fps_smoothed)
        cv2.putText(
            frame, f"FPS: {fps_smoothed:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        cv2.imshow("Wildfire Smoke/Fire Detection - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
