#simple script client for testing the /ws/detect websocket endpoint locally, without
#needing a frontend. streams frames from a video file or webcam and prints the
#detections returned for each
#usage: python3 backend/test_client.py --source path/to/video.mp4
#       python3 backend/test_client.py --source 0   #webcam
import argparse
import asyncio
import time

import cv2
import websockets


#opens the source, connects to the websocket, and sends frames one at a time,
#waiting for a response before sending the next (mirrors how the real frontend paces
#itself, so this is a decent stand in for round trip timing)
#input: (source - webcam index or video file path) (url - the ws:// endpoint to connect
#to) (max_frames - stop after sending this many) (jpeg_quality - jpeg encode quality,
#0-100)
#returns: nothing, prints each response and a final frames/s summary
async def stream(source, url, max_frames, jpeg_quality):
    #source might be a webcam index passed as a string, try to convert it
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    async with websockets.connect(url) as ws:
        sent = 0
        t_start = time.perf_counter()
        while sent < max_frames:
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue

            await ws.send(buf.tobytes())
            response = await ws.recv()
            sent += 1
            print(f"[{sent}] {response}")

        elapsed = time.perf_counter() - t_start
        print(f"\nSent {sent} frames in {elapsed:.2f}s ({sent / elapsed:.1f} frames/s round-trip)")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="webcam index or path to a video file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/detect")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    asyncio.run(stream(args.source, args.url, args.max_frames, args.jpeg_quality))
