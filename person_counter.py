#!/usr/bin/env python3
"""
Enhanced Person Counter with Video Recording + Flask API

Usage:
  python person_counter.py --source 0             # webcam
  python person_counter.py --source video.mp4     # video file
Options:
  --line 0.5      # fraction height for counting line (0.0 top, 1.0 bottom)
  --display 1     # show GUI window (0 to run headless)
  --no-server     # don't start Flask server
  --port 5000     # Flask server port
"""

import cv2, time, argparse, threading
import numpy as np
from imutils.object_detection import non_max_suppression
from flask import Flask, jsonify

# ----------------------
# Centroid Tracker
# ----------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=50):
        self.nextObjectID = 0
        self.objects = dict()      # objectID -> (centroid, bbox)
        self.disappeared = dict()  # objectID -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = (centroid, bbox)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = np.array([self.objects[oid][0] for oid in objectIDs])

        D = np.linalg.norm(objectCentroids[:, np.newaxis] - inputCentroids[np.newaxis, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = (inputCentroids[col], rects[col])
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # mark unmatched existing objects disappeared
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        # register unmatched new centroids
        unusedCols = set(range(D.shape[1])).difference(usedCols)
        for col in unusedCols:
            self.register(inputCentroids[col], rects[col])

        return self.objects

# ----------------------
# Trackable Object
# ----------------------
class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False

# ----------------------
# Global counts + lock
# ----------------------
COUNTS = {'in': 0, 'out': 0}
LOCK = threading.Lock()

# ----------------------
# Detector + counting loop
# ----------------------
def detect_and_count(source=0, line_position=0.5, display=True):
    global COUNTS, LOCK
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    vs = cv2.VideoCapture(source)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    ct = CentroidTracker(maxDisappeared=40, maxDistance=80)
    trackableObjects = dict()

    W = H = None
    out = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    time.sleep(1.0)

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = cv2.resize(frame, (900, int(frame.shape[0] * 900 / frame.shape[1])))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            line_y = int(H * line_position)
            out = cv2.VideoWriter("output_counter.mp4", fourcc, 20.0, (W, H))

        rects, weights = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
        boxes = [(x, y, x+w, y+h) for (x, y, w, h) in rects]

        if len(boxes) > 0:
            boxes_np = np.array(boxes)
            pick = non_max_suppression(boxes_np, probs=None, overlapThresh=0.65)
            rects_nms = [(int(xA), int(yA), int(xB), int(yB)) for (xA, yA, xB, yB) in pick]
        else:
            rects_nms = []

        objects = ct.update(rects_nms)

        for (objectID, (centroid, bbox)) in list(objects.items()):
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                prev_y = np.mean([c[1] for c in to.centroids]) if len(to.centroids) > 0 else None
                to.centroids.append(centroid)
                if (not to.counted) and (prev_y is not None):
                    direction = centroid[1] - prev_y
                    if direction < 0 and centroid[1] < line_y:
                        with LOCK:
                            COUNTS['in'] += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > line_y:
                        with LOCK:
                            COUNTS['out'] += 1
                        to.counted = True

            trackableObjects[objectID] = to

            (startX, startY, endX, endY) = bbox
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            cv2.putText(frame, f"ID {objectID}", (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

        cv2.line(frame, (0, line_y), (W, line_y), (0,0,255), 2)
        with LOCK:
            info = [("In", COUNTS['in']), ("Out", COUNTS['out']), ("Current", len(objects))]
        for (i, (k,v)) in enumerate(info):
            cv2.putText(frame, f"{k}: {v}", (10, H - ((i*20)+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        out.write(frame)

        if display:
            cv2.imshow("People Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    vs.release()
    out.release()
    cv2.destroyAllWindows()

# ----------------------
# Flask API
# ----------------------
app = Flask(__name__)

@app.route("/counts")
def counts_api():
    with LOCK:
        return jsonify(COUNTS)

# ----------------------
# CLI + run
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="video source (0 for webcam or path)")
    parser.add_argument("--line", type=float, default=0.5, help="line position fraction")
    parser.add_argument("--display", type=int, default=1, help="show window (1/0)")
    parser.add_argument("--no-server", action="store_true", help="disable Flask server")
    parser.add_argument("--port", type=int, default=5000, help="Flask server port")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    t = threading.Thread(target=detect_and_count, args=(source, args.line, bool(args.display)))
    t.daemon = True
    t.start()

    if args.no_server:
        print("[INFO] Running detector only. Press Ctrl+C to exit.")
        t.join()
    else:
        print(f"[INFO] Flask server on port {args.port}. Open http://localhost:{args.port}/counts")
        app.run(host="0.0.0.0", port=args.port, threaded=True)
