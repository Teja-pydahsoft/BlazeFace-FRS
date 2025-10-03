"""
Simple Centroid-Based Face Tracker
Keeps track of faces across frames to maintain identity
"""

from collections import OrderedDict
import numpy as np

class FaceTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize the face tracker
        
        Args:
            max_disappeared: Maximum number of consecutive frames a face can be marked as disappeared
            max_distance: Maximum distance between centroids to be considered the same face
        """
        self.next_face_id = 0
        self.faces = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Register a new face"""
        self.faces[self.next_face_id] = centroid
        self.disappeared[self.next_face_id] = 0
        self.next_face_id += 1

    def deregister(self, face_id):
        """Deregister a face"""
        del self.faces[face_id]
        del self.disappeared[face_id]

    def update(self, rects):
        """
        Update the tracker with new face bounding boxes
        
        Args:
            rects: List of (x, y, w, h) bounding boxes
            
        Returns:
            A dictionary of tracked faces
        """
        if len(rects) == 0:
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            return self.faces

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = x + w // 2
            cy = y + h // 2
            input_centroids[i] = (cx, cy)

        if len(self.faces) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            face_ids = list(self.faces.keys())
            face_centroids = np.array(list(self.faces.values()))

            D = np.sqrt(((face_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :])**2).sum(axis=2))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                face_id = face_ids[row]
                self.faces[face_id] = input_centroids[col]
                self.disappeared[face_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    face_id = face_ids[row]
                    self.disappeared[face_id] += 1
                    if self.disappeared[face_id] > self.max_disappeared:
                        self.deregister(face_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.faces
