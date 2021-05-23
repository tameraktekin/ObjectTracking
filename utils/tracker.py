from collections import OrderedDict
from .functions import *
from scipy.spatial import distance


class Tracker:
    def __init__(self, max_frames_to_disappear=30):
        self.next_objectID = 0
        self.objects = OrderedDict()
        self.disappeared_objects_count = OrderedDict()

        self.max_frames_to_disappear = max_frames_to_disappear

    def update(self, rects):
        if len(rects) == 0:
            self.update_disappeared_objects(self.disappeared_objects_count.keys())
        else:
            all_centroids = np.zeros((len(rects), 2), dtype=int)
            for (i, rect) in enumerate(rects):
                all_centroids[i] = calculate_centroid(rect)

            if self.check_if_no_objects():
                self.add_objects(all_centroids)
            else:
                self.update_tracked_objects(all_centroids)

        return self.objects

    def update_disappeared_objects(self, objectIDs):
        for objectID in objectIDs:
            self.update_disappeared_count(objectID)

            if self.check_if_removed(objectID):
                self.remove_object(objectID)

    def update_disappeared_count(self, objectID):
        self.disappeared_objects_count[objectID] += 1

    def check_if_removed(self, objectID):
        return self.disappeared_objects_count[objectID] > self.max_frames_to_disappear

    def remove_object(self, objectID):
        del self.objects[objectID]
        del self.disappeared_objects_count[objectID]
        self.next_objectID -= 1

    def check_if_no_objects(self):
        return len(self.objects) == 0

    def add_objects(self, all_centroids):
        for centroid in all_centroids:
            self.objects[self.next_objectID] = centroid
            self.disappeared_objects_count[self.next_objectID] = 0
            self.next_objectID += 1

    def update_tracked_objects(self, new_centroids):
        objectIDs = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        closest_pairs = self.get_closest_pairs(object_centroids, new_centroids)

        used_rows = set()
        used_cols = set()
        for (row, col) in closest_pairs:
            if row in used_rows or col in used_cols:
                continue

            self.update_object(objectIDs[row], new_centroids[col])
            used_rows.add(row)
            used_cols.add(col)

        if len(object_centroids) > len(new_centroids):
            disappeared_objects = set(range(0, len(object_centroids))).difference(used_rows)
            for idx in disappeared_objects:
                self.update_disappeared_objects([objectIDs[idx]])
        else:
            new_objects = set(range(0, len(new_centroids))).difference(used_cols)
            for idx in new_objects:
                self.add_objects([new_centroids[idx]])

    def get_closest_pairs(self, old_centroids, new_centroids):
        distances = distance.cdist(old_centroids, new_centroids)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        return zip(rows, cols)

    def update_object(self, objectID, centroid):
        self.objects[objectID] = centroid
        self.disappeared_objects_count[objectID] = 0
