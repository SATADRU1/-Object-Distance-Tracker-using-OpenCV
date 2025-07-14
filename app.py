import cv2
import numpy as np
import math
import time

class ReferenceObjectTracker:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.pixels_per_cm = None
        self.calibrated = False

        self.reference_object = None
        self.tracked_objects = []

        self.color_ranges = {
            'red': {
                'lower1': np.array([0, 50, 50]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 50, 50]),
                'upper2': np.array([180, 255, 255])
            },
            'blue': {'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])},
            'green': {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
            'yellow': {'lower': np.array([20, 50, 50]), 'upper': np.array([30, 255, 255])},
            'orange': {'lower': np.array([5, 50, 50]), 'upper': np.array([15, 255, 255])},
            'purple': {'lower': np.array([130, 50, 50]), 'upper': np.array([160, 255, 255])}
        }

        self.min_contour_area = 800
        self.reference_width_cm = 21.0

    def detect_objects_by_color(self, frame, color_name):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color_name not in self.color_ranges:
            return []

        color_range = self.color_ranges[color_name]
        if 'lower1' in color_range:
            mask1 = cv2.inRange(hsv, color_range['lower1'], color_range['upper1'])
            mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        'centroid': (cx, cy),
                        'area': area,
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'color': color_name
                    })
        return objects

    def detect_all_objects(self, frame):
        all_objects = []
        for color_name in self.color_ranges:
            objects = self.detect_objects_by_color(frame, color_name)
            all_objects.extend(objects)
        return all_objects

    def calculate_distance_pixels(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def pixels_to_cm(self, pixels):
        return pixels / self.pixels_per_cm if self.calibrated else pixels

    def is_same_object(self, obj1, obj2, max_distance=50):
        return self.calculate_distance_pixels(obj1['centroid'], obj2['centroid']) < max_distance

    def update_tracking(self, detected_objects):
        if self.reference_object is None and detected_objects:
            self.reference_object = detected_objects[0].copy()
            self.reference_object['id'] = 'REF'
            print(f"Reference object set: {self.reference_object['color']}")
            return

        self.tracked_objects = []
        for obj in detected_objects:
            if self.is_same_object(obj, self.reference_object):
                continue

            distance_px = self.calculate_distance_pixels(
                self.reference_object['centroid'], obj['centroid']
            )
            distance = self.pixels_to_cm(distance_px)

            new_obj = obj.copy()
            new_obj['distance_from_ref'] = distance
            self.tracked_objects.append(new_obj)

    def calibrate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    self.pixels_per_cm = w / self.reference_width_cm
                    self.calibrated = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    return True
        return False

    def draw(self, frame):
        color_map = {
            'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
            'yellow': (0, 255, 255), 'orange': (0, 165, 255), 'purple': (128, 0, 128)
        }

        if self.reference_object:
            ref = self.reference_object
            x, y, w, h = ref['bbox']
            cx, cy = ref['centroid']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 215, 255), 3)
            cv2.circle(frame, (cx, cy), 8, (0, 215, 255), -1)
            cv2.putText(frame, f"REF: {ref['color']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)

        for obj in self.tracked_objects:
            x, y, w, h = obj['bbox']
            cx, cy = obj['centroid']
            color = color_map.get(obj['color'], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"{obj['color']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            ref_cx, ref_cy = self.reference_object['centroid']
            cv2.line(frame, (ref_cx, ref_cy), (cx, cy), (255, 255, 255), 2)
            mid_x, mid_y = (ref_cx + cx)//2, (ref_cy + cy)//2
            dist_txt = f"{obj['distance_from_ref']:.1f} cm"
            text_size = cv2.getTextSize(dist_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (mid_x - text_size[0]//2 - 5, mid_y - text_size[1] - 5), 
                          (mid_x + text_size[0]//2 + 5, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, dist_txt, (mid_x - text_size[0]//2, mid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def reset(self):
        self.reference_object = None
        self.tracked_objects = []
        print("Tracking reset.")

    def run(self):
        print("Press 'c' to calibrate | 'r' to reset | 'q' to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detected_objects = self.detect_all_objects(frame)
            self.update_tracking(detected_objects)
            self.draw(frame)

            cv2.imshow("Object Distance Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('c'):
                if self.calibrate(frame):
                    print(f"Calibration successful: {self.pixels_per_cm:.2f} px/cm")
                else:
                    print("Calibration failed. Ensure A4 paper is visible.")

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    tracker = ReferenceObjectTracker()
    tracker.run()

if __name__ == '__main__':
    main()
