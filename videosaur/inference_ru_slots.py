from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics import settings
from shapely.geometry import box
from shapely.ops import unary_union
import imageio
import sys
import glob
import os
import math
import re

# Update multiple settings
settings.update({"runs_dir": "./yolo_runs", "weights_dir": "./yolo_weights"}) #TODO: Edit paths (empty folders) as needed

def video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    # Convert list of frames to a NumPy array
    video_array = np.array(frames)
    return video_array

def numeric_key(path):
    m = re.search(r'validation_sample_(\d+)\.mp4$', path) #TODO: Change for split (train|validation)
    return int(m.group(1)) if m else -1

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Video directory
video_dir = "./waymo/unsupervised_inference_results" # TODO:Change path as needed
video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mp4')), key=numeric_key)

video_names = []

active_slots_file = "./waymo/road_user_slots_waymo.txt"# TODO:Change path as needed
with open(active_slots_file, 'w') as f:
    pass  # just clear the file

for v_i, video_path in enumerate(video_paths):
    # Pick video and loop its frames
    video_name = os.path.basename(video_path)
    if video_name not in video_names:
        video_np = video_to_numpy(video_path)

        # Grid parameters
        sub_h, sub_w = 518, 518   # sub-image height and width
        pad = 2                   # padding between sub-images
        nrows, ncols = 2, 6       # grid size

        # Predetermined thresholds (adjust these as needed)
        threshold_area_ratio = 0.05    # e.g. cumulative detection area threshold
        threshold_confidence = 0.50    # e.g. highest confidence

        # Allowed road users (coco categories)
        road_user_ids = {0, 1, 2, 3, 5, 6, 7}

        conf_max = 0.85 # full alpha

        for frame_id, frame in enumerate(video_np):
            frame = frame[:1038, :3118, :]
            # Active slots
            active_slots = []
            no_of_total_slots = -1
            print(frame.shape)
            assert frame.shape[0]==1038
            assert frame.shape[1]==3118

            first_black_index = None

            # Loop over each sub-image (row-major order) to find 1st black sub image
            for i in range(nrows * ncols):
                r = i // ncols   # row index (0 or 1)
                c = i % ncols    # column index (0 to 5)
                # Calculate the coordinates of the sub-image:
                top = r * sub_h + r * pad
                left = c * sub_w + c * pad
                bottom = top + sub_h
                right = left + sub_w
                sub_img = frame[top:bottom, left:right, :]

                # Check if the sub-image is all black
                if np.all(sub_img == 0):
                    first_black_index = i
                    break
                else:
                    no_of_total_slots = no_of_total_slots + 1
            if first_black_index is not None:
                # Black out all images from the identified index
                # but if that is negative (i.e. if the first sub-image is black), use index 0.
                start_index = max(first_black_index, 0)
                # Set all sub-images from start_index to the end to black
                for i in range(start_index, nrows * ncols):
                    r = i // ncols
                    c = i % ncols
                    top = r * sub_h + r * pad
                    left = c * sub_w + c * pad
                    bottom = top + sub_h
                    right = left + sub_w
                    frame[top:bottom, left:right, :] = 0
            
            # Loop over each sub-image (row-major order) to filter
            for i in range(nrows * ncols):
                r = i // ncols  # row index (0 or 1)
                c = i % ncols   # column index (0 to 5)
                if (r==0 and c==0):
                    continue
                # Calculate the sub-image coordinates in the frame:
                top = r * sub_h + r * pad
                left = c * sub_w + c * pad
                bottom = top + sub_h
                right = left + sub_w
                sub_img = frame[top:bottom, left:right, :]

                # Skip if sub-image is already black (all pixels are zero)
                if np.all(sub_img == 0):
                    continue
                
                # Run YOLO detection on the sub-image.
                results = model(sub_img)

                # We assume a single result for a single image input:
                result = results[0]
                
                # Retrieve detection info: bounding boxes, classes, and confidence scores.
                boxes = result.boxes.xyxy.cpu().numpy()  # shape (N,4): [x1, y1, x2, y2]
                classes = result.boxes.cls.int().cpu().numpy()  # class indices for each detection
                confs = result.boxes.conf.cpu().numpy()  # confidence scores for each detection

                # Filter detections: consider only road users.
                road_indices = [j for j, cls in enumerate(classes) if cls in road_user_ids]

                if (len(road_indices) == 0):
                    # No road user detections: black out this sub-image.
                    frame[top:bottom, left:right, :] = 0
                    continue

                # Calculate cumulative detection area (in pixels) and track the highest confidence.
                max_conf = 0

                # Create Shapely box objects for each road user detection
                detection_boxes = []
                boxes_to_overlay = []
                confs_filtered = []
                for j in road_indices:
                    x1, y1, x2, y2 = boxes[j] # in pixels
                    # Create a box geometry: Shapely expects (minx, miny, maxx, maxy)
                    detection_boxes.append(box(x1, y1, x2, y2))
                    if confs[j] > max_conf:
                        max_conf = confs[j]

                    # Boxes to overlay
                    if (confs[j] > threshold_confidence):
                        boxes_to_overlay.append([x1, y1, x2, y2])
                        confs_filtered.append(confs[j])

                # Compute the union of all boxes
                if detection_boxes:
                    union = unary_union(detection_boxes)
                    total_detection_area = union.area
                else:
                    total_detection_area = 0

                # Count the total number of non-zero pixels in the sub-image.
                # Here, a pixel is considered non-zero if at least one of its three channels is non-zero.
                non_zero_pixels = np.sum(np.any(sub_img != 0, axis=2))
                
                # If for some reason the sub-image has no non-zero pixels, black it out.
                if non_zero_pixels == 0:
                    frame[top:bottom, left:right, :] = 0
                    continue
                
                # Compute the ratio of cumulative detection area to the number of non-zero pixels.
                area_ratio = total_detection_area / non_zero_pixels
                # Check the two conditions: area ratio and highest confidence.boxes
                if area_ratio < threshold_area_ratio or max_conf < threshold_confidence:
                    # If either condition is not met, set the entire sub-image to black.
                    frame[top:bottom, left:right, :] = 0
                    continue
                
                # Append index to active slots
                active_slots.append(i)

            # Loop over each remaining sub-image (row-major order) and label
            for i in range(nrows * ncols):
                if i in active_slots:
                    # Write to file
                    array_to_text = [
                        v_i,                    # Video id (index)
                        frame_id,               # Frame id
                        i                       # Slot id
                    ]
                    with open(active_slots_file, 'a') as f:
                        line = " ".join(str(item) for item in array_to_text)
                        f.write(line + "\n")
            print(f"Eval complete for video {v_i}, frame {frame_id}")