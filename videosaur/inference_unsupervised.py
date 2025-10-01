import argparse
import torch
from omegaconf import OmegaConf
from videosaur import configuration, models
from videosaur.data.transforms import CropResize, Normalize, Resize, build_inference_transform
import os
import numpy as np
import imageio
from torchvision import transforms as tvt
from videosaur.visualizations import mix_inputs_with_masks, draw_segmentation_masks_on_image, color_map, mix_videos_with_masks
import cv2
import tensorflow_datasets as tfds
import yaml
import time 
import sys

def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config

def prepare_video_from_tfds(video_np: np.ndarray, transfom_config=None):
    # Load video
    video = torch.from_numpy(video_np).float() / 255.0
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize((transfom_config.input_size, transfom_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        video = video.permute(3, 0, 1, 2)
        video = tfs(video).permute(1, 0, 2, 3)
     # Add batch dimension
    inputs = {"video": video.unsqueeze(0), 
              "video_visualization": video_vis.unsqueeze(0)}
    return inputs

def filter_dets(dets_array: np.ndarray, height: int, width: int, area_threshold: float = 0.005, inclusion_threshold: float = 0.1):
    """
    Filters detections based on two criteria:
    1) Removes detections whose normalized area is less than area_threshold.
    2) Removes detections if truncated (less than 20% of their width (in pixel coordinates) 
       lies within the allowed x-range [320, 1600]).
        
    Returns:
        np.ndarray: Filtered detections that meet both criteria.
    """
    filtered_dets = []

    # Remove zero padded rows
    dets_array = dets_array[~np.all(dets_array == 0, axis=1)]

    # Define allowed x-range in pixel coordinates.
    allowed_xmin, allowed_xmax = 320, 1600

    for det in dets_array:
        # Unpack the detection.
        y_min, y_max, x_min, x_max, cls = det

        # Compute the normalized area of the bounding box.
        norm_area = (y_max - y_min) * (x_max - x_min)
        if norm_area < area_threshold:
            continue  # Skip small detections.

        # Convert normalized x coordinates to pixel coordinates.
        x_min_pix = x_min * width
        x_max_pix = x_max * width
        box_width = x_max_pix - x_min_pix
        
        # Avoid division by zero or degenerate boxes.
        if box_width <= 0:
            continue
        
        # Calculate the overlap between the bounding box and the allowed x-range.
        overlap = max(0, min(x_max_pix, allowed_xmax) - max(x_min_pix, allowed_xmin))
        allowed_fraction = overlap / box_width
        
        # If less than x% of the width is within the allowed range, skip this detection.
        if allowed_fraction < inclusion_threshold:
            continue
        
        # Otherwise, keep the detection.
        filtered_dets.append(det)
    return np.array(filtered_dets)
  
def main(config):
    # Load the model from checkpoint
    model, _ = load_model_from_checkpoint(config.checkpoint, config.model_config)

    # Get path to TFDS and load dataset for inference
    data_dir = '/scratch/eng/t212-amlab/waymo/waymo-video-tfds-40' #TODO: Change
    original_data_path = '/scratch/eng/t212-amlab/waymo/waymo_ds_v_1_4_1/' #TODO: Change

    vn, vn_info = tfds.load(
    'waymo_video/video40',
    data_dir=data_dir,
    try_gcs=False,
    download_and_prepare_kwargs={
        'download_config': tfds.core.download.DownloadConfig(manual_dir=original_data_path)
    },
    shuffle_files=False,
    with_info=True,
    download=False
    )
    
    execution_split = 'validation' #TODO: change split if needed
    vn_split = vn[execution_split] 

    for idx, ex in enumerate(vn_split):
        if idx!=-1:
            # Get the n consecutive RGB frames
            cam_vid_orig = ex['camera_FRONT']['image'].numpy()  # shape: (n, 1280, 1920, 3)
            
            # Crop to aspect ratio
            video = np.stack([frame[:, 320:1600] for frame in cam_vid_orig]) # shape: (n, 1280, 1280, 3)
            
            # Get number of bounding boxes in first frame and set slots based on that (it has to be between config.min and 10)
            det_0 = (ex['camera_FRONT']['detections']).numpy()[0] # shape: (150, 5)
            
            # Remove detections out of boundaries and remove small ones
            det_0_filtered = filter_dets(det_0, cam_vid_orig.shape[1], cam_vid_orig.shape[2])

            model.initializer.n_slots = min(max(len(det_0_filtered) + 1, config.n_slots_min), 10)

            # Update save path
            video_name = f"{execution_split}_sample_{idx}"
            output_save_path = os.path.join(config.output.save_folder, f"{video_name}.mp4")
            
            # Prepare the input from the tensorflow dataset
            inputs = prepare_video_from_tfds(video, config.input.transforms)

            # Perform inference
            with torch.no_grad():
                start_time = time.time()
                outputs = model(inputs)
                elapsed_time = time.time() - start_time
                print(f"Model execution time: {elapsed_time}")
            if config.input.type=="video" and output_save_path:
                # Save the results
                save_dir = os.path.dirname(output_save_path)
                os.makedirs(save_dir, exist_ok=True)
                masked_video_frames = mix_inputs_with_masks(inputs, outputs)
                masked_video_frames = np.array(masked_video_frames) # (t, 1038, 3118, 3)
                imageio.mimwrite(output_save_path, masked_video_frames, fps=5, codec="libx264") # 5fps

            print(f"Inference completed for: {video_name}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a single MP4 video.")
    parser.add_argument("--config", default="configs/inference/waymo_tfds.yml", help="Configuration to run")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)