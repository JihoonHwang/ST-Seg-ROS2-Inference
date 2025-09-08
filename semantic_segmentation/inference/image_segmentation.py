#!/usr/bin/env python3

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm

# Updated imports for MMSegmentation 1.2.2
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.apis import init_model, inference_model


class Segmentation:
    def __init__(self, config_path, weight_path, device='cuda:0'):
        """
        Initialize the segmentation model.
        
        Args:
            config_path: Path to the model configuration file
            weight_path: Path to the model weights file
            device: Device to run the model on (cuda:0, cpu, etc.)
        """
        self.device = device
        self.config_path = config_path
        self.weight_path = weight_path
        
        # Define palette
        self.palette = np.array([
            0,     0,   0,  # background /black
            255, 128,   0,  # smooth ground / orange    
            255, 128,   0,  # rough ground / orange
            255, 128,   0,  # bumpy ground / orange
            0,   255,   0,  # soft veg / green
            0,    80,   0,  # hard veg / dark green
            0,     0, 255,  # puddle /  blue 
            255,   0, 127   # obstacle / pink
        ]).reshape(-1, 3)
        
        print(f"Loading model from {self.config_path}")
        print(f"Loading weights from {self.weight_path}")
        self.model = self.load_model()
        print("Model loaded successfully!")
    
    def load_model(self):
        """Load the segmentation model."""
        # Use init_model API for MMSegmentation 1.2.2
        model = init_model(self.config_path, self.weight_path, device=self.device)
        
        # Load checkpoint to get metadata
        checkpoint = load_checkpoint(
            model,
            self.weight_path,
            map_location=self.device,
            revise_keys=[(r'^module\.', ''), ('model.', '')]
        )
        
        # Set CLASSES and PALETTE from checkpoint metadata
        if 'meta' in checkpoint:
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint['meta']:
                model.PALETTE = checkpoint['meta']['PALETTE']
        
        # For compatibility, store the config
        model.cfg = Config.fromfile(self.config_path)
        
        model.eval()
        return model
    
    def segment_image(self, image_path):
        """
        Perform segmentation on a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            segmented_image: Color-coded segmentation result
            inference_time: Time taken for inference in seconds
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None, 0
        
        # BGR palette for OpenCV
        palette_bgr = self.palette[:, ::-1]
        
        # Perform inference
        start_time = time.time()
        result = inference_model(self.model, img)
        inference_time = time.time() - start_time
        
        # Extract semantic segmentation result
        if hasattr(result, 'pred_sem_seg'):
            sem_seg = result.pred_sem_seg.data.cpu().numpy()[0]
        else:
            # Fallback for compatibility
            sem_seg = result[0] if isinstance(result, (list, tuple)) else result
            if isinstance(sem_seg, torch.Tensor):
                sem_seg = sem_seg.cpu().numpy()
            if len(sem_seg.shape) == 3 and sem_seg.shape[0] == 1:
                sem_seg = sem_seg[0]
        
        # Create color segmentation image
        color_seg = palette_bgr[sem_seg]
        
        return color_seg, inference_time
    
    def process_folder(self, input_folder, output_folder, image_extensions=None):
        """
        Process all images in a folder.
        
        Args:
            input_folder: Path to the folder containing input images
            output_folder: Path to the folder where segmented images will be saved
            image_extensions: List of image file extensions to process
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create separate output folders
        segmented_path = output_path / 'segmented'
        comparison_path = output_path / 'comparison'
        
        segmented_path.mkdir(parents=True, exist_ok=True)
        comparison_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Segmented images will be saved to: {segmented_path}")
        print(f"Comparison images will be saved to: {comparison_path}")
        
        total_time = 0
        successful = 0
        
        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                # Segment the image
                seg_result, inference_time = self.segment_image(image_file)
                
                if seg_result is None:
                    continue
                
                # Save the segmented image in segmented folder
                output_file = segmented_path / f"{image_file.stem}_segmented.png"
                cv2.imwrite(str(output_file), seg_result)
                
                # Save side-by-side comparison in comparison folder
                original = cv2.imread(str(image_file))
                if original is not None:
                    # Resize segmentation to match original if needed
                    if seg_result.shape[:2] != original.shape[:2]:
                        seg_result_resized = cv2.resize(seg_result, (original.shape[1], original.shape[0]))
                    else:
                        seg_result_resized = seg_result
                    
                    comparison = np.hstack([original, seg_result_resized])
                    comparison_file = comparison_path / f"{image_file.stem}_comparison.png"
                    cv2.imwrite(str(comparison_file), comparison)
                
                total_time += inference_time
                successful += 1
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        if successful > 0:
            avg_time = total_time / successful
            print(f"\nProcessing complete!")
            print(f"Successfully processed: {successful}/{len(image_files)} images")
            print(f"Average inference time: {avg_time:.3f} seconds")
            print(f"Average FPS: {1/avg_time:.2f}")
            print(f"Output saved to:")
            print(f"  - Segmented: {segmented_path}")
            print(f"  - Comparison: {comparison_path}")
        else:
            print("No images were successfully processed")


def main():
    parser = argparse.ArgumentParser(description='Batch image segmentation using ST-Seg model')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing input images')
    parser.add_argument('output_folder', type=str, help='Path to the folder where segmented images will be saved')
    parser.add_argument('--config', type=str, 
                        default='/root/docker-workspace/st-seg/ST-Seg-ROS2-Inference/semantic_segmentation/library/model/segformer.py',
                        help='Path to the model configuration file')
    parser.add_argument('--weight', type=str,
                        default='/root/docker-workspace/st-seg/ST-Seg-ROS2-Inference/semantic_segmentation/library/model/ST_seg.pth',
                        help='Path to the model weights file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--extensions', nargs='+', default=None,
                        help='Image file extensions to process (e.g., .jpg .png)')
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist")
        sys.exit(1)
    
    # Check if model files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.weight):
        print(f"Error: Weight file {args.weight} does not exist")
        sys.exit(1)
    
    # Initialize the segmentation model
    segmenter = Segmentation(
        config_path=args.config,
        weight_path=args.weight,
        device=args.device
    )
    
    # Process the folder
    segmenter.process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        image_extensions=args.extensions
    )


if __name__ == '__main__':
    main()