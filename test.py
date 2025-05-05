# Enhanced Anomaly Detection System with GUI for image, video, and real-time camera detection
import cv2
import torch
import numpy as np
import time
import os
import glob
import argparse
from anomalib.deploy import TorchInferencer
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from datetime import datetime

class AnomalyDetector:
    def __init__(self, model_path=None, threshold=0.7, device='cuda'):
        """
        Initialize the anomaly detector with a model
        
        Args:
            model_path: Path to the exported model (.pt file)
            threshold: Threshold for considering an anomaly (0.0 - 1.0)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.threshold = threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize variables
        self.bg_subtractor = None
        self.inferencer = None
        
        # Load model if path is provided
        if model_path:
            self.load_model(model_path)
        
        # Reset background model
        self.reset_background_model()
        
        # Metrics tracking
        self.fps_history = []
        self.score_history = []
        
        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Stabilization variables
        self.anomaly_confidence = 0
        self.anomaly_cooldown = 0
        self.anomaly_detected = False
        
        # Create output directory
        os.makedirs("./outputs", exist_ok=True)
    
    def load_model(self, model_path):
        """Load a PyTorch model for anomaly detection"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Loading model from {model_path}...")
        try:
            self.inferencer = TorchInferencer(path=model_path, device=self.device)
            self.model_path = model_path
            print(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def reset_background_model(self):
        """Reset the background subtractor model"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False)
        self.anomaly_confidence = 0
        self.anomaly_cooldown = 0
        self.anomaly_detected = False
    
    def detect_anomalies_in_image(self, image_path, output_dir="./outputs"):
        """
        Process a single image file and save the result
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save the output image
        """
        if not self.inferencer:
            print("No model loaded. Please load a model first.")
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing image: {image_path}")
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image {image_path}")
                return None
            
            # Create a copy for drawing
            display_img = img.copy()
            
            # Start inference timer
            inference_start = time.time()
            
            # Run inference
            result = self.inferencer.predict(image=image_path)
            
            # Process result
            if isinstance(result.pred_score, torch.Tensor):
                anomaly_score = float(result.pred_score.cpu().item())
            else:
                anomaly_score = float(result.pred_score)
                
            if isinstance(result.pred_label, torch.Tensor):
                is_anomalous = bool(result.pred_label.cpu().item())
            else:
                is_anomalous = bool(result.pred_label)
            
            # End inference timer
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Process anomalies if detected
            if is_anomalous and anomaly_score > self.threshold:
                # Add red border
                border_size = 10
                display_img = cv2.copyMakeBorder(
                    display_img,
                    border_size, border_size, border_size, border_size,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 255)  # Red border
                )
                
                # Add anomaly text
                cv2.putText(
                    display_img, 
                    "ANOMALY DETECTED",
                    (int(display_img.shape[1]/2) - 150, 40),
                    self.font, 1.2, (0, 0, 255), 3
                )
                
                # Apply anomaly mask if available
                if hasattr(result, 'pred_mask') and result.pred_mask is not None:
                    try:
                        # Convert mask to numpy if it's a tensor
                        if isinstance(result.pred_mask, torch.Tensor):
                            mask = result.pred_mask.cpu().numpy()
                        else:
                            mask = result.pred_mask
                            
                        if len(mask.shape) > 2:
                            mask = mask[0]  # Take first item if batch dimension exists
                        
                        # Resize mask to match image size
                        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                        
                        # Create heatmap
                        heatmap = cv2.applyColorMap(
                            (mask_resized * 255).astype(np.uint8),
                            cv2.COLORMAP_JET
                        )
                        
                        # Create overlay
                        overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
                        
                        # Add text to overlay
                        cv2.putText(
                            overlay,
                            f"Anomaly Score: {anomaly_score:.4f}",
                            (20, 30),
                            self.font, 0.7, (255, 255, 255), 2
                        )
                        
                        # Add frame around regions with high anomaly scores
                        _, thresh = cv2.threshold(
                            (mask_resized * 255).astype(np.uint8),
                            int(self.threshold * 255),
                            255,
                            cv2.THRESH_BINARY
                        )
                        
                        # Find contours in thresholded mask
                        contours, _ = cv2.findContours(
                            thresh,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        # Draw bounding boxes around anomalous regions
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 100:  # Filter out tiny regions
                                x, y, w, h = cv2.boundingRect(contour)
                                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                
                                # Only add "ANOMALY" label for significant regions
                                if w > 50 and h > 50:
                                    cv2.putText(
                                        overlay,
                                        "ANOMALY",
                                        (x, max(y-10, 15)),
                                        self.font, 0.5, (0, 0, 255), 2
                                    )
                        
                        # Stack images side by side for comparison
                        display_img = np.hstack((display_img, overlay))
                    except Exception as e:
                        print(f"Error creating heatmap: {e}")
            else:
                # Normal case - green border
                border_size = 5
                display_img = cv2.copyMakeBorder(
                    display_img,
                    border_size, border_size, border_size, border_size,
                    cv2.BORDER_CONSTANT,
                    value=(0, 255, 0)  # Green border
                )
                
                # Add normal text
                cv2.putText(
                    display_img,
                    "NORMAL",
                    (int(display_img.shape[1]/2) - 80, 40),
                    self.font, 1.2, (0, 255, 0), 3
                )
                
                # Add score
                cv2.putText(
                    display_img,
                    f"Score: {anomaly_score:.4f}",
                    (20, 30),
                    self.font, 0.7, (255, 255, 255), 2
                )
            
            # Add inference time
            cv2.putText(
                display_img,
                f"Inference: {inference_time:.1f}ms",
                (20, 60),
                self.font, 0.7, (255, 255, 255), 2
            )
            
            # Save the output image
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_path, display_img)
            print(f"Saved result to {output_path}")
            
            return {
                'display_img': display_img,
                'anomaly_score': anomaly_score,
                'is_anomalous': is_anomalous,
                'inference_time': inference_time,
                'output_path': output_path
            }
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_folder(self, input_folder, output_folder="./outputs"):
        """
        Process all images in a folder
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Folder to save results
        """
        if not self.inferencer:
            print("No model loaded. Please load a model first.")
            return None
            
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return None
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for image_path in image_files:
            result = self.detect_anomalies_in_image(image_path, output_folder)
            if result:
                results.append(result)
        
        print(f"Processed {len(results)} images successfully")
        return results
    
    def process_video(self, video_path, output_dir="./outputs", output_video=True, skip_frames=0):
        """
        Process a video file for anomaly detection
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save results
            output_video: Whether to save a processed video
            skip_frames: Number of frames to skip between processing (0 = process all frames)
        """
        if not self.inferencer:
            print("No model loaded. Please load a model first.")
            return None
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Initialize video writer if needed
        output_video_path = None
        video_writer = None
        
        if output_video:
            # Create output video filename
            video_filename = os.path.basename(video_path)
            video_name, _ = os.path.splitext(video_filename)
            output_video_path = os.path.join(output_dir, f"result_{video_name}.mp4")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            print(f"Saving output video to {output_video_path}")
        
        # Initialize metrics tracking
        anomaly_frames = 0
        normal_frames = 0
        frame_times = []
        frame_scores = []
        
        # Process frames
        frame_idx = 0
        processing_frame_idx = 0
        
        print("Processing video... Press 'q' to stop")
        
        try:
            # Create window for display
            cv2.namedWindow("Video Processing", cv2.WINDOW_NORMAL)
            
            # Initial background calibration (for moving objects detection)
            if skip_frames <= 0:
                for _ in range(min(30, total_frames // 3)):
                    ret, frame = cap.read()
                    if ret:
                        self.bg_subtractor.apply(frame)
            
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if skip_frames > 0 and frame_idx % skip_frames != 0:
                    frame_idx += 1
                    if video_writer:
                        video_writer.write(frame)  # Write original frame
                    continue
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                # Update frame counter for display
                processing_frame_idx += 1
                
                # Object detection with background subtraction
                fg_mask = self.bg_subtractor.apply(frame)
                
                # Clean up mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                
                # Threshold mask
                _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000 and area < (width * height * 0.5):  # Reasonable size
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.1:  # Filter irregular shapes
                                valid_contours.append(contour)
                
                # Save frame to temporary file for inference
                temp_img_path = os.path.join(output_dir, "temp_frame.jpg")
                cv2.imwrite(temp_img_path, frame)
                
                # Start inference timer
                inference_start = time.time()
                
                # Run inference
                result = self.inferencer.predict(image=temp_img_path)
                
                # Process result
                if isinstance(result.pred_score, torch.Tensor):
                    anomaly_score = float(result.pred_score.cpu().item())
                else:
                    anomaly_score = float(result.pred_score)
                    
                if isinstance(result.pred_label, torch.Tensor):
                    is_anomalous = bool(result.pred_label.cpu().item())
                else:
                    is_anomalous = bool(result.pred_label)
                
                # Stabilize detection
                if is_anomalous and anomaly_score > self.threshold and len(valid_contours) > 0:
                    self.anomaly_confidence += 1
                    self.anomaly_confidence = min(self.anomaly_confidence, 5)
                else:
                    self.anomaly_confidence -= 1
                    self.anomaly_confidence = max(self.anomaly_confidence, 0)
                
                if self.anomaly_confidence >= 2:
                    self.anomaly_detected = True
                    self.anomaly_cooldown = 5
                    anomaly_frames += 1
                elif self.anomaly_cooldown > 0:
                    self.anomaly_cooldown -= 1
                    self.anomaly_detected = True
                    anomaly_frames += 1
                else:
                    self.anomaly_detected = False
                    normal_frames += 1
                
                # End inference timer
                inference_time = (time.time() - inference_start) * 1000
                frame_times.append(inference_time)
                frame_scores.append(anomaly_score)
                
                # Add info text
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_idx}/{total_frames}",
                    (20, 30),
                    self.font, 0.7, (255, 255, 255), 2
                )
                
                cv2.putText(
                    display_frame,
                    f"Score: {anomaly_score:.4f}",
                    (20, 60),
                    self.font, 0.7, (255, 255, 255), 2
                )
                
                cv2.putText(
                    display_frame,
                    f"Inference: {inference_time:.1f}ms",
                    (20, 90),
                    self.font, 0.7, (255, 255, 255), 2
                )
                
                # Process display based on detection result
                if self.anomaly_detected:
                    # Add red border
                    border_size = 5
                    display_frame = cv2.copyMakeBorder(
                        display_frame,
                        border_size, border_size, border_size, border_size,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 255)  # Red border
                    )
                    
                    # Add header
                    cv2.putText(
                        display_frame,
                        "ANOMALY DETECTED",
                        (width // 2 - 150, 40),
                        self.font, 1.2, (0, 0, 255), 3
                    )
                    
                    # Draw bounding boxes
                    for contour in valid_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Skip tiny or huge boxes
                        if w < 20 or h < 20 or w > width * 0.8 or h > height * 0.8:
                            continue
                            
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        
                        # Add label for significant regions
                        if w > 50 and h > 50:
                            cv2.putText(
                                display_frame,
                                "ANOMALY",
                                (x, max(y-10, 15)),
                                self.font, 0.5, (0, 0, 255), 2
                            )
                    
                    # Apply heatmap if available
                    if hasattr(result, 'pred_mask') and result.pred_mask is not None:
                        try:
                            # Process mask
                            mask = result.pred_mask.cpu().numpy()
                            if len(mask.shape) > 2:
                                mask = mask[0]
                                
                            # Resize mask to match frame
                            mask_resized = cv2.resize(mask, (width, height))
                            
                            # Create heatmap
                            heatmap = cv2.applyColorMap(
                                (mask_resized * 255).astype(np.uint8),
                                cv2.COLORMAP_JET
                            )
                            
                            # Apply heatmap only to detected object areas
                            mask_objects = np.zeros_like(frame, dtype=np.uint8)
                            for contour in valid_contours:
                                cv2.drawContours(mask_objects, [contour], 0, (255, 255, 255), -1)
                                
                            # Apply mask to heatmap
                            masked_heatmap = cv2.bitwise_and(heatmap, mask_objects)
                            
                            # Create overlay
                            display_frame = cv2.addWeighted(display_frame, 0.7, masked_heatmap, 0.3, 0)
                        except Exception as e:
                            print(f"Error applying heatmap: {e}")
                else:
                    # Normal state - green border
                    border_size = 5
                    display_frame = cv2.copyMakeBorder(
                        display_frame,
                        border_size, border_size, border_size, border_size,
                        cv2.BORDER_CONSTANT,
                        value=(0, 255, 0)  # Green border
                    )
                    
                    # Add normal text
                    cv2.putText(
                        display_frame,
                        "NORMAL",
                        (width // 2 - 80, 40),
                        self.font, 1.2, (0, 255, 0), 3
                    )
                
                # Save to output video if requested
                if video_writer:
                    video_writer.write(display_frame)
                
                # Save anomaly frames as images
                if self.anomaly_detected and frame_idx % 10 == 0:  # Save every 10th anomaly frame
                    anomaly_img_path = os.path.join(
                        output_dir, 
                        f"anomaly_frame_{frame_idx:06d}.jpg"
                    )
                    cv2.imwrite(anomaly_img_path, display_frame)
                
                # Display the frame
                cv2.imshow("Video Processing", display_frame)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Processing stopped by user")
                    break
                
                # Update frame counter
                frame_idx += 1
                
                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
        
        except Exception as e:
            print(f"Error processing video: {e}")
        
        finally:
            # Release resources
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Generate summary report
            if frame_times:
                # Create summary dictionary
                summary = {
                    'video_path': video_path,
                    'output_path': output_video_path,
                    'frames_processed': processing_frame_idx,
                    'normal_frames': normal_frames,
                    'anomaly_frames': anomaly_frames,
                    'avg_inference_time': np.mean(frame_times),
                    'avg_anomaly_score': np.mean(frame_scores),
                    'max_anomaly_score': np.max(frame_scores),
                    'anomaly_percentage': (anomaly_frames / processing_frame_idx) * 100 if processing_frame_idx > 0 else 0
                }
                
                # Print summary
                print("\n=== Video Processing Summary ===")
                print(f"Video: {os.path.basename(video_path)}")
                print(f"Frames Processed: {processing_frame_idx}/{total_frames}")
                print(f"Normal Frames: {normal_frames} ({normal_frames/processing_frame_idx*100:.2f}%)")
                print(f"Anomaly Frames: {anomaly_frames} ({anomaly_frames/processing_frame_idx*100:.2f}%)")
                print(f"Average Inference Time: {np.mean(frame_times):.2f} ms")
                print(f"Average Anomaly Score: {np.mean(frame_scores):.4f}")
                print(f"Maximum Anomaly Score: {np.max(frame_scores):.4f}")
                print("================================")
                
                # Plot metrics
                plt.figure(figsize=(12, 8))
                
                # Plot anomaly scores
                plt.subplot(2, 1, 1)
                plt.plot(frame_scores)
                plt.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold})')
                plt.title('Anomaly Scores')
                plt.xlabel('Frame')
                plt.ylabel('Score')
                plt.grid(True)
                plt.legend()
                
                # Plot inference times
                plt.subplot(2, 1, 2)
                plt.plot(frame_times)
                plt.title('Inference Times')
                plt.xlabel('Frame')
                plt.ylabel('Time (ms)')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"metrics_{os.path.basename(video_path)}.png"))
                
                return summary
            
            return None
    
    def run_webcam(self, output_dir="./outputs", camera_id=0):
        """
        Run real-time anomaly detection using webcam
        
        Args:
            output_dir: Directory to save outputs
            camera_id: Camera device ID (usually 0 for default webcam)
        """
        if not self.inferencer:
            print("No model loaded. Please load a model first.")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize webcam
        print(f"Initializing camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}. Please check connection.")
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual resolution (may differ from requested)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create window
        cv2.namedWindow('Anomaly Detection', cv2.WINDOW_NORMAL)
        
        # Frame counter for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Initial background calibration
        print("Calibrating background (please keep camera view clear)...")
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                self.bg_subtractor.apply(frame)
                cv2.waitKey(30)
        
        print("Starting real-time detection:")
        print("• Press 'q' to quit")
        print("• Press 's' to save current frame")
        print("• Press 'r' to reset background model")
        print("• Press 'v' to start/stop recording")
        
        # Recording variables
        recording = False
        video_writer = None
        record_start_time = None
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image from camera")
                    break
                
                # Create a copy of the frame for drawing
                display_frame = frame.copy()
                
                # Start inference timer
                inference_start = time.time()
                
                # Object detection with improved foreground mask processing
                fg_mask = self.bg_subtractor.apply(frame)
                
                # Clean up the mask with morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                
                # Threshold the mask
                _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                
                # Find contours with hierarchy to identify only parent contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by size and shape
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000 and area < 100000:  # Reasonable size range
                        # Calculate contour properties
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.1:  # Filter out very irregular shapes
                                valid_contours.append(contour)
                
                # Save frame to temporary file for inference
                temp_img_path = os.path.join(output_dir, "temp_frame.jpg")
                cv2.imwrite(temp_img_path, frame)
                
                # Predict
                result = self.inferencer.predict(image=temp_img_path)
                
                # Process result
                if isinstance(result.pred_score, torch.Tensor):
                    anomaly_score = float(result.pred_score.cpu().item())
                else:
                    anomaly_score = float(result.pred_score)
                    
                if isinstance(result.pred_label, torch.Tensor):
                    is_anomalous = bool(result.pred_label.cpu().item())
                else:
                    is_anomalous = bool(result.pred_label)
                
                # Stabilize anomaly detection with cooldown and confidence
                if is_anomalous and anomaly_score > self.threshold and len(valid_contours) > 0:
                    self.anomaly_confidence += 1
                    self.anomaly_confidence = min(self.anomaly_confidence, 10)  # Cap confidence
                else:
                    self.anomaly_confidence -= 1
                    self.anomaly_confidence = max(self.anomaly_confidence, 0)  # Floor at 0
                
                if self.anomaly_confidence >= 3:  # Need 3 consecutive detections to trigger
                    self.anomaly_detected = True
                    self.anomaly_cooldown = 30  # Stay in anomaly mode for 30 frames
                elif self.anomaly_cooldown > 0:
                    self.anomaly_cooldown -= 1
                    self.anomaly_detected = True
                else:
                    self.anomaly_detected = False
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    fps = frame_count / elapsed_time
                    self.fps_history.append(fps)
                    frame_count = 0
                    start_time = time.time()
                
                # Store anomaly score
                self.score_history.append(anomaly_score)
                if len(self.score_history) > 100:
                    self.score_history.pop(0)
                
                # End inference timer
                inference_time = (time.time() - inference_start) * 1000  # ms
                
                # Add info text
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 30), self.font, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Score: {anomaly_score:.4f}", (20, 60), self.font, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Inference: {inference_time:.1f}ms", (20, 90), self.font, 0.7, (255, 255, 255), 2)
                
                # Add recording indicator if recording
                if recording:
                    # Calculate recording duration
                    record_duration = time.time() - record_start_time
                    minutes = int(record_duration // 60)
                    seconds = int(record_duration % 60)
                    
                    # Add red recording indicator
                    cv2.circle(display_frame, (width - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        f"REC {minutes:02d}:{seconds:02d}",
                        (width - 120, 40),
                        self.font, 0.7, (0, 0, 255), 2
                    )
                
                if self.anomaly_detected:
                    # Show one prominent "ANOMALY DETECTED" header instead of multiple labels
                    # Add red border
                    border_size = 10
                    display_frame = cv2.copyMakeBorder(
                        display_frame, 
                        border_size, border_size, border_size, border_size,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 255)  # Red border
                    )
                    
                    # Add single header
                    cv2.putText(display_frame, "ANOMALY DETECTED", 
                               (int(display_frame.shape[1]/2) - 150, 40),
                               self.font, 1.2, (0, 0, 255), 3)
                    
                    # Draw bounding boxes only for valid contours
                    for contour in valid_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Skip tiny or huge bounding boxes
                        if w < 20 or h < 20 or w > display_frame.shape[1] * 0.8 or h > display_frame.shape[0] * 0.8:
                            continue
                            
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        
                        # Only add "ANOMALY" text to significant objects (avoid cluttering)
                        if w > 50 and h > 50:
                            cv2.putText(display_frame, "ANOMALY", (x, max(y-10, 15)), 
                                       self.font, 0.5, (0, 0, 255), 2)
                    
                    # Heat map overlay (only if available)
                    if hasattr(result, 'pred_mask') and result.pred_mask is not None:
                        try:
                            mask = result.pred_mask.cpu().numpy()
                            if len(mask.shape) > 2:
                                mask = mask[0]
                            
                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            
                            # Only apply heatmap to areas with detected objects
                            mask_objects = np.zeros_like(frame, dtype=np.uint8)
                            for contour in valid_contours:
                                cv2.drawContours(mask_objects, [contour], 0, (255, 255, 255), -1)
                            
                            # Apply colormap to mask
                            heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            
                            # Only apply heatmap to object areas
                            masked_heatmap = cv2.bitwise_and(heatmap, mask_objects)
                            
                            # Overlay
                            display_frame = cv2.addWeighted(display_frame, 0.7, masked_heatmap, 0.3, 0)
                        except Exception as e:
                            print(f"Error applying heatmap: {e}")
                else:
                    # Normal state - green border
                    border_size = 5
                    display_frame = cv2.copyMakeBorder(
                        display_frame,
                        border_size, border_size, border_size, border_size,
                        cv2.BORDER_CONSTANT,
                        value=(0, 255, 0)  # Green border
                    )
                    
                    # Add normal status text
                    cv2.putText(display_frame, "NORMAL", 
                               (int(display_frame.shape[1]/2) - 80, 40),
                               self.font, 1.2, (0, 255, 0), 3)
                
                # Write frame to video if recording
                if recording and video_writer:
                    video_writer.write(display_frame)
                
                # Show the result
                cv2.imshow('Anomaly Detection', display_frame)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    save_path = os.path.join(output_dir, f"anomaly_detection_{timestamp}.jpg")
                    cv2.imwrite(save_path, display_frame)
                    print(f"Saved current frame to {save_path}")
                elif key == ord('r'):
                    # Reset background model
                    print("Resetting background model...")
                    self.reset_background_model()
                elif key == ord('v'):
                    # Toggle recording
                    if not recording:
                        # Start recording
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        video_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (display_frame.shape[1], display_frame.shape[0]))
                        record_start_time = time.time()
                        recording = True
                        print(f"Started recording to {video_path}")
                    else:
                        # Stop recording
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        recording = False
                        print("Recording stopped")
        
        except Exception as e:
            print(f"Error during webcam processing: {e}")
        
        finally:
            # Release resources
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Remove temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            # Plot metrics if collected
            if self.fps_history:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(self.fps_history)
                plt.title('FPS over time')
                plt.xlabel('Seconds')
                plt.ylabel('Frames per Second')
                
                plt.subplot(1, 2, 2)
                plt.plot(self.score_history)
                plt.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold})')
                plt.title('Anomaly Score over time')
                plt.xlabel('Frames')
                plt.ylabel('Anomaly Score')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'webcam_metrics.png'))
                plt.show()


class AnomalyDetectionApp:
    """GUI application for anomaly detection"""
    
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("Anomaly Detection System")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Set app icon if available
        try:
            icon_path = "app_icon.ico"  # Replace with your icon path
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Create detector
        self.detector = AnomalyDetector(
            model_path=model_path, 
            threshold=0.7,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header with logo and title
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # App title
        title_label = ttk.Label(
            self.header_frame, 
            text="Product Anomaly Detection System",
            font=("Arial", 18, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.model_tab = ttk.Frame(self.notebook, padding=10)
        self.image_tab = ttk.Frame(self.notebook, padding=10)
        self.video_tab = ttk.Frame(self.notebook, padding=10)
        self.webcam_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.image_tab, text="Image")
        self.notebook.add(self.video_tab, text="Video")
        self.notebook.add(self.webcam_tab, text="Webcam")
        
        # Create contents for each tab
        self.setup_model_tab()
        self.setup_image_tab()
        self.setup_video_tab()
        self.setup_webcam_tab()
        
        # Create status bar
        self.status_var = tk.StringVar()
        if model_path:
            self.status_var.set(f"Model: {os.path.basename(model_path)} | Device: {self.detector.device}")
        else:
            self.status_var.set("No model loaded | Device: " + ("GPU" if torch.cuda.is_available() else "CPU"))
        
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Setup is complete
        print("Anomaly Detection System initialized")
        
    def setup_model_tab(self):
        """Setup the model tab for loading and configuring models"""
        # Model selection frame
        model_frame = ttk.LabelFrame(self.model_tab, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=10)
        
        # Model path entry
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.model_path_var = tk.StringVar()
        if self.detector.model_path:
            self.model_path_var.set(self.detector.model_path)
        
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Browse button
        def browse_model():
            file_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
            )
            if file_path:
                self.model_path_var.set(file_path)
        
        browse_btn = ttk.Button(model_frame, text="Browse", command=browse_model)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Load button
        def load_model():
            model_path = self.model_path_var.get()
            if not model_path:
                messagebox.showerror("Error", "Please select a model file.")
                return
                
            try:
                if self.detector.load_model(model_path):
                    messagebox.showinfo("Success", f"Model loaded successfully: {os.path.basename(model_path)}")
                    self.status_var.set(f"Model: {os.path.basename(model_path)} | Device: {self.detector.device}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        
        load_btn = ttk.Button(model_frame, text="Load Model", command=load_model)
        load_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Model settings frame
        settings_frame = ttk.LabelFrame(self.model_tab, text="Model Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Threshold setting
        ttk.Label(settings_frame, text="Detection Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=self.detector.threshold)
        
        threshold_scale = ttk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        threshold_scale.grid(row=0, column=1, padx=5, pady=5)
        
        threshold_value = ttk.Label(settings_frame, text=f"{self.detector.threshold:.2f}")
        threshold_value.grid(row=0, column=2, padx=5, pady=5)
        
        # Update threshold value label when slider changes
        def update_threshold(event):
            threshold = self.threshold_var.get()
            threshold_value.config(text=f"{threshold:.2f}")
            self.detector.threshold = threshold
        
        threshold_scale.bind("<Motion>", update_threshold)
        
        # Device selection
        ttk.Label(settings_frame, text="Computation Device:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        
        device_cpu = ttk.Radiobutton(settings_frame, text="CPU", variable=self.device_var, value="cpu")
        device_cpu.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        device_gpu = ttk.Radiobutton(settings_frame, text="GPU (CUDA)", variable=self.device_var, value="cuda")
        device_gpu.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Disable GPU option if not available
        if not torch.cuda.is_available():
            device_gpu.config(state=tk.DISABLED)
        
        # Apply button for settings
        def apply_settings():
            threshold = self.threshold_var.get()
            device = self.device_var.get()
            
            self.detector.threshold = threshold
            
            # Only reload model if device changed
            if device != self.detector.device and self.detector.model_path:
                try:
                    self.detector.device = device
                    self.detector.load_model(self.detector.model_path)
                    messagebox.showinfo("Success", f"Model reloaded with device: {device}")
                    self.status_var.set(f"Model: {os.path.basename(self.detector.model_path)} | Device: {device}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to reload model: {str(e)}")
            
            messagebox.showinfo("Settings Applied", f"Threshold: {threshold:.2f}\nDevice: {device}")
        
        apply_btn = ttk.Button(settings_frame, text="Apply Settings", command=apply_settings)
        apply_btn.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Model info frame
        info_frame = ttk.LabelFrame(self.model_tab, text="Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Info text
        info_text = tk.Text(info_frame, wrap=tk.WORD, height=10, width=60)
        info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to info text
        info_scroll = ttk.Scrollbar(info_text, command=info_text.yview)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        info_text.config(yscrollcommand=info_scroll.set)
        
        # Add some instructions and information
        info_text.insert(tk.END, """Anomaly Detection System - Instructions

1. Load a PyTorch (.pt) model using the "Browse" and "Load Model" buttons.
2. Adjust the detection threshold as needed (higher values reduce false positives).
3. Select the computation device (GPU recommended if available).
4. Use the tabs above to process images, videos, or use webcam for real-time detection.

Note: This application requires a trained anomaly detection model created with Anomalib library.
""")
        
        info_text.config(state=tk.DISABLED)  # Make text read-only
        
    def setup_image_tab(self):
        """Setup the image tab for processing single images"""
        # Image selection frame
        image_select_frame = ttk.LabelFrame(self.image_tab, text="Image Selection", padding=10)
        image_select_frame.pack(fill=tk.X, pady=10)
        
        # Image path entry
        ttk.Label(image_select_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.image_path_var = tk.StringVar()
        
        image_entry = ttk.Entry(image_select_frame, textvariable=self.image_path_var, width=50)
        image_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Browse button
        def browse_image():
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All Files", "*.*")
                ]
            )
            if file_path:
                self.image_path_var.set(file_path)
        
        browse_btn = ttk.Button(image_select_frame, text="Browse", command=browse_image)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Process button
        def process_image():
            image_path = self.image_path_var.get()
            if not image_path:
                messagebox.showerror("Error", "Please select an image file.")
                return
                
            if not self.detector.inferencer:
                messagebox.showerror("Error", "No model loaded. Please load a model first.")
                return
                
            try:
                # Process image
                output_dir = os.path.join("outputs", "images")
                result = self.detector.detect_anomalies_in_image(image_path, output_dir)
                
                if result:
                    # Show result
                    messagebox.showinfo(
                        "Detection Complete", 
                        f"Anomaly Score: {result['anomaly_score']:.4f}\n"
                        f"Anomaly Detected: {'Yes' if result['is_anomalous'] else 'No'}\n"
                        f"Inference Time: {result['inference_time']:.2f}ms\n\n"
                        f"Result saved to: {result['output_path']}"
                    )
                    
                    # Open result image
                    os.startfile(result['output_path'])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
        
        process_btn = ttk.Button(image_select_frame, text="Process Image", command=process_image)
        process_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Batch processing frame
        batch_frame = ttk.LabelFrame(self.image_tab, text="Batch Processing", padding=10)
        batch_frame.pack(fill=tk.X, pady=10)
        
        # Folder path entry
        ttk.Label(batch_frame, text="Folder Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.folder_path_var = tk.StringVar()
        
        folder_entry = ttk.Entry(batch_frame, textvariable=self.folder_path_var, width=50)
        folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Browse button
        def browse_folder():
            folder_path = filedialog.askdirectory(title="Select Folder with Images")
            if folder_path:
                self.folder_path_var.set(folder_path)
        
        browse_folder_btn = ttk.Button(batch_frame, text="Browse", command=browse_folder)
        browse_folder_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Process button
        def process_folder():
            folder_path = self.folder_path_var.get()
            if not folder_path:
                messagebox.showerror("Error", "Please select a folder.")
                return
                
            if not self.detector.inferencer:
                messagebox.showerror("Error", "No model loaded. Please load a model first.")
                return
                
            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing Folder")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Center the progress window
            progress_window.update_idletasks()
            width = progress_window.winfo_width()
            height = progress_window.winfo_height()
            x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
            y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
            progress_window.geometry(f"{width}x{height}+{x}+{y}")
            
            # Progress message
            ttk.Label(progress_window, text="Processing images...").pack(pady=10)
            
            # Progress bar
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate', length=300)
            progress_bar.pack(pady=10, padx=20)
            progress_bar.start(10)
            
            # Cancel button
            cancel_var = tk.BooleanVar(value=False)
            
            def cancel_processing():
                cancel_var.set(True)
                progress_window.destroy()
            
            cancel_btn = ttk.Button(progress_window, text="Cancel", command=cancel_processing)
            cancel_btn.pack(pady=10)
            
            # Run processing in a separate thread
            def run_processing():
                try:
                    # Process folder
                    output_folder = os.path.join("outputs", "batch")
                    results = self.detector.process_folder(folder_path, output_folder)
                    
                    # Check if processing was cancelled
                    if cancel_var.get():
                        return
                    
                    # Close progress window
                    progress_window.destroy()
                    
                    # Show results
                    if results:
                        messagebox.showinfo(
                            "Batch Processing Complete",
                            f"Processed {len(results)} images\n"
                            f"Results saved to: {output_folder}"
                        )
                        
                        # Open output folder
                        os.startfile(output_folder)
                    else:
                        messagebox.showwarning(
                            "Batch Processing Result",
                            "No images were processed successfully."
                        )
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Error", f"Failed to process folder: {str(e)}")
            
            # Start processing thread
            threading.Thread(target=run_processing, daemon=True).start()
        
        process_folder_btn = ttk.Button(batch_frame, text="Process Folder", command=process_folder)
        process_folder_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.image_tab, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Open results folder button
        def open_results_folder():
            output_dir = os.path.join("outputs", "images")
            os.makedirs(output_dir, exist_ok=True)
            os.startfile(output_dir)
        
        open_results_btn = ttk.Button(
            results_frame, 
            text="Open Results Folder", 
            command=open_results_folder
        )
        open_results_btn.pack(pady=10)
        
    def setup_video_tab(self):
        """Setup the video tab for processing video files"""
        # Video selection frame
        video_select_frame = ttk.LabelFrame(self.video_tab, text="Video Selection", padding=10)
        video_select_frame.pack(fill=tk.X, pady=10)
        
        # Video path entry
        ttk.Label(video_select_frame, text="Video Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.video_path_var = tk.StringVar()
        
        video_entry = ttk.Entry(video_select_frame, textvariable=self.video_path_var, width=50)
        video_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Browse button
        def browse_video():
            file_path = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[
                    ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All Files", "*.*")
                ]
            )
            if file_path:
                self.video_path_var.set(file_path)
        
        browse_btn = ttk.Button(video_select_frame, text="Browse", command=browse_video)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Video settings frame
        video_settings_frame = ttk.LabelFrame(self.video_tab, text="Processing Options", padding=10)
        video_settings_frame.pack(fill=tk.X, pady=10)
        
        # Frame skip setting
        ttk.Label(video_settings_frame, text="Process every N frames:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.frame_skip_var = tk.IntVar(value=1)
        
        frame_skip_combo = ttk.Combobox(
            video_settings_frame,
            textvariable=self.frame_skip_var,
            values=[1, 2, 5, 10, 15, 30],
            width=5
        )
        frame_skip_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Save output video option
        self.save_video_var = tk.BooleanVar(value=True)
        
        save_video_check = ttk.Checkbutton(
            video_settings_frame,
            text="Save processed video",
            variable=self.save_video_var
        )
        save_video_check.grid(row=0, column=2, padx=20, pady=5, sticky=tk.W)
        
        # Process button
        def process_video():
            video_path = self.video_path_var.get()
            if not video_path:
                messagebox.showerror("Error", "Please select a video file.")
                return
                
            if not self.detector.inferencer:
                messagebox.showerror("Error", "No model loaded. Please load a model first.")
                return
            
            # Get processing options
            frame_skip = self.frame_skip_var.get()
            save_video = self.save_video_var.get()
            
            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing Video")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Center the progress window
            progress_window.update_idletasks()
            width = progress_window.winfo_width()
            height = progress_window.winfo_height()
            x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
            y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
            progress_window.geometry(f"{width}x{height}+{x}+{y}")
            
            # Progress message
            progress_label = ttk.Label(progress_window, text="Initializing video processing...")
            progress_label.pack(pady=10)
            
            # Progress bar
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate', length=300)
            progress_bar.pack(pady=10, padx=20)
            progress_bar.start(10)
            
            # Status label
            status_label = ttk.Label(progress_window, text="")
            status_label.pack(pady=5)
            
            # Cancel button
            cancel_var = tk.BooleanVar(value=False)
            
            def cancel_processing():
                cancel_var.set(True)
                status_label.config(text="Canceling...")
            
            cancel_btn = ttk.Button(progress_window, text="Cancel", command=cancel_processing)
            cancel_btn.pack(pady=10)
            
            # Run processing in a separate thread
            def run_processing():
                try:
                    # Process video
                    output_dir = os.path.join("outputs", "videos")
                    
                    # Update progress intermittently
                    def update_status(message):
                        if not cancel_var.get():
                            status_label.config(text=message)
                    
                    # Start processing with periodic updates
                    update_status("Processing video...")
                    
                    # Call video processing function
                    result = self.detector.process_video(
                        video_path=video_path,
                        output_dir=output_dir,
                        output_video=save_video,
                        skip_frames=frame_skip - 1  # Convert to skip value (0 = process all)
                    )
                    
                    # Check if processing was cancelled
                    if cancel_var.get():
                        update_status("Processing cancelled.")
                        time.sleep(1)
                        progress_window.destroy()
                        return
                    
                    # Close progress window
                    progress_window.destroy()
                    
                    # Show results
                    if result:
                        messagebox.showinfo(
                            "Video Processing Complete",
                            f"Processed video: {os.path.basename(video_path)}\n"
                            f"Normal Frames: {result['normal_frames']}\n"
                            f"Anomaly Frames: {result['anomaly_frames']}\n"
                            f"Average Score: {result['avg_anomaly_score']:.4f}\n\n"
                            f"Results saved to: {output_dir}"
                        )
                        
                        # Open output folder
                        os.startfile(output_dir)
                    else:
                        messagebox.showwarning(
                            "Video Processing Result",
                            "Video processing did not complete successfully."
                        )
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Error", f"Failed to process video: {str(e)}")
            
            # Start processing thread
            threading.Thread(target=run_processing, daemon=True).start()
        
        process_btn = ttk.Button(video_settings_frame, text="Process Video", command=process_video)
        process_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.video_tab, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Open results folder button
        def open_video_results_folder():
            output_dir = os.path.join("outputs", "videos")
            os.makedirs(output_dir, exist_ok=True)
            os.startfile(output_dir)
        
        open_results_btn = ttk.Button(
            results_frame, 
            text="Open Results Folder", 
            command=open_video_results_folder
        )
        open_results_btn.pack(pady=10)
        
    def setup_webcam_tab(self):
        """Setup the webcam tab for real-time detection"""
        # Webcam selection frame
        webcam_select_frame = ttk.LabelFrame(self.webcam_tab, text="Camera Settings", padding=10)
        webcam_select_frame.pack(fill=tk.X, pady=10)
        
        # Camera selection
        ttk.Label(webcam_select_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.camera_id_var = tk.IntVar(value=0)
        
        camera_combo = ttk.Combobox(
            webcam_select_frame,
            textvariable=self.camera_id_var,
            values=[0, 1, 2, 3],
            width=5
        )
        camera_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(webcam_select_frame, text="(0 = Default, try other numbers for additional cameras)").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Start/Stop webcam button
        self.webcam_running = False
        
        def toggle_webcam():
            if not self.detector.inferencer:
                messagebox.showerror("Error", "No model loaded. Please load a model first.")
                return
                
            if not self.webcam_running:
                # Start webcam in a separate thread
                def run_webcam():
                    try:
                        # Hide the main window temporarily
                        self.root.withdraw()
                        
                        # Start webcam
                        camera_id = self.camera_id_var.get()
                        output_dir = os.path.join("outputs", "webcam")
                        self.detector.run_webcam(output_dir=output_dir, camera_id=camera_id)
                        
                        # Show main window again
                        self.root.deiconify()
                        
                        # Reset button text
                        self.webcam_running = False
                        webcam_btn.config(text="Start Camera")
                    except Exception as e:
                        self.root.deiconify()
                        messagebox.showerror("Error", f"Webcam error: {str(e)}")
                        self.webcam_running = False
                        webcam_btn.config(text="Start Camera")
                
                # Update button state
                self.webcam_running = True
                webcam_btn.config(text="Starting...")
                
                # Start thread
                threading.Thread(target=run_webcam, daemon=True).start()
            else:
                # This is handled automatically when webcam window is closed
                pass
        
        webcam_btn = ttk.Button(
            webcam_select_frame, 
            text="Start Camera", 
            command=toggle_webcam
        )
        webcam_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Instructions frame
        instructions_frame = ttk.LabelFrame(self.webcam_tab, text="Instructions", padding=10)
        instructions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Instructions text
        instructions_text = tk.Text(instructions_frame, wrap=tk.WORD, height=10, width=60)
        instructions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        instructions_scroll = ttk.Scrollbar(instructions_text, command=instructions_text.yview)
        instructions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        instructions_text.config(yscrollcommand=instructions_scroll.set)
        
        # Add instructions
        instructions_text.insert(tk.END, """
Camera Controls:
- Press 'q' to quit camera view
- Press 's' to save current frame as image
- Press 'r' to reset background model (use if detection seems stuck)
- Press 'v' to start/stop video recording

Tips:
- Keep the camera view clear when starting to calibrate the background
- If false detections occur, try increasing the detection threshold
- For best results, ensure good lighting and minimal background movement
- The system works best with consistent lighting conditions
- If the camera doesn't start, try changing the camera number

Results:
- Images and videos will be saved in the outputs/webcam folder
- Performance metrics will be shown after closing the camera
""")
        
        instructions_text.config(state=tk.DISABLED)  # Make text read-only

def main():
    """Main function to run the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Anomaly Detection System")
    parser.add_argument("--model", type=str, help="Path to the model file (.pt)")
    
    args = parser.parse_args()
    
    # Create Tkinter root window
    root = tk.Tk()
    
    # Create app with model path if provided
    app = AnomalyDetectionApp(root, model_path=args.model)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()