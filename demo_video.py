#!/usr/bin/env python3
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import gradio as gr
import numpy as np
import torch
from PIL import Image
import cv2
import os
import argparse
import tempfile

# Import DAM and disable torch init to speed up model loading (no effect on results)
from dam import DescribeAnythingModel, disable_torch_init
# Import SAM2 video predictor builder (make sure your PYTHONPATH is set correctly)
from sam2.build_sam import build_sam2_video_predictor

query = 'Video: <image><image><image><image><image><image><image><image>\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail.'

#############################################
# Utility functions (extracted from image demo)
#############################################

def extract_points_from_mask(mask_pil):
    # Select the first channel of the mask (expected to be a single-channel mask)
    mask = np.asarray(mask_pil)[..., 0]
    # Nonzero returns tuple (y, x); we swap and stack them as (x, y)
    coords = np.nonzero(mask)
    coords = np.stack((coords[1], coords[0]), axis=1)
    return coords

def add_contour(img, mask, color=(1.0, 1.0, 1.0)):
    img = img.copy()
    mask = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=6)
    return img

#############################################
# SAM2 propagation helper (adapted from the video code)
#############################################

def apply_sam2(image_files, points=None, box=None, normalized_coords=False):
    """Apply SAM2 to video frames using annotated points or bounding box on the first frame."""
    if normalized_coords:
        # If the coordinates are normalized, convert them to absolute coordinates
        first_frame = cv2.imread(image_files[0])
        height, width = first_frame.shape[:2]
        if points is not None:
            points = np.array(points, dtype=np.float32)
            points[:, 0] *= width
            points[:, 1] *= height
        elif box is not None:
            box = np.array(box, dtype=np.float32)
            box[0] *= width  # x1
            box[1] *= height # y1
            box[2] *= width  # x2
            box[3] *= height # y2

    # Use the directory of extracted frames to initialize SAM2
    video_dir = os.path.dirname(image_files[0])
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    ann_frame_idx = 0
    ann_obj_id = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if points is not None:
            # Convert points to numpy array and use positive labels
            points = np.array(points, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels
            )
        elif box is not None:
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box
            )
        
        # Propagate through the video frames and collect the predicted masks
        masks = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            masks.append(mask)
    return masks

#############################################
# Function to extract the first frame from a video file
#############################################

def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise gr.Error("Could not read the video file.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

#############################################
# Gradio main function: process the video and generate description
#############################################

def describe_video(video_path, annotated_frame):
    """
    Given the uploaded video file and the annotated first frame,
    extract frames from the video, use SAM2 to propagate the mask from the first frame,
    and call DAM to get the description for the video.
    """
    # Create a temporary directory to save extracted video frames
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    cap.release()

    if frame_count == 0:
        raise gr.Error("No frames were extracted from the video.")

    # Process the annotated frame from the image editor
    if isinstance(annotated_frame, dict):
        # Get the composite image with annotations
        frame_img = annotated_frame.get("image", annotated_frame.get("background"))
        if frame_img is None:
            raise gr.Error("No valid annotation found in the image editor.")
        frame_img = frame_img.convert("RGB")
        
        # Get the annotation layer
        if "layers" in annotated_frame and len(annotated_frame["layers"]) > 0:
            mask = Image.fromarray((np.asarray(annotated_frame["layers"][0])[..., 3] > 0).astype(np.uint8) * 255).convert("RGB")
        else:
            mask = Image.new("RGB", frame_img.size, 0)
    else:
        frame_img = annotated_frame.convert("RGB")
        mask = Image.new("RGB", frame_img.size, 0)

    # Extract points from the annotated mask (using the first channel)
    points = extract_points_from_mask(mask)
    np.random.seed(0)
    if points.shape[0] == 0:
        raise gr.Error("No points were selected in the annotation.")
    # Randomly select up to 8 points
    points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
    points = points[points_selected_indices]
    # Propagate the annotation to all video frames using SAM2
    masks = apply_sam2(frame_paths, points=points, normalized_coords=False)
    # Find frames with non-empty masks
    non_empty_mask_indices = [i for i, mask in enumerate(masks) if np.any(mask)]
    
    if not non_empty_mask_indices:
        raise gr.Error("No frames with non-empty masks were found.")
    
    # Uniformly sample 8 frames from those with non-empty masks
    if len(non_empty_mask_indices) <= 8:
        selected_indices = non_empty_mask_indices
    else:
        selected_indices = np.linspace(0, len(non_empty_mask_indices)-1, 8, dtype=int)
        selected_indices = [non_empty_mask_indices[i] for i in selected_indices]
    
    selected_frame_paths = [frame_paths[i] for i in selected_indices]
    selected_masks = [masks[i] for i in selected_indices]
    processed_images = [Image.open(fp).convert("RGB") for fp in selected_frame_paths]
    processed_masks = [Image.fromarray((m.squeeze() * 255).astype(np.uint8)) for m in selected_masks]

    # For visualization, draw the contour on the first selected frame
    first_frame_np = np.asarray(processed_images[0]).astype(float) / 255.0
    mask_np = np.asarray(processed_masks[0])[..., 0] / 255.0

    # Use fixed white color for contour
    color_rgb = (1.0, 1.0, 1.0)  # White color
    first_frame_with_contour_np = add_contour(first_frame_np, mask_np, color=color_rgb)
    output_frame = Image.fromarray((first_frame_with_contour_np * 255).astype(np.uint8))

    text = ""
    # Yield the initial outputs: annotated (first) frame with contour and an empty description
    yield output_frame, text

    # Call DAM to generate a description; streaming tokens are yielded progressively
    for token in dam.get_description(
            processed_images,
            processed_masks,
            query,
            streaming=True,
            temperature=args_cli.temperature,
            top_p=args_cli.top_p,
            num_beams=1,
            max_new_tokens=512):
        text += token
        yield gr.update(), text

    yield output_frame, text

#############################################
# Main Gradio demo UI and model initialization
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe Anything Video Demo")
    parser.add_argument("--server_addr", "--host", type=str, default="127.0.0.1", help="Server address to listen on.")
    parser.add_argument("--server_port", "--port", type=int, default=7860, help="Port to listen on.")
    parser.add_argument("--model-path", type=str, default="nvidia/DAM-3B-Video", help="Path to the DAM model checkpoint")
    parser.add_argument("--prompt-mode", type=str, default="focal_prompt", help="Prompt mode")
    parser.add_argument("--conv-mode", type=str, default="v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p for sampling")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the SAM2 predictor for video (using SAM2.1 hiera large checkpoint and config)
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Disable model weight initialization (for faster model loading)
    disable_torch_init()

    # Initialize the Describe Anything Model (DAM)
    prompt_modes = {"focal_prompt": "full+focal_crop"}
    dam = DescribeAnythingModel(
        model_path=args_cli.model_path,
        conv_mode=args_cli.conv_mode,
        prompt_mode=prompt_modes[args_cli.prompt_mode],
    ).to(device)

    # Build the Gradio demo
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
        f"""
        # 🎥 Describe Anything Video Demo
        
        ## 🚀 Overview
        This demo allows you to upload a video and get detailed descriptions of objects within it. The system uses advanced AI models to track and describe objects across video frames.

        **DAM Model Path:** `{args_cli.model_path}`

        ## 📝 Instructions
        1. **Upload** a video file (MP4 format recommended)
        2. Click **"Load First Frame"** to extract the first frame
        3. **Draw** on the first frame to indicate the region of interest (click the brush button and draw on the image)
        4. Click **"Describe"** to generate the description

        ⚠️ **Note:** This demo only supports annotation on the first frame. For more advanced features like annotation on any frame, please use `examples/dam_video_with_sam2.py`.
        
        ⚠️ **Note:** Do not upload videos that are too long. Otherwise SAM2 will be very slow.
        """)
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="📹 Upload Video", height=300)
                gr.Examples(
                    examples=["videos/1.mp4"],
                    inputs=video_input,
                    label="🎥 Demo Video",
                    examples_per_page=1
                )
                load_btn = gr.Button("🖼️ Load First Frame", variant="secondary")
            with gr.Column():
                first_frame_editor = gr.ImageEditor(
                    label="✏️ Annotate First Frame",
                    type="pil",
                    sources=["upload"],
                    brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=20),
                    eraser=False,
                    layers=True,
                    transforms=[],
                    height=768,
                )
                describe_btn = gr.Button("✨ Describe", variant="primary")
                output_image = gr.Image(label="🖼️ First Frame with Annotation", visible=False)
                output_description = gr.Textbox(label="📝 Description", visible=True, lines=5)

        # When the video is uploaded and user clicks load, extract the first frame and populate the image editor.
        load_btn.click(
            fn=load_first_frame,
            inputs=video_input,
            outputs=first_frame_editor,
        )

        # When the user clicks "Describe", run the full process.
        describe_btn.click(
            fn=describe_video,
            inputs=[video_input, first_frame_editor],
            outputs=[output_image, output_description],
            api_name="describe_video"
        )

    demo.launch(
        share=True,
        server_name=args_cli.server_addr,
        server_port=args_cli.server_port,
    )