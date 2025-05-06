# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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
from transformers import SamModel, SamProcessor
from PIL import Image
import os
import cv2
import argparse
# This is for making model initialization faster and has no effect since we are loading the weights
from dam import DescribeAnythingModel, disable_torch_init

prompt_modes = {
    "focal_prompt": "full+focal_crop",
}

def extract_points_from_mask(mask_pil):
    # Select the first channel of the mask
    mask = np.asarray(mask_pil)[..., 0]

    # coords is in (y_arr, x_arr) format
    coords = np.nonzero(mask)

    # coords is in [(x, y), ...] format
    coords = np.stack((coords[1], coords[0]), axis=1)

    # print(coords)

    return coords

def add_contour(img, mask, color=(1., 1., 1.)):
    img = img.copy()

    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=6)

    return img

def describe(image, skip_sam, query, contour_color):
    # Create an image object from the uploaded image
    print(image.keys())
    
    image['image'] = image['background'].convert('RGB')
    del image['background'], image['composite']
    assert len(image['layers']) == 1, f"Expected 1 layer, got {len(image['layers'])}"

    mask = Image.fromarray((np.asarray(image['layers'][0])[..., 3] > 0).astype(np.uint8) * 255).convert('RGB')
    if not skip_sam:
        points = extract_points_from_mask(mask)

        np.random.seed(0)

        if points.shape[0] == 0:
            raise gr.Error("No points selected")
        
        # Randomly sample 8 points from the mask
        points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
        points = points[points_selected_indices]

        print(f"Selected points (to SAM): {points}")

        coords = [points.tolist()]

        mask_np = apply_sam(image['image'], coords)
        mask = Image.fromarray(mask_np)
    else:
        mask_np = np.asarray(mask)[..., 0]
        mask = Image.fromarray(mask_np)

    img_np = np.asarray(image['image']).astype(float) / 255.

    # Handle both hex and rgba color formats
    if contour_color.startswith('#'):
        color_hex = contour_color.lstrip('#')
        color_rgb = tuple(int(color_hex[i:i+2], 16)/255.0 for i in (0, 2, 4))
    else:
        # Handle rgba format: rgba(r, g, b, a)
        import re
        rgba_match = re.match(r'rgba?\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)(?:,\s*[\d.]+)?\)', contour_color)
        if rgba_match:
            r, g, b = map(float, rgba_match.groups()[:3])
            color_rgb = (r/255.0, g/255.0, b/255.0)
        else:
            # Default to white if format is unknown
            color_rgb = (1.0, 1.0, 1.0)
    img_with_contour_np = add_contour(img_np, mask_np, color=color_rgb)
    img_with_contour_pil = Image.fromarray((img_with_contour_np * 255.).astype(np.uint8))

    img = image['image']
    
    # Get the description generator
    description_generator = dam.get_description(
        img, 
        mask, 
        query, 
        streaming=True, 
        temperature=args_cli.temperature, 
        top_p=args_cli.top_p, 
        num_beams=1, 
        max_new_tokens=512, 
    )
    
    # Initialize empty text
    text = ""
    yield img_with_contour_pil, text
    
    # Stream the tokens, but keep the same image
    for token in description_generator:
        text += token
        yield gr.update(), text

def apply_sam(image, input_points):
    inputs = sam_processor(image, input_points=input_points, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np


# Example
# python demo_simple.py --model-path nvidia/DAM-3B --server_addr 0.0.0.0 --server_port 7860

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe Anything gradio demo")
    parser.add_argument("--server_addr", "--host", type=str, default="127.0.0.1", help="The server address to listen on.")
    parser.add_argument("--server_port", "--port", type=int, default=7860, help="The port to listen on.")
    parser.add_argument("--model-path", type=str, default="nvidia/DAM-3B", help="Path to the model checkpoint")
    parser.add_argument("--prompt-mode", type=str, default="focal_prompt", help="Prompt mode")
    parser.add_argument("--conv-mode", type=str, default="v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p for sampling")

    args_cli = parser.parse_args()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown(
        f"""
        # Describe Anything Demo
        Model path: {args_cli.model_path}
        
        Please upload or select an example image and scribble on the image.
        """)
        with gr.Row():
            with gr.Column():
                image_input = gr.ImageEditor(
                    type="pil", 
                    sources=['upload'], 
                    brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=20),
                    eraser=False,
                    layers=False,
                    transforms=[],
                    height=768,
                )
                skip_sam = gr.Checkbox(label="Use scribble as the mask directly (skipping SAM), please ensure to fill in the region completely", value=False)
                query = gr.Textbox(label="Prompt", value="<image>\nDescribe the masked region in detail.", interactive=True)
                contour_color = gr.ColorPicker(label="Contour Color", value="#FFFFFF")
                submit_btn = gr.Button("Describe", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Image with Region", visible=True)
                description = gr.Textbox(label="Description", visible=True)

        with gr.Row():
            with gr.Column():
                gr.Examples([f"images/{i+1}.jpg" for i in range(20)], inputs=image_input, label="Examples")

        submit_btn.click(
            fn=describe,
            inputs=[image_input, skip_sam, query, contour_color],
            outputs=[output_image, description],
            api_name="describe"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    disable_torch_init()

    print(f"Using model {args_cli.model_path}")
    
    dam = DescribeAnythingModel(
        model_path=args_cli.model_path, 
        conv_mode=args_cli.conv_mode, 
        prompt_mode=prompt_modes[args_cli.prompt_mode], 
    )

    demo.launch(
        share=True,
        server_name=args_cli.server_addr,
        server_port=args_cli.server_port
    )
