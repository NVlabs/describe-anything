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

from segment_anything import sam_model_registry, SamPredictor
import gradio as gr
import numpy as np
import cv2
import base64
import torch
from PIL import Image
import io
import argparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from transformers import SamModel, SamProcessor
from dam import DescribeAnythingModel, disable_torch_init
try:
    from spaces import GPU
except ImportError:
    print("Spaces not installed, using dummy GPU decorator")
    def GPU(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

# Load SAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

@GPU(duration=75)
def image_to_sam_embedding(base64_image):
    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process image with SAM processor
        inputs = sam_processor(image, return_tensors="pt").to(device)
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = sam_model.get_image_embeddings(inputs["pixel_values"])
        
        # Convert to CPU and numpy
        image_embedding = image_embedding.cpu().numpy()
        
        # Encode the embedding as base64
        embedding_bytes = image_embedding.tobytes()
        embedding_base64 = base64.b64encode(embedding_bytes).decode('utf-8')
        
        return embedding_base64
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise gr.Error(f"Failed to process image: {str(e)}")

@GPU(duration=75)
def describe(image_base64: str, mask_base64: str, query: str):
    # Convert base64 to PIL Image
    image_bytes = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    mask_bytes = base64.b64decode(mask_base64.split(',')[1] if ',' in mask_base64 else mask_base64)
    mask = Image.open(io.BytesIO(mask_bytes))
    
    # Process the mask
    mask = Image.fromarray((np.array(mask.convert('L')) > 0).astype(np.uint8) * 255)
    
    # Get description using DAM with streaming
    description_generator = dam.get_description(
        img, 
        mask, 
        query, 
        streaming=True, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        num_beams=1, 
        max_new_tokens=512, 
    )
    
    # Stream the tokens
    text = ""
    for token in description_generator:
        text += token
        yield text
    
@GPU(duration=75)
def describe_without_streaming(image_base64: str, mask_base64: str, query: str):
    # Convert base64 to PIL Image
    image_bytes = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
    img = Image.open(io.BytesIO(image_bytes))
    mask_bytes = base64.b64decode(mask_base64.split(',')[1] if ',' in mask_base64 else mask_base64)
    mask = Image.open(io.BytesIO(mask_bytes))
    
    # Process the mask
    mask = Image.fromarray((np.array(mask.convert('L')) > 0).astype(np.uint8) * 255)
    
    # Get description using DAM
    description = dam.get_description(
        img, 
        mask, 
        query, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        num_beams=1, 
        max_new_tokens=512, 
    )
    
    return description

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe Anything gradio demo")
    parser.add_argument("--server_addr", "--host", type=str, default=None, help="The server address to listen on.")
    parser.add_argument("--server_port", "--port", type=int, default=None, help="The port to listen on.")
    parser.add_argument("--model-path", type=str, default="nvidia/DAM-3B", help="Path to the model checkpoint")
    parser.add_argument("--prompt-mode", type=str, default="full+focal_crop", help="Prompt mode")
    parser.add_argument("--conv-mode", type=str, default="v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p for sampling")

    args = parser.parse_args()

    # Initialize DAM model
    disable_torch_init()
    dam = DescribeAnythingModel(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=args.prompt_mode,
    ).to(device)

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Interface(
            fn=image_to_sam_embedding,
            inputs=gr.Textbox(label="Image Base64"),
            outputs=gr.Textbox(label="Embedding Base64"),
            title="Image Embedding Generator",
            api_name="image_to_sam_embedding"
        )
        gr.Interface(
            fn=describe,
            inputs=[
                gr.Textbox(label="Image Base64"),
                gr.Text(label="Mask Base64"),
                gr.Text(label="Prompt")
            ],
            outputs=[
                gr.Text(label="Description")
            ],
            title="Mask Description Generator",
            api_name="describe"
        )
        gr.Interface(
            fn=describe_without_streaming,
            inputs=[
                gr.Textbox(label="Image Base64"),
                gr.Text(label="Mask Base64"),
                gr.Text(label="Prompt")
            ],
            outputs=[
                gr.Text(label="Description")
            ],
            title="Mask Description Generator (Non-Streaming)",
            api_name="describe_without_streaming"
        )

    demo._block_thread = demo.block_thread
    demo.block_thread = lambda: None
    demo.launch(
        share=True,
        server_name=args.server_addr,
        server_port=args.server_port,
        ssr_mode=False,
    )

    for route in demo.app.routes:
        if route.path == "/":
            demo.app.routes.remove(route)
    demo.app.mount("/", StaticFiles(directory="dist", html=True), name="demo")

    demo._block_thread()
    