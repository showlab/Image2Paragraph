import gradio as gr
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from models.image_text_transformation import ImageTextTransformation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_src', default='examples/1.jpg')
parser.add_argument('--out_image_name', default='output/1_result.jpg')
parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=True, help='Set this flag to True if you want to use semantic segmentation')
parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_b', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
parser.add_argument('--dense_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, <6G GPU is not recommended>')

args = parser.parse_args()

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def add_logo():
    with open("examples/logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    return logo_base64

def process_image(image_src, options, devices, processor):
    processor.args.image_caption = "Image Caption" in options
    processor.args.dense_caption = "Dense Caption" in options
    processor.args.semantic_segment = "Semantic Segment" in options
    processor.args.image_caption_device = "cuda" if "cuda_ic" in devices else "cpu"
    processor.args.dense_caption_device = "cuda" if "cuda_dc" in devices else "cpu"
    processor.args.semantic_segment_device = "cuda" if "cuda_ss" in devices else "cpu"
    processor.args.contolnet_device = "cuda" if "cuda_cn" in devices else "cpu"
    gen_text = processor.image_to_text(image_src)
    gen_image = processor.text_to_image(gen_text)
    gen_image_str = pil_image_to_base64(gen_image)
    # Combine the outputs into a single HTML output
    custom_output = f'''
    <h2>Image->Text->Image:</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1;">
            <h3>Image2Text</h3>
            <p>{gen_text}</p>
        </div>
        <div style="flex: 1;">
            <h3>Text2Image</h3>
            <img src="data:image/jpeg;base64,{gen_image_str}" width="100%" />
        </div>
    </div>
    '''

    return custom_output

processor = ImageTextTransformation(args)

# Create Gradio input and output components
image_input = gr.inputs.Image(type='filepath', label="Input Image")
image_caption_checkbox = gr.inputs.Checkbox(label="Image Caption", default=True)
dense_caption_checkbox = gr.inputs.Checkbox(label="Dense Caption", default=True)
semantic_segment_checkbox = gr.inputs.Checkbox(label="Semantic Segment", default=False)
image_caption_device = gr.inputs.Radio(choices=['cuda', 'cpu'], default='cpu', label='Image Caption Device')
dense_caption_device = gr.inputs.Radio(choices=['cuda', 'cpu'], default='cuda', label='Dense Caption Device')
semantic_segment_device = gr.inputs.Radio(choices=['cuda', 'cpu'], default='cuda', label='Semantic Segment Device')
controlnet_device = gr.inputs.Radio(choices=['cuda', 'cpu'], default='cpu', label='ControlNet Device')


logo_base64 = add_logo()
# Create the title with the logo
title_with_logo = f'<img src="data:image/jpeg;base64,{logo_base64}" width="400" style="vertical-align: middle;"> Understanding Image with Text'

# Create Gradio interface
interface = gr.Interface(
    fn=lambda image, options, devices: process_image(image, options, devices, processor),
    inputs=[image_input,        
            gr.CheckboxGroup(
            label="Options",
            choices=["Image Caption", "Dense Caption", "Semantic Segment"],
            ),
            gr.CheckboxGroup(
                label="Device, ic: image caption, dc: dense caption, ss: semantic segment, cn: controlnet",
                choices=["cuda_ic", "cuda_dc", "cuda_ss", "cuda_cn"],
                default=["cuda_dc", "cuda_cn"],
            )],
    outputs=gr.outputs.HTML(),
    title=title_with_logo,
    description="""
    This code support image to text transformation. Then the generated text can do retrieval, question answering et al to conduct zero-shot.
    \n Semantic segment is very slow in cpu, best use on gpu.
    \n If you have only 8GB Memory GPU, please set device as cpu for IC and SS.
    """
)

# Launch the interface
interface.launch()