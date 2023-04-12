import gradio as gr
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from models.image_text_transformation import ImageTextTransformation

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def add_logo():
    with open("examples/logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    return logo_base64

def process_image(image_src, processor):
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
    <h2>Using Source Image to do Retrieval on COCO:</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1;">
            <h3>Retrieval Top-3 Text</h3>
            <p>{gen_text}</p>
        </div>
        <div style="flex: 1;">
            <h3>Retrieval Top-3 Image</h3>
            <img src="data:image/jpeg;base64,{gen_image_str}" width="100%" />
        </div>
    </div>
    <h2>Using Generated texts to do Retrieval on COCO:</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1;">
            <h3>Retrieval Top-3 Text</h3>
            <p>{gen_text}</p>
        </div>
        <div style="flex: 1;">
            <h3>Retrieval Top-3 Image</h3>
            <img src="data:image/jpeg;base64,{gen_image_str}" width="100%" />
        </div>
    </div>
    '''

    return custom_output

processor = ImageTextTransformation()

# Create Gradio input and output components
image_input = gr.inputs.Image(type='filepath', label="Input Image")

logo_base64 = add_logo()
# Create the title with the logo
title_with_logo = f'<img src="data:image/jpeg;base64,{logo_base64}" width="400" style="vertical-align: middle;"> Understanding Image with Text'

# Create Gradio interface
interface = gr.Interface(
    fn=lambda image: process_image(image, processor),  # Pass the processor object using a lambda function
    inputs=image_input,
    outputs=gr.outputs.HTML(),
    title=title_with_logo,
    description="""
    This code support image to text transformation. Then the generated text can do retrieval, question answering et al to conduct zero-shot.
    """
)

# Launch the interface
interface.launch()