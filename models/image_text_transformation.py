from models.blip2_model import ImageCaptioning
from models.grit_model import DenseCaptioning
from models.gpt_model import ImageToText
from models.controlnet_model import TextToImage
from models.region_semantic import RegionSemantic
from utils.util import read_image_width_height, display_images_and_text
import argparse
from PIL import Image
import base64
from io import BytesIO
import os

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


class ImageTextTransformation:
    def __init__(self):
        # Load your big model here
        self.init_models()
        self.ref_image = None
    
    def init_models(self):
        openai_key = os.environ['OPENAI_KEY']
        self.image_caption_model = ImageCaptioning()
        self.dense_caption_model = DenseCaptioning()
        self.gpt_model = ImageToText(openai_key)
        self.controlnet_model = TextToImage()
        self.region_semantic_model = RegionSemantic()

    
    def image_to_text(self, img_src):
        # the information to generate paragraph based on the context
        self.ref_image = Image.open(img_src)
        width, height = read_image_width_height(img_src)
        image_caption = self.image_caption_model.image_caption(img_src)
        dense_caption = self.dense_caption_model.image_dense_caption(img_src)
        region_semantic = self.region_semantic_model.region_semantic(img_src)
        generated_text = self.gpt_model.paragraph_summary_with_gpt(image_caption, dense_caption, region_semantic, width, height)
        return generated_text

    def text_to_image(self, text):
        generated_image = self.controlnet_model.text_to_image(text, self.ref_image)
        return generated_image

    def text_to_image_retrieval(self, text):
        pass

    def image_to_text_retrieval(self, image):
        pass