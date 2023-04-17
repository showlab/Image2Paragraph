from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import textwrap
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def read_image_width_height(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height

def resize_long_edge(image, target_size=384):
    # Calculate the aspect ratio
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    # Determine the new dimensions
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

def resize_long_edge_cv2(image, target_size=384):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    else:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def display_images_and_text(source_image_path, generated_image, generated_paragraph, outfile_name):
    source_image = Image.open(source_image_path)
    # Create a new image that can fit the images and the text
    width = source_image.width + generated_image.width
    height = max(source_image.height, generated_image.height)
    new_image = Image.new("RGB", (width, height + 150), "white")

    # Paste the source image and the generated image onto the new image
    new_image.paste(source_image, (0, 0))
    new_image.paste(generated_image, (source_image.width, 0))

    # Write the generated paragraph onto the new image
    draw = ImageDraw.Draw(new_image)
    # font_size = 12
    # font = ImageFont.load_default().font_variant(size=font_size)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=14)

    # Wrap the text for better display
    wrapped_text = textwrap.wrap(generated_paragraph, width=170)
    # Draw each line of wrapped text
    line_spacing = 18
    y_offset = 0
    for line in wrapped_text:
        draw.text((0, height + y_offset), line, font=font, fill="black")
        y_offset += line_spacing

    # Show the final image
    # new_image.show()
    new_image.save(outfile_name)
    return 1


def extract_nouns_nltk(paragraph):
    words = word_tokenize(paragraph)
    pos_tags = pos_tag(words)
    nouns = [word for word, tag in pos_tags if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
    return nouns
