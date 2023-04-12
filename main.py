import argparse
from models.image_text_transformation import ImageTextTransformation
from utils.util import display_images_and_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_src', default='examples/1.jpg')
    parser.add_argument('--out_image_name', default='output/1_result.jpg')
    args = parser.parse_args()

    processor = ImageTextTransformation()
    generated_text = processor.image_to_text(args.image_src)
    generated_image = processor.text_to_image(generated_text)
    ## then text to image
    print("*" * 50)
    print("Generated Text:")
    print(generated_text)
    print("*" * 50)

    results = display_images_and_text(args.image_src, generated_image, generated_text, args.out_image_name)
