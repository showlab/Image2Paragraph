import argparse
from models.image_text_transformation import ImageTextTransformation
from utils.util import display_images_and_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_src', default='examples/1.jpg')
    parser.add_argument('--out_image_name', default='output/1_result.jpg')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=True, help='Set this flag to True if you want to use semantic segmentation')
    parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: semantic segment anything or edit anything')
    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
    parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, <6G GPU is not recommended>')

    args = parser.parse_args()

    processor = ImageTextTransformation(args)
    generated_text = processor.image_to_text(args.image_src)
    generated_image = processor.text_to_image(generated_text)
    ## then text to image
    print("*" * 50)
    print("Generated Text:")
    print(generated_text)
    print("*" * 50)

    results = display_images_and_text(args.image_src, generated_image, generated_text, args.out_image_name)
