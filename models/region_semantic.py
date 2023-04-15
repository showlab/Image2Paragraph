from models.segment_models.semgent_anything_model import SegmentAnything
from models.segment_models.semantic_segment_anything_model import SemanticSegment


class RegionSemantic():
    def __init__(self, device):
        self.device = device
        self.init_models()

    def init_models(self):
        self.segment_model = SegmentAnything(self.device)
        self.semantic_segment_model = SemanticSegment(self.device)

    def semantic_prompt_gen(self, anns):
        """
        fliter too small objects and objects with low stability score
        anns: [{'class_name': 'person', 'bbox': [0.0, 0.0, 0.0, 0.0], 'size': [0, 0], 'stability_score': 0.0}, ...]
        semantic_prompt: "person: [0.0, 0.0, 0.0, 0.0]; ..."
        """
        # Sort annotations by area in descending order
        sorted_annotations = sorted(anns, key=lambda x: x['area'], reverse=True)
        # Select the top 10 largest regions
        top_10_largest_regions = sorted_annotations[:10]
        semantic_prompt = ""
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("\nStep3, Semantic Prompt:")
        for region in top_10_largest_regions:
            semantic_prompt += region['class_name'] + ': ' + str(region['bbox']) + "; "
        print(semantic_prompt)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return semantic_prompt

    def region_semantic(self, img_src):
        anns = self.segment_model.generate_mask(img_src)
        anns_w_class = self.semantic_segment_model.semantic_class_w_mask(img_src, anns)
        return self.semantic_prompt_gen(anns_w_class)
    
    def region_semantic_debug(self, img_src):
        return "region_semantic_debug"