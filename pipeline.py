import cv2
import torch
from detect import HandwritingDetector
from segment import segment_lines
from ocr import HandwritingOCR
from style_encoder import StyleEncoder, transform
from generate import HandwritingGenerator

def run_pipeline(image_path, new_text):
    image = cv2.imread(image_path)

    detector = HandwritingDetector()
    ocr = HandwritingOCR()
    style_encoder = StyleEncoder()
    generator = HandwritingGenerator("files/iam_model.pth")

    # 1. Detect handwriting regions
    boxes = detector.detect(image)
    regions = detector.crop(image, boxes)

    all_lines = []
    all_text = []

    # 2. Segment + OCR
    for region in regions:
        lines = segment_lines(region)
        for line in lines:
            text = ocr.read(line)
            if len(text.strip()) > 0:
                all_lines.append(line)
                all_text.append(text)

    # 3. Style embedding
    tensors = torch.stack([transform(l) for l in all_lines])
    style_vector = style_encoder(tensors).mean(dim=0)

    # 4. Generate handwriting
    print("Recognized text:")
    print("\n".join(all_text))

    print("\nGenerating new handwriting...")
    # NOTE: actual conditioning requires modifying TRGAN
    return style_vector

if __name__ == "__main__":
    run_pipeline("sample_page.jpg",
                 "This text is generated in the same handwriting")
