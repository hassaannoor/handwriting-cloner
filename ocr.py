import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class HandwritingOCR:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

    def read(self, image):
        pil = Image.fromarray(image)
        pixel_values = self.processor(pil, return_tensors="pt").pixel_values
        ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text
