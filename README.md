# Handwriting Cloner Pro

A modular, production-grade Python system for end-to-end handwriting cloning. This system takes raw page images, detects handwriting, segments it, performs OCR, extracts a style embedding, and generates new handwriting in that style.

## System Overview

1.  **Detection**: Identifies handwriting regions in an image (ignoring drawings/margins) using YOLOv8.
2.  **Segmentation**: Breaks down regions into individual text lines using computer vision techniques.
3.  **OCR**: Recognizes the text content using Microsoft's TrOCR (Transformer-based OCR).
4.  **Style Extraction**: Encodes the visual style of the handwriting into a vector using a ResNet18 encoder.
5.  **Generation**: Synthesizes new text in the target style using a TRGAN-based generator.

## Directory Structure

```
handwriting_cloner/
│
├── detect.py          # Handwriting vs drawings detection (YOLOv8)
├── segment.py         # Line segmentation (OpenCV)
├── ocr.py             # Handwriting OCR (TrOCR)
├── style_encoder.py   # Style feature extraction (ResNet18)
├── generate.py        # Handwriting generation (TRGAN)
├── pipeline.py        # Full end-to-end pipeline orchestration
└── requirements.txt   # Project dependencies
```

## Installation

1.  **Prerequisites**: Python 3.8+ recommended.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point is `pipeline.py`.

```python
python pipeline.py
```

### Example

In `pipeline.py`:

```python
if __name__ == "__main__":
    # Runs the full pipeline on a sample image
    style_vector = run_pipeline("sample_page.jpg", "This text is generated in the same handwriting")
```

## Configuration & Models

**Note**: This is a production-grade architecture skeleton. To be fully operational, it requires trained model weights:
- **Detection**: Fine-tuned YOLOv8 model for document layout analysis.
- **Generation**: Pre-trained TRGAN generator weights (e.g., trained on IAM dataset).

The code is structure to load these weights easily in the respective class `__init__` methods.

## Technical Details

- **Frameworks**: PyTorch, Transformers (Hugging Face), Ultralytics (YOLO), OpenCV.
- **Architecture**: Modular design allowing individual components (e.g., OCR or Detector) to be swapped or upgraded independently.
