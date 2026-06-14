from pathlib import Path
import random
from typing import Dict, List, Union
from PIL import Image, ImageDraw, ImageFont


def _draw_detections(image: Image.Image, detections: List[List[Union[float, int]]], categories: Dict[int, str], output_path: Path) -> str:
    """Draws bounding boxes and labels onto the image and saves it."""
    draw = ImageDraw.Draw(image)
    
    try:
        # Load a simple font, default if unavailable
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        # Expected format: [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2, conf, cls_id = det[:6]
        
        cls_id = int(cls_id)
        class_name = categories.get(cls_id, f"Class {cls_id}")
        
        # Use a deterministic color based on the class ID
        random.seed(cls_id * 10) 
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        
        # Calculate text bounding box to draw the background block
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([x1, y1 - text_height - 5, text_bbox[2] + 5, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

    image.save(output_path)
    return str(output_path)