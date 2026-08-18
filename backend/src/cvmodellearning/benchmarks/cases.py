from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    prompt: str
    task: str
    classes: frozenset[str]


CASES = (
    BenchmarkCase(
        "traffic-participant-detection",
        "I need a model for a traffic-monitoring system that detects the traffic participants in each image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using CPU or Metal acceleration. It should aim for a mAP@0.5:0.95 of around 0.30 or higher.",
        "detection",
        frozenset({"person", "bicycle", "car", "motorcycle", "bus", "truck"}),
    ),
    BenchmarkCase(
        "robust-car-detection",
        "I need a model for a traffic-monitoring system that detects cars in each image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using CPU or Metal acceleration. It should aim for a mAP@0.5:0.95 of around 0.30 or higher. Processing an image within roughly 500 milliseconds would be desirable, but reliable classification under different viewpoints, lighting conditions, weather conditions, and partial occlusion is more important than inference speed.",
        "detection",
        frozenset({"car"}),
    ),
    BenchmarkCase(
        "small-traffic-object-detection-m4",
        "I need an object detection model to identify traffic lights and traffic signs in dense urban street scenes. The objects may be small and far away in the image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using Metal acceleration where supported. It should aim for a mAP@0.5:0.95 of around 0.30 or higher. An inference time of approximately 500 milliseconds or less per frame would be desirable, although somewhat slower processing is acceptable when it improves detection quality in difficult conditions. Memory usage during inference should preferably remain below 6 GB.",
        "detection",
        frozenset({"traffic light", "traffic sign"}),
    ),
    BenchmarkCase(
        "small-traffic-object-detection-rtx2060",
        "I need an object detection model to identify traffic lights and traffic signs in dense urban street scenes. The objects may be small and far away in the image. The model will run inference locally on a server with an RTX 2060 GPU with 6 GB of memory. It should aim for a mAP@0.5:0.95 of around 0.35 or higher. Inference time is not important. Memory usage during inference should preferably remain below 6 GB.",
        "detection",
        frozenset({"traffic light", "traffic sign"}),
    ),
    BenchmarkCase(
        "people-and-pets-detection",
        "I need an object detection model to locate people, dogs, and cats in indoor and outdoor photographs. The objects may appear at different scales, under varied lighting, and may be partially occluded by furniture or other people. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using Metal acceleration where supported. It should aim for a mAP@0.5:0.95 of approximately 0.40 or higher and recall of at least 0.75 for each class. Processing an image within roughly 500 milliseconds would be desirable.",
        "detection",
        frozenset({"person", "dog", "cat"}),
    ),
    BenchmarkCase(
        "indoor-furniture-detection",
        "I need an object detection model to locate nightstands, coffee tables and desks in indoor photographs. It should handle cluttered rooms, partial occlusion, varied lighting, and objects viewed from different angles. Inference will run on CPU-only backend servers with approximately 8 CPU cores and 16 GB of RAM. The model should aim for a mAP@0.5:0.95 of approximately 0.30 or higher.",
        "detection",
        frozenset({"nightstand", "coffee table", "desk"}),
    ),
    BenchmarkCase(
        "furniture-image-classification",
        "I need an image classification model for a furniture marketplace that categorizes the primary product in an uploaded photo as a chair, sofa, table, cabinet and or lamp. Each image should primarily contain one product. Inference will run on CPU-only backend servers with approximately 4 CPU cores and 8 GB of RAM. The model should aim for a macro-F1 score of at least 0.85, use less than approximately 1.5 GB of runtime memory, and preferably classify an image within 500 milliseconds.",
        "classification",
        frozenset({"chair", "sofa", "table", "cabinet", "lamp"}),
    ),
    BenchmarkCase(
        "handwritten-digit-classification",
        "I need a lightweight image classification model that recognizes handwritten numbers. The model will run on a CPU-only system with 4 GB of RAM. It should aim for accuracy of at least 95%, use less than approximately 500 MB of runtime memory, and process an image within 200 milliseconds.",
        "classification",
        frozenset({str(value) for value in range(10)}),
    ),
    BenchmarkCase(
        "dinov2-lora-furniture-classification",
        "I need an image classification model for a furniture marketplace that categorizes the primary product in an uploaded photo as a chair, sofa, table, cabinet and or lamp. Please use the dinov2_vits14 and LoRA. The model will run on a CPU-only system with 8 GB of RAM.",
        "classification",
        frozenset({"chair", "sofa", "table", "cabinet", "lamp"}),
    )
)
