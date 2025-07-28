# simple user prompts
simple_prompt = """Please build me a model that classifies whether a fruit in a photo 
is fresh or rotten. I have a thousand of labeled images for both classes."""

simple_prompt2 = """Please build a wildlife monitoring model to detect and localize 
each animal in camera trap images from African savanna. I need the model to work 
for elephants, lions, and zebras"""

simple_prompt3 = """Can you create a model to count the number of cars, bicycles, 
and trucks in city street photographs? I don't have data yet but need the model 
for urban traffic monitoring."""

simple_prompt4 = """I need to segment MR scans to separate tumors from healthy tissue. 
I have a set of labeled mask images for training."""

simple_prompt5 = """Segment the furniture in photos so each chair, sofa, and table is 
outlined separately."""

advanced_prompt = """Build a lightweight MobileNetV3 model to classify whether a fruit 
in a photo is fresh or rotten, suitable for running in real-time on a smartphone. 
My training data consists of around 1,000 labeled images per class. I’d like the model 
to achieve at least 90% F1 score on my test set."""

advanced_prompt2 = """Develop a wildlife detection model using Faster R-CNN that can 
identify and draw bounding boxes around elephants, lions, and zebras in African savanna 
camera trap images. I would like Grad-CAM visualizations for model interpretability and a 
minimum mean Average Precision (mAP) of 0.80 on validation data."""

advanced_prompt3 = """I need a YOLOv5-based object detection model that counts and 
localizes cars, bicycles, and trucks in city street photos. The model should be able 
to process images efficiently for practical use. Since I don’t have my own dataset, 
please recommend a suitable open-source dataset for this task."""

advanced_prompt4 = """Train a U-Net model to segment tumors from healthy tissue in MRI 
scans. The model should be able to run on a hospital workstation with limited GPU resources 
(8GB VRAM max). My data includes paired MR images and binary masks. Aim for at least 
85% Dice coefficient on the test set."""

advanced_prompt5 = """Implement a DeepLabV3+ model to segment and label each piece 
of furniture (chair, sofa, table) in photos of living rooms. The model should support 
efficient batch inference for moderate image processing workloads. I have annotated 
segmentation masks for training."""
