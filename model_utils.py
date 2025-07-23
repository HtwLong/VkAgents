from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def get_ssdlite_mobilenet_v3(num_classes: int):
    """
    Initializes an SSDLite320 MobileNetV3 Large model for object detection, with a custom number of classes and randomly initialized weights.

    Args:
        num_classes (int): Number of output classes (including background).

    Returns:
        model (nn.Module): SSDLite object detection model ready for training.
    """
    model = ssdlite320_mobilenet_v3_large(
        weights=None,          
        num_classes=num_classes
    )
    return model
