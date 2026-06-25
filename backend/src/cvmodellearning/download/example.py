from cvmodellearning.download.visionkg_utils import get_multi_class_stats

classes = ["car","pedestrian","bicycle","truck","bus","motorcycle", "traffic light", "traffic sign"]
stats = get_multi_class_stats(classes)
print(stats)
