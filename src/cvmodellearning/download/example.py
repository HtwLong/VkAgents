from cvmodellearning.download.visionkg_utils import get_multi_class_stats

classes = ["car","pedestrian","bicycle","truck","bus","motorcycle"]
stats = get_multi_class_stats(classes)
print(stats)
