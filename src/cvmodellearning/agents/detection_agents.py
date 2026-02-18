from agents import Agent

from cvmodellearning.models.detection_models.torchvision_trainer import train_torchvision_model
from cvmodellearning.models.detection_models.yolo_trainer import train_yolo_model



yolo_trainer_agent = Agent(
    model="gpt-4o-mini",
    tools=[train_yolo_model],
    instructions=(
        "You are an expert in configuring and launching Ultralytics YOLO training runs. "
        "Your sole purpose is to call the `train_yolo_model` function with **all required parameters** "
        "derived from the user's request. **You MUST output the final function call and nothing else.** "
        "Always ensure to provide values for: **job_id**, **model_version**, **model_size**, **epochs**, **batch**, **imgsz**, "
        "**optimizer**, **lr0**, **momentum**, **weight_decay**, **patience**, **lrf**, **warmup_epochs**, **warmup_momentum**, "
        "**box**, **cls**, **dfl**, **mosaic**, **mixup**, **fliplr**, **scale**, **degrees**, **hsv_h**, **freeze**, and **close_mosaic**. "
        "IF the user specifies any other advanced optimizer parameters, encode them correctly into the `optimizer_override_json` string as a valid JSON object."
    ),
    name="YOLO Trainer Agent"
)

torchvision_trainer_agent = Agent(
    model="gpt-4o-mini",
    tools=[train_torchvision_model], 
    instructions=(
        "You are an expert in configuring and launching standard PyTorch/TorchVision "
        "object detection training runs. Your sole purpose is to call the "
        "`train_torchvision_model` function with **all required parameters** "
        "derived from the user's request. **You MUST output the final function call and nothing else.** "
        "Always ensure to provide: **model_name**, **num_classes**, **batch_size**, **learning_rate**, "
        "**epochs**, **monitor_metric**, **patience**, **save_best_model**, and **job_id**."
        "IF the user specifies any other advanced configuration, encode it correctly into the `config_override_json` string as a valid JSON object."
    ),
    name="TorchVision Trainer Agent"
)
