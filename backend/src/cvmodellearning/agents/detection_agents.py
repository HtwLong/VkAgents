from agents import Agent
from cvmodellearning.llm_config import PLANNING_MODEL

from cvmodellearning.models.detection_models.torchvision_trainer import train_torchvision_model
torchvision_trainer_agent = Agent(
    model=PLANNING_MODEL,
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
