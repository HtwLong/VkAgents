from agents import Agent, Runner, trace
from model_requirements import StructuredOutputModel
import asyncio
import json
import shutil

# agents for the workflow
model_selector_agent = Agent(
    name="Model Selector",
    instructions=(
        "Given the available information about task, data, model and evaluation metrics, "
        "fill in only the fields model_architecture, model_family, description of the model property."
        "If evaluation metrics is empty, find the best fititng ones to the specific "
        "model architecture."
    ),
    output_type=StructuredOutputModel,
    model="gpt-4o-mini"
)

data_preprocessing_agent = Agent(
    name="Data Preprocessor",
    instructions=(
        "Determine the necessary data preprocessing methods we "
        "need to use and if there is not enough data (images and labels) for the given model, find "
        "fitting augmentation strategies based on the information on data, model and "
        "task. Fill in how many images per class and how many images in total we need at least to get good results in the evaluation metrics specified."
    ),
    output_type=StructuredOutputModel,
    model="gpt-4o-mini"
)

hyperparameter_suggest_agent = Agent(
    name="Hyperparameters Suggester",
    instructions="Think about combinations of hyperparameter for the " \
        "given model, task and data information.",
    model="gpt-4o-mini"
)

hyperparameter_opt_agent = Agent(
    name="Hyperparameter Optimizer",
    instructions=" Doublecheck the hyperparameters by predicting whether " \
        "the combinations are likely to return good results. If a combination is " \
        "unlikely to give good results, discard that combination and rerun the " \
        "hyperparameter suggester to find another good combination of hyperparameters.",
    model="gpt-4o-mini"
)

workflow_agent = Agent(
    name="Workflow Evaluator",
    instructions="",
    handoffs=[model_selector_agent, data_preprocessing_agent, hyperparameter_opt_agent],
    model="gpt-4o-mini"
)

# utility functions
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def validate_with_pydantic(data):
    """Validate against Pydantic model. Raises if invalid."""
    StructuredOutputModel.model_validate(data)

def wrap_input_as_messages(input_dict):
    """
    Helper to wrap a dictionary input as a list of messages for OpenAI GPT agents.
    Converts dict to json string and sets role to 'user'.
    """
    return [
        {
            "role": "user",
            "content": json.dumps(input_dict, indent=2)
        }
    ]

# create deterministic workflow
async def main():
    base_file = "result_interpretation.json"
    selector_file = "result_model.json"
    preprocessing_file = "result_preprocessing.json"

    # Copy the base file for step 1
    shutil.copyfile(base_file, selector_file)

    with trace("Deterministic model & data flow with schema enforcement"):
        model_input = load_json(selector_file)
        validate_with_pydantic(model_input)  # Validate input

        model_selector_result = await Runner.run(
            model_selector_agent,
            wrap_input_as_messages(model_input)
        )

        model_selector_output = model_selector_result.final_output
        validate_with_pydantic(model_selector_output.dict())
        save_json(model_selector_output.dict(), selector_file)
        print(f"Step 1: Model selector output saved to {selector_file}")

        data_input = load_json(selector_file)
        validate_with_pydantic(data_input)

        data_preprocessing_result = await Runner.run(
            data_preprocessing_agent,
            wrap_input_as_messages(data_input)
        )

        data_preprocessing_output = data_preprocessing_result.final_output
        validate_with_pydantic(data_preprocessing_output.dict())
        save_json(data_preprocessing_output.dict(), preprocessing_file)
        print(f"Step 2: Data preprocessor output saved to {preprocessing_file}")

    print("Deterministic structured flow complete.")
    print(f"Final output written to: {preprocessing_file}")

if __name__ == "__main__":
    asyncio.run(main())
