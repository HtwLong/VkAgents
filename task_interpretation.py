from openai import OpenAI
import user_prompts
from model_requirements import StructuredOutputModel

client = OpenAI()

# Prepare messages to send
messages = [
    {
        "role": "system",
        "content": (
            "Extract information from the user prompt about a computer vision model. "
            "If there are no preprocessing or augmentation strategies specified, leave the respective fields empty. "
            "If there are no specific model architecture or model family mentioned leave them empty."
            "If there are percentages, turn them into decimals."
        )
    },
    {
        "role": "user",
        "content": user_prompts.simple_prompt2
    }
]

# Call the new parse method with Pydantic model directly as response_format
completion = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=messages,
    response_format=StructuredOutputModel,
)

# Access the structured parsed output as a Pydantic model instance
structured_output = completion.choices[0].message.parsed

# Serialize the Pydantic model instance to JSON string for saving to file
with open('result_interpretation.json', 'w') as fout:
    fout.write(structured_output.model_dump_json(indent=2))

print("Structured interpretation saved to result_interpretation.json.")
