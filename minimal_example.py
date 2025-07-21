from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()    

user_prompt = "I want a model that can look at a photo and tell me if there is a cat in it or not."

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": "Extract the user prompt and return a Json which fits the given schema"},
        {"role": "user", "content": user_prompt}
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "math_response",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "output": {"type": "string"}
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": False
                        }
                    },
                    "final_answer": {"type": "string"}
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

print(response.output_text)

print(response.output_text)

#if __name__ == "__main__":
#    main()
