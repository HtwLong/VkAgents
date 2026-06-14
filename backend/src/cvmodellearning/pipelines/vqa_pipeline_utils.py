from vllm import LLM, SamplingParams
import os, json, random

def generate_annotations(image_folder, output_file, questions_list, use_case_description, num_qa_pairs=5):
    # --- 1. Initialize the vLLM Engine ---
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ" 
    # For loading the large vLLM model, we use a single GPU (RTX 6000 (48GB)
    # Adaptive Weight Quantization (AWQ) allows us to fit the model in memory while maintaining performance 
    print(f"Loading {model_name} on a single GPU (RTX 6000 (48GB)...") 
    
    llm = LLM(
        model=model_name,
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True
    )

    sampling_params = SamplingParams(temperature=0.2, max_tokens=512, top_p=0.9)
    system_prompt = "You are an expert visual assistant building a dataset."

    # --- 2. Process Images ---
    dataset = []
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images. Starting batch generation...")

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        
        # --- DYNAMIC RANDOM SAMPLING PER IMAGE ---
        current_image_questions = questions_list.copy()
        
        # Check against our new parameter instead of a hardcoded 5
        if len(current_image_questions) > num_qa_pairs:
            current_image_questions = random.sample(current_image_questions, num_qa_pairs)
            
        num_provided = len(current_image_questions)
        num_additional = num_qa_pairs - num_provided

        # Build the contextual instructions based on the list length
        if num_provided > 0 and num_additional > 0:
            instruction_text = (
                f"Analyze this image carefully. Your overall goal/use case is: '{use_case_description}'.\n\n"
                f"First, answer these {num_provided} specific questions:\n"
            )
            for q in current_image_questions:
                instruction_text += f"- {q}\n"
            
            instruction_text += (
                f"\nThen, generate exactly {num_additional} MORE high-quality question and answer pairs "
                f"that are highly relevant to the use case described above. Focus on entirely different details."
            )
            
        elif num_provided >= num_qa_pairs:
            instruction_text = (
                f"Analyze this image carefully. Your overall goal/use case is: '{use_case_description}'.\n\n"
                f"Please answer these {num_qa_pairs} specific questions:\n"
            )
            for q in current_image_questions:
                instruction_text += f"- {q}\n"
                
        else:
            instruction_text = (
                f"Analyze this image carefully. Your overall goal/use case is: '{use_case_description}'.\n\n"
                f"Generate exactly {num_qa_pairs} high-quality question and answer pairs that are highly relevant to this use case."
            )

        # Build the rigid formatting template (The "Hook") dynamically
        format_text = "\n\nFormat your response strictly as:\n"
        for i in range(1, num_qa_pairs + 1):
            if i <= num_provided:
                format_text += f"Q{i}: {current_image_questions[i-1]}\nA{i}:"
            else:
                format_text += f"Q{i}:\nA{i}:"
                
            if i < num_qa_pairs:
                format_text += "\n"

        user_prompt = instruction_text + format_text

        # --- Generate Response ---
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(img_path)}"},
                    {"type": "text", "text": user_prompt}
                ]}
            ]

            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            
            dataset.append({
                "image_path": img_path,
                "raw_annotation": generated_text
            })
            print(f"Annotated: {img_name}")
            
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

    # --- 3. Export the Dataset ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nDone! Successfully annotated {len(dataset)} images. Saved to {output_file}")