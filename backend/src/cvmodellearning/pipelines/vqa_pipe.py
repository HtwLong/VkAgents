
from typing import Any, Dict, List

from cvmodellearning.paths import data_dir, json_labels_path, planning_artifacts_dir
from cvmodellearning.download.example_visionkg import download_visionkg_images_flat
from cvmodellearning.pipelines.vqa_pipeline_utils import generate_annotations


class VQAPipeline:
    """
    Pipeline for visual question answering (VQA) tasks using Vision-Language Models (VLMs).
    This pipeline includes downloading images, preparing the data (annotating images + split into train/val/test sets),
    finetuning the selected model, validation, optimization, and evaluation.
    """
    def __init__(self):
        # Initialize any necessary state or configurations here
        pass

    def download_data_step(self, config: Dict[str, Any], job_id: str):
        """Downloads data and create a consolidated COCO-style JSON label file."""

        selected_data: List[Dict[str, Any]] = config["selected_data"]

        data_base = data_dir(job_id)
        data_base.mkdir(parents=True, exist_ok=True)
                
        download_visionkg_images_flat(job_id, selected_data)

        return    


    def prepare_data_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        First annotate images with question-answer pairs using the generate_annotations method, 
        then parse them into ShareGPT format and split into train/val/test sets.
        """
        import json
        import re
        from sklearn.model_selection import train_test_split
        from cvmodellearning.paths import (
            data_dir, json_labels_path, train_json_path, val_json_path, test_json_path
        )
        
        image_folder_path = data_dir(job_id)
        output_file_path = json_labels_path(job_id)
        
        # --- NEW: Extract parameters dynamically from Planning Artifacts ---
        planning_dir = planning_artifacts_dir(job_id)
        
        # 1. Load Interpretation Data (for questions_list and use_case_description)
        interp_path = planning_dir / "RESULT_INTERPRETATION.json"
        interp_data = {}
        if interp_path.exists():
            with open(interp_path, "r", encoding="utf-8") as f:
                interp_data = json.load(f)
                
        questions_list = interp_data.get("questions_list") or []
        use_case_description = interp_data.get("use_case_description", "General visual question answering")
        
        # 2. Load Preprocessing Data (for num_qa_pairs)
        prep_path = planning_dir / "RESULT_PREPROCESSING.json"
        prep_data = {}
        if prep_path.exists():
            with open(prep_path, "r", encoding="utf-8") as f:
                prep_data = json.load(f)
                
        num_qa_pairs = prep_data.get("num_qa_pairs", 5) # Fallback to 5 if agent omits it
        
        # 1. Generate Annotations (Casting Paths to strings to ensure OS compatibility)
        generate_annotations(
            str(image_folder_path), 
            str(output_file_path), 
            questions_list, 
            use_case_description, 
            num_qa_pairs
        )
        
        if not output_file_path.exists():
            raise FileNotFoundError(f"Expected annotation file not found at {output_file_path}")
            
        # 2. Parse raw text into multi-turn ShareGPT format
        with open(output_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        sharegpt_data = []
        # Regex to capture everything between "Qx:" and "Ax:" and the next "Qx:"
        pattern = r"Q\d+:\s*(.*?)\s*A\d+:\s*(.*?)(?=\nQ\d+:|$)"
        
        for item in raw_data:
            matches = re.findall(pattern, item["raw_annotation"], re.DOTALL)
            
            # Skip this image if the teacher model hallucinated a completely broken format
            if not matches:
                continue 
                
            messages = [{"role": "system", "content": "You are a helpful visual assistant."}]
            for idx, (q, a) in enumerate(matches):
                q = q.strip()
                a = a.strip()
                
                # The first user message MUST contain the <image> tag
                if idx == 0:
                    messages.append({"role": "user", "content": f"<image>{q}"})
                else:
                    messages.append({"role": "user", "content": q})
                    
                messages.append({"role": "assistant", "content": a})
                
            sharegpt_data.append({
                "messages": messages,
                "images": [item["image_path"]]
            })
            
        # 3. Split into Train/Val/Test
        train_ratio = config.get("train_ratio", 0.8)
        val_ratio = config.get("val_ratio", 0.1)
        test_ratio = config.get("test_ratio", 0.1)
        
        # Normalize ratios to ensure they sum perfectly to 1.0
        total = train_ratio + val_ratio + test_ratio
        train_ratio, val_ratio, test_ratio = train_ratio/total, val_ratio/total, test_ratio/total
        
        # First split: Train vs Temp (Val + Test)
        train_data, temp_data = train_test_split(
            sharegpt_data, 
            test_size=(val_ratio + test_ratio), 
            random_state=42
        )
        
        # Second split: Val vs Test
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=(test_ratio / (val_ratio + test_ratio)), 
            random_state=42
        )
        
        # 4. Save the formatted splits using your paths.py definitions
        for data_split, path_func in zip(
            [train_data, val_data, test_data], 
            [train_json_path, val_json_path, test_json_path]
        ):
            with open(path_func(job_id), "w", encoding="utf-8") as f:
                json.dump(data_split, f, indent=4, ensure_ascii=False)
                
        # Return metrics to your orchestrator
        return {
            "status": "success",
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data)
        }


    async def train_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Finetune using ms-swift Python API to allow custom validation metrics and early stopping."""
        import os
        import asyncio
        import numpy as np
        from cvmodellearning.paths import train_json_path, val_json_path, artifacts_dir
        
        # Core ms-swift and HF imports
        from swift.llm import sft_main, SftArguments
        from transformers import EarlyStoppingCallback
        
        # --- UPDATED IMPORT FOR MS-SWIFT v3.x/v4.x ---
        from swift.plugin.metric import METRIC_MAPPING
        
        # --- 1. Environment & Model Type Mapping ---
        os.environ["USE_HF"] = "1"
        
        config_model_name = config.get("model_name", "Qwen3-VL-2B-Instruct")
        swift_model_type = config_model_name.lower()
        
        output_dir = artifacts_dir(job_id) / "swift_checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        track_metric = config.get("track_metric", "val_loss")
        is_generative_metric = track_metric != "val_loss"
        
        # --- 2. Custom Metric Injection (Refined for ROUGE, CIDEr, F1) ---
        if is_generative_metric:
            import evaluate
            from transformers import AutoTokenizer
            
            print(f"Loading custom metric: {track_metric}")
            
            # Note: Standard HF 'f1' crashes on strings. Use Exact Match or ROUGE instead for generative tasks.
            metric_module = evaluate.load(track_metric if track_metric != "f1" else "exact_match")
            
            hf_repo_path = f"Qwen/{config_model_name}" if not config_model_name.startswith("Qwen/") else config_model_name
            tokenizer = AutoTokenizer.from_pretrained(hf_repo_path, trust_remote_code=True)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
            def compute_custom_metrics(eval_preds):
                preds, labels = eval_preds
                labels = np.where(labels != -100, labels, pad_id)
                
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # CIDEr explicitly requires a list of lists for references
                if track_metric == "cider":
                    refs = [[l] for l in decoded_labels]
                    result = metric_module.compute(predictions=decoded_preds, references=refs)
                else:
                    result = metric_module.compute(predictions=decoded_preds, references=decoded_labels)
                
                # Safely extract the exact sub-score based on the specific metric
                if track_metric == "rouge":
                    score = result.get("rougeL", 0.0)
                elif track_metric in result:
                    score = result[track_metric]
                elif track_metric == "exact_match" and "exact_match" not in result:
                    score = result.get("exact_match", 0.0)
                else:
                    score = list(result.values())[0] if result else 0.0
                    
                return {track_metric: score}

            METRIC_MAPPING[track_metric] = (compute_custom_metrics, None)

        # --- 3. Hardware & Optimizer Mapping ---
        precision = config.get("precision", "bf16")
        bf16 = precision == "bf16"
        fp16 = precision == "fp16"
            
        opt_name = config.get("optimizer_name", "adamw")
        optimizer_kwargs = {}
        if opt_name in ["paged_adamw_8bit", "adamw"]:
            optim_arg = "paged_adamw_8bit" if opt_name == "paged_adamw_8bit" else "adamw_torch"
            optimizer_kwargs = {
                "adam_beta1": config.get("beta1", 0.9), 
                "adam_beta2": config.get("beta2", 0.999), 
                "adam_epsilon": config.get("eps", 1e-8)
            }
        else:
            optim_arg = opt_name

        # --- 4. Programmatic Configuration ---
        sft_args = SftArguments(
            model_type=swift_model_type,
            dataset=[str(train_json_path(job_id))],
            val_dataset=[str(val_json_path(job_id))],
            output_dir=str(output_dir),
            max_length=config.get("max_seq_length", 2048),
            num_train_epochs=config.get("num_epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 2),
            per_device_eval_batch_size=config.get("batch_size", 2),
            learning_rate=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
            
            optim=optim_arg,
            **optimizer_kwargs,
            
            bf16=bf16,
            fp16=fp16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,  # Prevent disk overflow by only keeping the 2 best checkpoints
            gradient_accumulation_steps=4,
            
            sft_type="lora" if config.get("use_lora", True) else "full",
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            lora_target_modules=["ALL"],
            
            predict_with_generate=is_generative_metric,
            metric_for_best_model=track_metric if is_generative_metric else "loss",
            greater_is_better=is_generative_metric
        )
        
        # --- 5. Implement Early Stopping (Patience) ---
        patience = config.get("patience", 0)
        custom_callbacks = []
        if patience > 0:
            custom_callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        
        # --- 6. Execute Training Asynchronously ---
        print(f"Starting programmatic SWIFT fine-tuning for job {job_id}...")
        
        def run_swift():
            if custom_callbacks:
                return sft_main(sft_args, callbacks=custom_callbacks) 
            return sft_main(sft_args)
            
        await asyncio.to_thread(run_swift)
        
        print(f"Training completed successfully! Artifacts saved to {output_dir}")
        return {
            "status": "success",
            "model_dir": str(output_dir),
            "track_metric_used": track_metric
        }


    def evaluate_model_step(self, config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Evaluate the finetuned model on the test set and return performance metrics."""
        pass