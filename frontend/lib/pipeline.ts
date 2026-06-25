export type TaskType = "automatic" | "classification" | "detection" | "vqa"

export const TASK_LABELS: Record<TaskType, string> = {
  automatic: "Automatic",
  classification: "Classification",
  detection: "Detection",
  vqa: "Visual Question Answering",
}

export type StepStatus = "pending" | "running" | "done"

export interface PipelineStep {
  id: string
  title: string
  /** Output lines streamed while the step runs. */
  outputs: string[]
}

export interface PipelineStage {
  id: string
  title: string
  description: string
  steps: PipelineStep[]
}

export interface ExamplePrompt {
  id: string
  task: TaskType
  text: string
}

export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    id: "ex-1",
    task: "classification",
    text: "Train a model that classifies product photos into 12 retail categories with at least 92% top-1 accuracy.",
  },
  {
    id: "ex-2",
    task: "detection",
    text: "Build a detector that finds and localizes hard hats and safety vests in construction site images.",
  },
  {
    id: "ex-3",
    task: "vqa",
    text: "Create a visual question answering model that can answer free-form questions about medical X-ray scans.",
  },
  {
    id: "ex-4",
    task: "classification",
    text: "Distinguish ripe from unripe strawberries using a lightweight model that runs on an edge device under 10ms.",
  },
  {
    id: "ex-5",
    task: "detection",
    text: "Detect potholes and road cracks from dashcam footage and output bounding boxes per frame.",
  },
  {
    id: "ex-6",
    task: "automatic",
    text: "Here are 4,000 satellite tiles. Figure out the best vision task and train a model to flag deforestation.",
  },
  {
    id: "ex-7",
    task: "vqa",
    text: "Given a photo of a fridge, answer questions like 'how many eggs are left?' and 'is there milk?'.",
  },
  {
    id: "ex-8",
    task: "classification",
    text: "Grade the severity of plant leaf disease into 5 levels from smartphone photos of crops.",
  },
]

/**
 * Builds the three-stage pipeline. Content adapts slightly to the chosen task
 * so the simulated run feels coherent with the user's intent.
 */
export function buildPipeline(task: TaskType): PipelineStage[] {
  const resolvedTask: Exclude<TaskType, "automatic"> =
    task === "automatic" ? "classification" : task

  const taskName = TASK_LABELS[resolvedTask]

  return [
    {
      id: "planning",
      title: "Planning Stage",
      description: "Interpret the request and design the learning strategy.",
      steps: [
        {
          id: "plan-parse",
          title: "Parse task specification",
          outputs: [
            "Tokenizing natural language prompt...",
            task === "automatic"
              ? `Task not specified — inferring from intent and data signals.`
              : `User-selected task: ${taskName}.`,
            task === "automatic"
              ? `Inferred task → ${taskName} (confidence 0.91).`
              : `Locked objective → ${taskName}.`,
            "Extracted constraints: latency budget, target metric, deployment target.",
          ],
        },
        {
          id: "plan-data",
          title: "Resolve dataset candidates",
          outputs: [
            "Scanning attached corpus + public dataset registry...",
            "Matched 3 candidate datasets by semantic similarity.",
            "Selected primary dataset: 8,742 images / 12 classes.",
            "Proposed split → train 70% · val 15% · test 15%.",
          ],
        },
        {
          id: "plan-arch",
          title: "Select architecture",
          outputs: [
            "Querying model zoo for compatible backbones...",
            resolvedTask === "detection"
              ? "Recommended backbone: RT-DETR-R50 (real-time detection)."
              : resolvedTask === "vqa"
                ? "Recommended backbone: ViT-L/14 + cross-attention text head."
                : "Recommended backbone: ConvNeXt-Tiny (efficient classifier).",
            "Estimated params: 28.4M · est. VRAM: 6.1GB.",
          ],
        },
        {
          id: "plan-hparams",
          title: "Draft training plan",
          outputs: [
            "Optimizer: AdamW · lr 3e-4 · cosine schedule.",
            "Batch size: 64 · epochs: 30 · early-stop patience: 5.",
            "Augmentations: random-resize-crop, flip, color-jitter, mixup.",
            "Plan approved by policy model. Handing off to execution.",
          ],
        },
      ],
    },
    {
      id: "execution",
      title: "Execution Stage",
      description: "Prepare data and train the candidate model.",
      steps: [
        {
          id: "exec-preprocess",
          title: "Preprocess data",
          outputs: [
            "Decoding & validating 8,742 images...",
            "Removed 31 corrupt / duplicate samples.",
            "Normalizing to ImageNet mean/std · resize 224×224.",
            "Built augmentation pipeline · caching to disk.",
          ],
        },
        {
          id: "exec-init",
          title: "Initialize model",
          outputs: [
            "Loading pretrained weights...",
            "Replaced classification head for target classes.",
            "Froze backbone for 2 warmup epochs.",
          ],
        },
        {
          id: "exec-train",
          title: "Train model",
          outputs: [
            "epoch 01/30 · loss 1.842 · val_acc 0.612",
            "epoch 08/30 · loss 0.731 · val_acc 0.847",
            "epoch 16/30 · loss 0.402 · val_acc 0.911",
            "epoch 24/30 · loss 0.241 · val_acc 0.938",
            "epoch 29/30 · loss 0.188 · val_acc 0.951 · early-stop triggered",
          ],
        },
        {
          id: "exec-checkpoint",
          title: "Checkpoint best model",
          outputs: [
            "Best epoch: 29 · val_acc 0.951.",
            "Exporting weights → model.safetensors (112MB).",
            "Compiling to ONNX for portable inference.",
          ],
        },
      ],
    },
    {
      id: "evaluation",
      title: "Evaluation Stage",
      description: "Measure quality and produce deliverables.",
      steps: [
        {
          id: "eval-metrics",
          title: "Compute test metrics",
          outputs: [
            "Running inference on held-out test set...",
            resolvedTask === "detection"
              ? "mAP@0.5: 0.883 · mAP@0.5:0.95: 0.641"
              : resolvedTask === "vqa"
                ? "Exact-match: 0.792 · soft-accuracy: 0.864"
                : "Top-1 accuracy: 0.947 · Top-5: 0.992 · F1: 0.944",
            "Latency: 8.3ms / image on target hardware.",
          ],
        },
        {
          id: "eval-analysis",
          title: "Error analysis",
          outputs: [
            "Generating confusion matrix & failure clusters...",
            "Largest confusion: class 4 ↔ class 9 (visually similar).",
            "Recommendation: add 200 hard-negative samples for class 4.",
          ],
        },
        {
          id: "eval-report",
          title: "Compile report & artifacts",
          outputs: [
            "Assembling metrics, plots, and model card...",
            "Rendering PDF summary...",
            "Packaging downloadable model bundle.",
            "Pipeline complete. Deliverables ready.",
          ],
        },
      ],
    },
  ]
}
