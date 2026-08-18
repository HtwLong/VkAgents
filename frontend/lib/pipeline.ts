export type TaskType = "automatic" | "classification" | "detection" | "vqa"

export const TASK_LABELS: Record<TaskType, string> = {
  automatic: "Automatic",
  classification: "Classification",
  detection: "Detection",
  vqa: "Visual Question Answering",
}

export type DomainType = "traffic" | "animals" | "retail" | "handwriting" | "interiors"

export const DOMAIN_LABELS: Record<DomainType, string> = {
  traffic: "Traffic",
  animals: "Animals & People",
  retail: "Retail",
  handwriting: "Handwriting",
  interiors: "Indoor Spaces",
}

export type StepStatus = "pending" | "running" | "done" | "failed"

export type RevisionTarget =
  | "task-interpretation"
  | "model-selection"
  | "dataset-selection"
  | "choose-hyperparameters"

export type RevisionScope = RevisionTarget | "automatic"
export type ChangeStrength = "required" | "preferred"

export interface RevisionChange {
  id: string
  target_step: RevisionTarget
  field: string
  operation: "set" | "include" | "exclude" | "prefer" | "avoid"
  value: unknown
  strength: ChangeStrength
  summary: string
}

export interface RevisionPlan {
  required_text: string
  preferred_text: string
  summary: string
  restart_from: RevisionTarget
  changes: RevisionChange[]
}

export interface RevisionVerification {
  satisfied: boolean
  checks: Array<{
    change_id: string
    field: string
    strength: ChangeStrength
    expected: unknown
    actual: unknown
    satisfied: boolean
    summary: string
  }>
}

export interface PostTrainingAssessment {
  assessment_id: string
  job_id: string
  created_at: string
  verdict: "satisfied" | "partially_satisfied" | "not_satisfied" | "unknown"
  summary: string
  requirements: Array<{
    requirement: string
    status: "satisfied" | "not_satisfied" | "unknown"
    evidence: string[]
    explanation: string
  }>
  recommended_plan: RevisionPlan | null
  limitations: string[]
}

export interface AssessmentEligibility {
  eligible: boolean
  reason?: string | null
  can_create_revision: boolean
  revision_reason?: string | null
}

export interface PlanningLLMUsageBucket {
  requests: number
  input_tokens: number
  cached_input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  total_tokens: number
  calculated_cost_usd: string | null
}

export interface PlanningLLMUsage {
  schema_version: number
  job_id: string
  scope: "planning"
  currency: "USD"
  totals: PlanningLLMUsageBucket
  models: Record<string, PlanningLLMUsageBucket>
  operations: Record<string, PlanningLLMUsageBucket>
  pricing: Record<string, {
    input_per_million: string
    cached_input_per_million: string
    output_per_million: string
    effective_date: string
    source: string
  }>
  usage_notes: string[]
}

export interface DeliverableArtifact {
  id: string
  kind: string
  label: string
  filename: string
  downloadUrl?: string
  description?: string
  standalone?: boolean
  requiredBaseModel?: string
  generatedOnDownload?: boolean
}

export interface EvaluationReport {
  schema_version?: number
  job_id: string
  task: "classification" | "detection"
  model: { name: string; weights?: string | null; training_mode?: string | null }
  classes: string[]
  metrics: Record<string, number>
  per_class: Array<{
    class_name: string
    precision?: number | null
    recall?: number | null
    f1?: number | null
    support?: number | null
    ap?: number | null
    ap50?: number | null
    ap75?: number | null
  }>
  confusion_matrix: number[][]
  confusion_matrix_labels?: string[]
  curves?: Record<string, { x: number[]; y: number[]; x_label?: string }>
  size_metrics?: Record<string, number>
  evaluation_slices?: Array<{ name: string; metrics: Record<string, number> }>
  dataset_statistics?: {
    images?: number
    instances?: number
    images_without_annotations?: number
    mean_box_area_pixels?: number
    per_class?: Array<{ class_name: string; images: number; instances: number }>
  }
  visualizations?: Array<{ name: string; path: string; url?: string }>
  operating_point?: Record<string, number>
  training_history: Array<Record<string, number>>
  dataset: {
    splits: Record<string, number>
    assignment_fingerprint?: string | null
  }
  configuration: Record<string, string | number | boolean>
}

export interface PipelineStep {
  id: string
  title: string
  /** Output lines rendered while the step runs. */
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
  name: string
  task: TaskType
  domain: DomainType
  text: string
}

export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    id: "traffic-participant-detection",
    name: "Traffic Participant Detection",
    task: "detection",
    domain: "traffic",
    text: "I need a model for a traffic-monitoring system that detects the traffic participants in each image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using CPU or Metal acceleration. It should aim for a mAP@0.5:0.95 of around 0.30 or higher.",
  },
  {
    id: "robust-car-detection",
    name: "Robust Car Detection",
    task: "detection",
    domain: "traffic",
    text: "I need a model for a traffic-monitoring system that detects cars in each image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using CPU or Metal acceleration. It should aim for a mAP@0.5:0.95 of around 0.30 or higher. Processing an image within roughly 500 milliseconds would be desirable, but reliable classification under different viewpoints, lighting conditions, weather conditions, and partial occlusion is more important than inference speed.",
  },
  {
    id: "small-traffic-object-detection-m4",
    name: "Small Traffic Object Detection (MacBook M4)",
    task: "detection",
    domain: "traffic",
    text: "I need an object detection model to identify traffic lights and traffic signs in dense urban street scenes. The objects may be small and far away in the image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using Metal acceleration where supported. It should aim for a mAP@0.5:0.95 of around 0.30 or higher. An inference time of approximately 500 milliseconds or less per frame would be desirable, although somewhat slower processing is acceptable when it improves detection quality in difficult conditions. Memory usage during inference should preferably remain below 6 GB.",
  },
  {
    id: "small-traffic-object-detection-rtx2060",
    name: "Small Traffic Object Detection (RTX 2060)",
    task: "detection",
    domain: "traffic",
    text: "I need an object detection model to identify traffic lights and traffic signs in dense urban street scenes. The objects may be small and far away in the image. The model will run inference locally on a server with an RTX 2060 GPU with 6 GB of memory. It should aim for a mAP@0.5:0.95 of around 0.35 or higher. Inference time is not important. Memory usage during inference should preferably remain below 6 GB.",
  },
  {
    id: "traffic-scene-vqa",
    name: "Traffic Scene Visual Question Answering",
    task: "vqa",
    domain: "traffic",
    text: "I need a compact visual question answering model for traffic images that analyzes real-world road scenarios, answers user questions, and recommends appropriate actions for traffic participants. It should be possible to fine-tune and run the model locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using lightweight fine-tuning and a compact or quantized model where necessary. The model should aim for an overall answer accuracy of around 70% or higher on a representative traffic VQA validation set. A typical response time below 5 seconds would be desirable, with memory usage preferably below 10 GB.",
  },
  {
    id: "people-and-pets-detection",
    name: "People and Pet Detection",
    task: "detection",
    domain: "animals",
    text: "I need an object detection model to locate people, dogs, and cats in indoor and outdoor photographs. The objects may appear at different scales, under varied lighting, and may be partially occluded by furniture or other people. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using Metal acceleration where supported. It should aim for a mAP@0.5:0.95 of approximately 0.40 or higher and recall of at least 0.75 for each class. Processing an image within roughly 500 milliseconds would be desirable.",
  },
  {
    id: "indoor-furniture-detection",
    name: "Indoor Furniture Detection",
    task: "detection",
    domain: "interiors",
    text: "I need an object detection model to locate nightstands, coffee tables and desks in indoor photographs. It should handle cluttered rooms, partial occlusion, varied lighting, and objects viewed from different angles. Inference will run on CPU-only backend servers with approximately 8 CPU cores and 16 GB of RAM. The model should aim for a mAP@0.5:0.95 of approximately 0.30 or higher.",
  },
  {
    id: "furniture-image-classification",
    name: "Furniture Image Classification",
    task: "classification",
    domain: "retail",
    text: "I need an image classification model for a furniture marketplace that categorizes the primary product in an uploaded photo as a chair, sofa, table, cabinet and or lamp. Each image should primarily contain one product. Inference will run on CPU-only backend servers with approximately 4 CPU cores and 8 GB of RAM. The model should aim for a macro-F1 score of at least 0.85, use less than approximately 1.5 GB of runtime memory, and preferably classify an image within 500 milliseconds."
  },
  {
    id: "handwritten-digit-classification",
    name: "Handwritten Digit Classification",
    task: "classification",
    domain: "handwriting",
    text: "I need a lightweight image classification model that recognizes handwritten numbers. The model will run on a CPU-only system with 4 GB of RAM. It should aim for accuracy of at least 95%, use less than approximately 500 MB of runtime memory, and process an image within 200 milliseconds.",
  },
  {
    id: "dinov2-lora-furniture-classification",
    name: "DINOv2 LoRA Furniture Classification",
    task: "classification",
    domain: "retail",
    text: "I need an image classification model for a furniture marketplace that categorizes the primary product in an uploaded photo as a chair, sofa, table, cabinet and or lamp. Please use the dinov2 vits14 and LoRA. The model will run on a CPU-only system with 8 GB of RAM.",
  },
]

export function buildPipeline(): PipelineStage[] {
  return [
    {
      id: "planning",
      title: "Planning Stage",
      description: "Interpret the request and design the learning strategy.",
      steps: [
        {
          id: "task-interpretation",
          title: "Task Interpretation",
          outputs: ["Waiting for /planning/task-interpret."],
        },
        {
          id: "check-data",
          title: "Check Data",
          outputs: ["Waiting for /planning/check-data."],
        },
        {
          id: "model-selection",
          title: "Model Selection",
          outputs: ["Waiting for /planning/select-model."],
        },
        {
          id: "dataset-selection",
          title: "Dataset Split Plan",
          outputs: ["Waiting for /planning/select-datasets."],
        },
        {
          id: "choose-hyperparameters",
          title: "Choose Hyperparameters",
          outputs: ["Waiting for /planning/choose-hyperparameters."],
        },
        {
          id: "ask-change-requests",
          title: "Ask for Change Requests",
          outputs: ["Review the proposed hyperparameters before execution."],
        },
      ],
    },
    {
      id: "execution",
      title: "Execution Stage",
      description: "Materialize the planned splits and train without test-data access.",
      steps: [
        {
          id: "download-data",
          title: "Download assigned data",
          outputs: ["Waiting for /download-data."],
        },
        {
          id: "prepare-data",
          title: "Materialize data splits",
          outputs: ["Waiting for /prepare-data."],
        },
        {
          id: "train-model",
          title: "Train model",
          outputs: ["Waiting for /train/start."],
        },
      ],
    },
    {
      id: "evaluation",
      title: "Evaluation Stage",
      description: "Evaluate once on the test split and produce auditable deliverables.",
      steps: [
        {
          id: "running-evaluation",
          title: "Running Evaluation",
          outputs: ["Waiting for /evaluate."],
        },
        {
          id: "preparing-trained-model",
          title: "Preparing Trained Model",
          outputs: ["Waiting for the typed artifact manifest."],
        },
        {
          id: "preparing-results",
          title: "Preparing Results",
          outputs: ["Waiting for /evaluate/{job_id}/report."],
        },
      ],
    },
  ]
}
