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
  }>
  confusion_matrix: number[][]
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
  task: TaskType
  domain: DomainType
  text: string
}

export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    id: "traffic-participants",
    task: "detection",
    domain: "traffic",
    text: "I need a model for a traffic-monitoring system that detects the traffic participants in each image. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using CPU or Metal acceleration. It should aim for a mAP@0.5:0.95 of around 0.30 or higher. Processing an image within roughly 500 milliseconds would be desirable, but reliable classification under different viewpoints, lighting conditions, weather conditions, and partial occlusion is more important than inference speed.",
  },
  {
    id: "traffic-lights-signs",
    task: "detection",
    domain: "traffic",
    text: "I need an object detection model to identify traffic lights and signs in dense urban street scenes under low-light and rainy conditions. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using Metal acceleration where supported. It should aim for a mAP@0.5:0.95 of around 0.30 or higher. An inference time of approximately 500 milliseconds or less per frame would be desirable, although somewhat slower processing is acceptable when it improves detection quality in difficult conditions. Memory usage during inference should preferably remain below 6 GB.",
  },
  {
    id: "ex-vqa",
    task: "vqa",
    domain: "traffic",
    text: "I need a compact visual question answering model for traffic images that analyzes real-world road scenarios, answers user questions, and recommends appropriate actions for traffic participants. It should be possible to fine-tune and run the model locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using lightweight fine-tuning and a compact or quantized model where necessary. The model should aim for an overall answer accuracy of around 70% or higher on a representative traffic VQA validation set. A typical response time below 5 seconds would be desirable, with memory usage preferably below 10 GB.",
  },
  {
    id: "ex-pets-and-people",
    task: "detection",
    domain: "animals",
    text: "I need an object detection model to locate people, dogs, and cats in indoor and outdoor photographs. The objects may appear at different scales, under varied lighting, and may be partially occluded by furniture or other people. The model will run locally on a MacBook Air with an Apple M4 chip and 16 GB of unified memory, using Metal acceleration where supported. It should aim for a mAP@0.5:0.95 of approximately 0.40 or higher and recall of at least 0.75 for each class. Processing an image within roughly 500 milliseconds would be desirable.",
  },
  {
    id: "indoor-furniture",
    task: "detection",
    domain: "interiors",
    text: "I need an object detection model to locate furniture in indoor photographs. It should handle cluttered rooms, partial occlusion, varied lighting, and objects viewed from different angles. Inference will run on CPU-only backend servers with approximately 8 CPU cores and 16 GB of RAM. The model should aim for a mAP@0.5:0.95 of approximately 0.30 or higher, with typical inference latency below 500 milliseconds per image.",
  },
  {
    id: "ex-furniture-classification",
    task: "classification",
    domain: "retail",
    text: "I need an image classification model for a furniture marketplace that categorizes the primary product in an uploaded photo as a chair, sofa, or table. Each image should primarily contain one product. Inference will run on CPU-only backend servers with approximately 4 CPU cores and 8 GB of RAM. The model should aim for a macro-F1 score of at least 0.85, use less than approximately 1.5 GB of runtime memory, and preferably classify an image within 500 milliseconds."
  },
  {
    id: "ex-handwritten-numbers",
    task: "classification",
    domain: "handwriting",
    text: "I need a lightweight image classification model that recognizes handwritten numbers. The model will run on a CPU-only system with 4 GB of RAM. It should aim for accuracy of at least 90%, use less than approximately 500 MB of runtime memory, and process an image within 200 milliseconds.",
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
