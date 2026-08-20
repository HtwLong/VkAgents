"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  buildPipeline,
  type DeliverableArtifact,
  type EvaluationReport,
  type PlanningLLMUsage,
  type PipelineStage,
  type AssessmentEligibility,
  type PostTrainingAssessment,
  type RevisionPlan,
  type RevisionScope,
  type RevisionVerification,
  type StepStatus,
} from "@/lib/pipeline"
import type { DecisionEvidence } from "@/components/decision-evidence"

export type RunStatus = "idle" | "running" | "cancelling" | "stopped" | "waiting" | "done" | "failed"

export interface BackendCapabilities {
  mode: "full" | "viewer"
  planning: boolean
  graphrag: boolean
  planning_revisions: boolean
  assessment_revision: boolean
  post_training_assessment: boolean
  run_viewing: boolean
  artifact_downloads: boolean
  data_download: boolean
  data_preparation: boolean
  training: boolean
  evaluation_execution: boolean
  inference: boolean
}

const FULL_CAPABILITIES: BackendCapabilities = {
  mode: "full",
  planning: true,
  graphrag: true,
  planning_revisions: true,
  assessment_revision: true,
  post_training_assessment: true,
  run_viewing: true,
  artifact_downloads: true,
  data_download: true,
  data_preparation: true,
  training: true,
  evaluation_execution: true,
  inference: true,
}

const VIEWER_CAPABILITIES: BackendCapabilities = {
  ...FULL_CAPABILITIES,
  mode: "viewer",
  graphrag: false,
  planning_revisions: false,
  assessment_revision: false,
  data_download: false,
  data_preparation: false,
  training: false,
  evaluation_execution: false,
  inference: false,
}

type PipelineContext = Record<string, unknown> | string

interface CompletenessIssue {
  reason?: string | null
  suggestions?: string[] | null
}

interface ArtifactManifestResponse {
  artifacts: Array<{
    id: string
    kind: string
    label: string
    filename: string
    download_url: string
    description?: string
    standalone?: boolean
    required_base_model?: string
    generated_on_download?: boolean
  }>
}

interface RunSnapshotResponse {
  job_id: string
  status: RunStatus
  steps: Record<string, {
    status: StepStatus
    outputs: string[]
    duration_ms: number | null
  }>
  context: PipelineContext | null
  chosen_parameters: unknown
  decision_evidence: Record<string, DecisionEvidence>
  evaluation_report: EvaluationReport | null
  planning_llm_usage: PlanningLLMUsage | null
  post_training_assessment: PostTrainingAssessment | null
  assessment_eligibility: AssessmentEligibility
  artifacts: ArtifactManifestResponse["artifacts"]
  errors?: unknown
  run_state?: {
    status: string
    active_step?: string | null
  } | null
}

// Vercel commonly exposes optional variables as an empty string. Treat that the
// same as an unset value so browser requests continue through our same-origin
// Next.js proxy instead of incorrectly targeting `/api/v1/...` on Vercel.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "/api/backend"
const TRAINING_POLL_MS = 10_000
const DOWNLOAD_POLL_MS = 500

interface DownloadProgress {
  status: "pending" | "running" | "completed" | "failed" | "stopped"
  downloaded: number
  processed: number
  failed?: number
  total?: number
  current_image?: string | null
  previous_downloaded?: number
  active?: boolean
}

interface TrainingStatus {
  status?: string
  current_epoch?: number
  total_epochs?: number
  train_loss?: number
  train_accuracy?: number
  val_loss?: number
  val_accuracy?: number
  val_macro_f1?: number
  val_micro_f1?: number
  val_mAP?: number
  val_mAP50?: number
  tracked_metric?: string
  tracked_value?: number
  elapsed_seconds?: number
  message?: string
  result?: unknown
  error?: unknown
}

const STEP_ORDER = buildPipeline().flatMap((stage) => stage.steps.map((step) => step.id))
const REVISION_STEPS = [
  ["task-interpretation", "/api/v1/planning/task-interpret", "Updated task interpretation"],
  ["check-data", "/api/v1/planning/check-data", "Updated data check"],
  ["model-selection", "/api/v1/planning/select-model", "Updated model selection"],
  ["dataset-selection", "/api/v1/planning/select-datasets", "Updated dataset selection"],
  ["choose-hyperparameters", "/api/v1/planning/choose-hyperparameters", "Updated hyperparameters"],
] as const

function createJobId() {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`
}

function formatJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function summarizeValue(label: string, value: unknown) {
  if (value == null) return `${label}: no data returned.`
  const text = typeof value === "string" ? value : formatJson(value)
  return `${label}:\n${text}`
}

function formatTrainingProgress(progress: TrainingStatus) {
  if (progress.current_epoch == null) {
    return progress.message ?? "Waiting for the first epoch to finish…"
  }

  const metric = (label: string, value?: number) =>
    value == null ? null : `${label} ${value.toFixed(4)}`
  const values = [
    metric("train loss", progress.train_loss),
    metric("train accuracy", progress.train_accuracy),
    metric("validation loss", progress.val_loss),
    metric("validation accuracy", progress.val_accuracy),
    metric("validation macro F1", progress.val_macro_f1),
    metric("validation micro F1", progress.val_micro_f1),
    metric("validation mAP", progress.val_mAP),
    metric("validation mAP@50", progress.val_mAP50),
  ].filter(Boolean)
  const elapsed = progress.elapsed_seconds == null ? "" : ` · ${progress.elapsed_seconds.toFixed(1)}s`
  return `Epoch ${progress.current_epoch}/${progress.total_epochs ?? "?"}${elapsed}${values.length ? `\n${values.join(" · ")}` : ""}`
}

function extractChosenParameters(context: PipelineContext | null) {
  if (!context || typeof context === "string") return null
  return (
    context.hpo_config ??
    context.chosen_parameters ??
    context.candidate ??
    null
  )
}

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(resolve, ms)
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timeout)
        reject(new DOMException("Aborted", "AbortError"))
      },
      { once: true },
    )
  })
}

async function readError(response: Response) {
  try {
    const body = await response.json()
    return typeof body.detail === "string" ? body.detail : formatJson(body)
  } catch {
    return `${response.status} ${response.statusText}`
  }
}

/** Drives the real FastAPI pipeline and stores per-step backend output. */
export function usePipeline() {
  const configuredViewer = process.env.NEXT_PUBLIC_DEPLOYMENT_MODE === "viewer"
  const [capabilities, setCapabilities] = useState<BackendCapabilities>(
    configuredViewer ? VIEWER_CAPABILITIES : FULL_CAPABILITIES,
  )
  const [stepStatuses, setStepStatuses] = useState<Record<string, StepStatus>>({})
  const [stepOutputs, setStepOutputs] = useState<Record<string, string[]>>({})
  const [stepDurations, setStepDurations] = useState<Record<string, number>>({})
  const [status, setStatus] = useState<RunStatus>("idle")
  const [isLoadedRun, setIsLoadedRun] = useState(false)
  const [activeStepId, setActiveStepId] = useState<string | undefined>()
  const [clarification, setClarification] = useState<CompletenessIssue | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [context, setContext] = useState<PipelineContext | null>(null)
  const [chosenParameters, setChosenParameters] = useState<unknown>(null)
  const [artifacts, setArtifacts] = useState<DeliverableArtifact[]>([])
  const [evaluationReport, setEvaluationReport] = useState<EvaluationReport | null>(null)
  const [planningLLMUsage, setPlanningLLMUsage] = useState<PlanningLLMUsage | null>(null)
  const [postTrainingAssessment, setPostTrainingAssessment] = useState<PostTrainingAssessment | null>(null)
  const [assessmentEligibility, setAssessmentEligibility] = useState<AssessmentEligibility | null>(null)
  const [decisionEvidence, setDecisionEvidence] = useState<Record<string, DecisionEvidence>>({})
  const [revisionPlan, setRevisionPlan] = useState<RevisionPlan | null>(null)
  const [revisionVerification, setRevisionVerification] = useState<RevisionVerification | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const contextRef = useRef<PipelineContext | null>(null)
  const jobIdRef = useRef<string | null>(null)
  const chosenParametersRef = useRef<unknown>(null)
  const activeStepIdRef = useRef<string | undefined>(undefined)
  const useGraphRagRef = useRef(true)
  const stepStartedAtRef = useRef<Record<string, number>>({})
  const stepStatusesRef = useRef<Record<string, StepStatus>>({})

  const executionAvailable = capabilities.training && capabilities.evaluation_execution

  useEffect(() => {
    const controller = new AbortController()
    void fetch(`${API_BASE}/api/v1/capabilities`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
        return response.json() as Promise<BackendCapabilities>
      })
      .then(setCapabilities)
      .catch((error) => {
        if (!(error instanceof DOMException && error.name === "AbortError") && configuredViewer) {
          setCapabilities(VIEWER_CAPABILITIES)
        }
      })
    return () => controller.abort()
  }, [configuredViewer])

  const pipeline = useMemo<PipelineStage[]>(() => {
    const base = buildPipeline().filter((stage) => executionAvailable || stage.id === "planning")
    return base.map((stage) => ({
      ...stage,
      steps: stage.steps.map((step) => ({
        ...step,
        outputs: stepOutputs[step.id] ?? step.outputs,
      })),
    }))
  }, [executionAvailable, stepOutputs])

  const setCurrentContext = useCallback((next: PipelineContext) => {
    contextRef.current = next
    setContext(next)
    const params = extractChosenParameters(next)
    if (params) {
      chosenParametersRef.current = params
      setChosenParameters(params)
    }
  }, [])

  const appendOutput = useCallback((stepId: string, line: string) => {
    setStepOutputs((prev) => ({
      ...prev,
      [stepId]: [...(prev[stepId] ?? []), line],
    }))
  }, [])

  const upsertOutput = useCallback((stepId: string, prefix: string, line: string) => {
    setStepOutputs((previous) => {
      const outputs = previous[stepId] ?? []
      const index = outputs.findIndex((output) => output.startsWith(prefix))
      if (index === -1) {
        return { ...previous, [stepId]: [...outputs, line] }
      }
      const updated = [...outputs]
      updated[index] = line
      return { ...previous, [stepId]: updated }
    })
  }, [])

  const markStep = useCallback((stepId: string, nextStatus: StepStatus) => {
    stepStatusesRef.current = { ...stepStatusesRef.current, [stepId]: nextStatus }
    setStepStatuses((prev) => ({ ...prev, [stepId]: nextStatus }))
    if (nextStatus === "running") {
      if (stepStartedAtRef.current[stepId] === undefined) {
        stepStartedAtRef.current[stepId] = Date.now()
      }
      activeStepIdRef.current = stepId
      setActiveStepId(stepId)
      return
    }

    if (activeStepIdRef.current === stepId) {
      activeStepIdRef.current = undefined
      setActiveStepId(undefined)
    }
  }, [])

  const requestJson = useCallback(
    async <T,>(
      path: string,
      body?: unknown,
      signal?: AbortSignal,
      method?: "POST" | "PUT" | "DELETE",
    ): Promise<T> => {
      const response = await fetch(`${API_BASE}${path}`, {
        method: method ?? (body ? "POST" : "GET"),
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal,
      })

      if (!response.ok) {
        throw new Error(await readError(response))
      }

      return response.json() as Promise<T>
    },
    [],
  )

  const finishStep = useCallback(
    async (stepId: string, nextStatus: "done" | "failed") => {
      const startedAt = stepStartedAtRef.current[stepId]
      const durationMs = startedAt === undefined ? 0 : Math.max(0, Date.now() - startedAt)
      delete stepStartedAtRef.current[stepId]
      setStepDurations((previous) => ({ ...previous, [stepId]: durationMs }))
      markStep(stepId, nextStatus)

      const currentJobId = jobIdRef.current
      if (currentJobId) {
        await requestJson(
          `/api/v1/runs/${encodeURIComponent(currentJobId)}/steps/${encodeURIComponent(stepId)}/timing`,
          { duration_ms: durationMs, status: nextStatus },
          undefined,
          "PUT",
        )
      }
    },
    [markStep, requestJson],
  )

  const runContextStep = useCallback(
    async (
      stepId: string,
      path: string,
      signal: AbortSignal,
      label: string,
      includeGraphRagOption = false,
    ) => {
      const currentJobId = jobIdRef.current
      const currentContext = contextRef.current
      if (!currentJobId || !currentContext) {
        throw new Error("Pipeline context is missing. Start from completeness check again.")
      }

      markStep(stepId, "running")
      if (
        stepId === "model-selection" ||
        stepId === "dataset-selection" ||
        stepId === "choose-hyperparameters"
      ) {
        setDecisionEvidence((previous) => {
          const next = { ...previous }
          delete next[stepId]
          return next
        })
      }
      appendOutput(stepId, `POST ${path}`)

      const result = await requestJson<{
        context: PipelineContext
        decision_evidence?: DecisionEvidence | null
      }>(
        path,
        {
          context: currentContext,
          job_id: currentJobId,
          ...(includeGraphRagOption
            ? {
                use_graphrag: useGraphRagRef.current,
              }
            : {}),
        },
        signal,
      )

      setCurrentContext(result.context)
      if (result.decision_evidence) {
        setDecisionEvidence((previous) => ({
          ...previous,
          [stepId]: result.decision_evidence as DecisionEvidence,
        }))
      }
      appendOutput(stepId, summarizeValue(label, result.context))
      await finishStep(stepId, "done")
      return result.context
    },
    [appendOutput, finishStep, markStep, requestJson, setCurrentContext],
  )

  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    contextRef.current = null
    jobIdRef.current = null
    chosenParametersRef.current = null
    activeStepIdRef.current = undefined
    stepStartedAtRef.current = {}
    stepStatusesRef.current = {}
    setStatus("idle")
    setIsLoadedRun(false)
    setActiveStepId(undefined)
    setStepStatuses({})
    setStepOutputs({})
    setStepDurations({})
    setClarification(null)
    setError(null)
    setJobId(null)
    setContext(null)
    setChosenParameters(null)
    setArtifacts([])
    setEvaluationReport(null)
    setPlanningLLMUsage(null)
    setPostTrainingAssessment(null)
    setAssessmentEligibility(null)
    setDecisionEvidence({})
    setRevisionPlan(null)
    setRevisionVerification(null)
  }, [])

  const stop = useCallback(async () => {
    const currentJobId = jobIdRef.current
    if (!currentJobId) return
    setStatus("cancelling")
    setError(null)
    try {
      await requestJson(
        `/api/v1/runs/${encodeURIComponent(currentJobId)}/cancel`,
        {},
      )
      abortRef.current?.abort()
      abortRef.current = null
      const pollController = new AbortController()
      abortRef.current = pollController
      while (true) {
        await sleep(500, pollController.signal)
        const snapshot = await requestJson<RunSnapshotResponse>(
          `/api/v1/runs/${encodeURIComponent(currentJobId)}`,
          undefined,
          pollController.signal,
        )
        if (snapshot.status === "stopped" || snapshot.status === "done") {
          const statuses = Object.fromEntries(
            Object.entries(snapshot.steps).map(([id, step]) => [id, step.status]),
          )
          stepStatusesRef.current = statuses
          setStepStatuses(statuses)
          activeStepIdRef.current = undefined
          setActiveStepId(undefined)
          setIsLoadedRun(true)
          setStatus(snapshot.status)
          abortRef.current = null
          return
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      setError(`Cancellation was requested but not confirmed: ${err instanceof Error ? err.message : String(err)}`)
      setStatus("failed")
    }
  }, [requestJson])

  const runExecutionAndEvaluation = useCallback(
    async (signal: AbortSignal, startAt = "download-data") => {
      const currentJobId = jobIdRef.current
      const params = chosenParametersRef.current
      if (!currentJobId || !params) {
        throw new Error("Hyperparameters are missing. Complete planning before execution.")
      }

      const startIndex = STEP_ORDER.indexOf(startAt)
      const runs = (stepId: string) => STEP_ORDER.indexOf(stepId) >= startIndex

      if (stepStatusesRef.current["ask-change-requests"] !== "done") {
        await finishStep("ask-change-requests", "done")
      }

      if (runs("download-data")) {
      markStep("download-data", "running")
      const existingDownload = await requestJson<DownloadProgress>(
        `/api/v1/download-data/status/${encodeURIComponent(currentJobId)}`,
        undefined,
        signal,
      )
      appendOutput("download-data", existingDownload.active
        ? "Reconnected to the active data download."
        : "POST /api/v1/download-data")
      let download: unknown
      let downloadError: unknown
      let downloadSettled = !existingDownload.active
      if (!existingDownload.active) {
        downloadSettled = false
        void requestJson<unknown>(
          "/api/v1/download-data",
          { chosen_parameters: params, job_id: currentJobId },
          signal,
        ).then((result) => {
          download = result
        }).catch((caught) => {
          downloadError = caught
        }).finally(() => {
          downloadSettled = true
        })
      }

      while (!downloadSettled || existingDownload.active) {
        await sleep(DOWNLOAD_POLL_MS, signal)
        const progress = await requestJson<DownloadProgress>(
          `/api/v1/download-data/status/${encodeURIComponent(currentJobId)}`,
          undefined,
          signal,
        )
        const total = progress.total ?? 0
        const amount = total > 0 ? `${progress.downloaded}/${total}` : `${progress.downloaded}`
        const failures = progress.failed ? `, ${progress.failed} failed` : ""
        const previous = progress.previous_downloaded
          ? `; previous attempt reached ${progress.previous_downloaded}`
          : ""
        const current = progress.current_image ? ` — ${progress.current_image}` : ""
        upsertOutput(
          "download-data",
          "Download progress:",
          `Download progress: ${amount} images ready${failures}${previous}${current}`,
        )
        if (existingDownload.active && !progress.active) {
          if (progress.status !== "completed") {
            throw new Error("The previously active data download did not complete successfully.")
          }
          break
        }
        if (!existingDownload.active && downloadSettled) break
      }
      if (downloadError) throw downloadError
      appendOutput("download-data", download === undefined
        ? "Download completed while disconnected."
        : summarizeValue("Download output", download))
      await finishStep("download-data", "done")
      }

      if (runs("prepare-data")) {
      markStep("prepare-data", "running")
      appendOutput("prepare-data", "POST /api/v1/prepare-data")
      const prepare = await requestJson<unknown>(
        "/api/v1/prepare-data",
        { chosen_parameters: params, job_id: currentJobId },
        signal,
      )
      appendOutput("prepare-data", summarizeValue("Prepare output", prepare))
      await finishStep("prepare-data", "done")
      }

      if (runs("train-model")) {
      markStep("train-model", "running")
      let reconnectTraining = false
      try {
        const existingTraining = await requestJson<Record<string, unknown>>(
          `/api/v1/train/result/${currentJobId}`,
          undefined,
          signal,
        )
        reconnectTraining = existingTraining.status === "running"
      } catch {
        reconnectTraining = false
      }
      if (reconnectTraining) {
        appendOutput("train-model", "Reconnected to the active training job.")
      } else {
        appendOutput("train-model", "POST /api/v1/train/start")
        const trainStart = await requestJson<{ status_url?: string }>(
          "/api/v1/train/start",
          { chosen_parameters: params, job_id: currentJobId },
          signal,
        )
        appendOutput("train-model", summarizeValue("Training started", trainStart))
      }

      while (true) {
        await sleep(TRAINING_POLL_MS, signal)
        const trainingStatus = await requestJson<TrainingStatus>(
          `/api/v1/train/status/${currentJobId}`,
          undefined,
          signal,
        )
        upsertOutput(
          "train-model",
          "Training progress:",
          `Training progress:\n${formatTrainingProgress(trainingStatus)}`,
        )

        if (trainingStatus.status === "completed") {
          appendOutput("train-model", summarizeValue("Training result", trainingStatus.result))
          break
        }
        if (trainingStatus.status === "error") {
          throw new Error(summarizeValue("Training failed", trainingStatus))
        }
      }
      await finishStep("train-model", "done")
      }

      if (runs("running-evaluation")) {
      markStep("running-evaluation", "running")
      appendOutput("running-evaluation", "POST /api/v1/evaluate")
      const evaluation = await requestJson<unknown>(
        "/api/v1/evaluate",
        { chosen_parameters: params, job_id: currentJobId },
        signal,
      )
      appendOutput("running-evaluation", summarizeValue("Evaluation output", evaluation))
      await finishStep("running-evaluation", "done")
      }

      if (runs("preparing-trained-model")) {
      markStep("preparing-trained-model", "running")
      appendOutput("preparing-trained-model", `GET /artifacts/${currentJobId}/manifest`)
      const manifest = await requestJson<ArtifactManifestResponse>(
        `/artifacts/${currentJobId}/manifest`,
        undefined,
        signal,
      )
      const deliverables = manifest.artifacts.map((artifact) => ({
        id: artifact.id,
        kind: artifact.kind,
        label: artifact.label,
        filename: artifact.filename,
        downloadUrl: artifact.download_url.startsWith("http")
          ? artifact.download_url
          : `${API_BASE}${artifact.download_url}`,
        description: artifact.description,
        standalone: artifact.standalone,
        requiredBaseModel: artifact.required_base_model,
        generatedOnDownload: artifact.generated_on_download,
      }))
      if (!deliverables.some((artifact) =>
        ["full_model", "lora_adapter_bundle", "merged_model"].includes(artifact.kind)
      )) {
        throw new Error("The artifact manifest contains no downloadable model artifact.")
      }
      setArtifacts(deliverables)
      appendOutput(
        "preparing-trained-model",
        `Available model artifacts: ${deliverables
          .filter((artifact) => ["full_model", "lora_adapter_bundle", "merged_model"].includes(artifact.kind))
          .map((artifact) => artifact.label)
          .join(", ")}.`,
      )
      await finishStep("preparing-trained-model", "done")
      }

      if (runs("preparing-results")) {
      markStep("preparing-results", "running")
      appendOutput("preparing-results", `GET /api/v1/evaluate/${currentJobId}/report`)
      const report = await requestJson<EvaluationReport>(
        `/api/v1/evaluate/${currentJobId}/report`,
        undefined,
        signal,
      )
      setEvaluationReport(report)
      setPostTrainingAssessment(null)
      setAssessmentEligibility({
        eligible: true,
        can_create_revision: true,
        reason: null,
        revision_reason: null,
      })
      appendOutput("preparing-results", "Interactive evaluation results are ready.")
      await finishStep("preparing-results", "done")
      const usageSnapshot = await requestJson<RunSnapshotResponse>(
        `/api/v1/runs/${encodeURIComponent(currentJobId)}`,
        undefined,
        signal,
      )
      setPlanningLLMUsage(usageSnapshot.planning_llm_usage)
      }

      setStatus("done")
      setActiveStepId(undefined)
    },
    [appendOutput, finishStep, markStep, requestJson, upsertOutput],
  )

  const runPlanning = useCallback(
    async (prompt: string, signal: AbortSignal) => {
      const nextJobId = createJobId()
      jobIdRef.current = nextJobId
      setJobId(nextJobId)

      markStep("task-interpretation", "running")
      appendOutput("task-interpretation", "POST /api/v1/planning/completenesscheck")
      const completeness = await requestJson<{
        accept: boolean
        reason?: string | null
        suggestions?: string[] | null
        context?: PipelineContext | null
      }>(
        "/api/v1/planning/completenesscheck",
        { job_id: nextJobId, user_prompt: prompt, user_replies: [] },
        signal,
      )

      if (!completeness.accept) {
        setClarification({
          reason: completeness.reason,
          suggestions: completeness.suggestions,
        })
        appendOutput(
          "task-interpretation",
          summarizeValue("Completeness check needs clarification", completeness),
        )
        setStatus("idle")
        setActiveStepId(undefined)
        await finishStep("task-interpretation", "failed")
        const snapshot = await requestJson<RunSnapshotResponse>(
          `/api/v1/runs/${encodeURIComponent(nextJobId)}`,
          undefined,
          signal,
        )
        setPlanningLLMUsage(snapshot.planning_llm_usage)
        return
      }

      if (!completeness.context) {
        throw new Error("Completeness check accepted the prompt but did not return context.")
      }

      setCurrentContext(completeness.context)
      appendOutput("task-interpretation", "Completeness check accepted the prompt.")

      await runContextStep("task-interpretation", "/api/v1/planning/task-interpret", signal, "Task interpretation", true)
      await runContextStep("check-data", "/api/v1/planning/check-data", signal, "Data check", true)
      await runContextStep(
        "model-selection",
        "/api/v1/planning/select-model",
        signal,
        "Model selection",
        true,
      )
      await runContextStep(
        "dataset-selection",
        "/api/v1/planning/select-datasets",
        signal,
        "Dataset selection",
        true,
      )
      await runContextStep(
        "choose-hyperparameters",
        "/api/v1/planning/choose-hyperparameters",
        signal,
        "Hyperparameters",
        true,
      )

      const snapshot = await requestJson<RunSnapshotResponse>(
        `/api/v1/runs/${encodeURIComponent(nextJobId)}`,
        undefined,
        signal,
      )
      setPlanningLLMUsage(snapshot.planning_llm_usage)

      markStep("ask-change-requests", "running")
      appendOutput(
        "ask-change-requests",
        executionAvailable
          ? "Submit change requests or continue to execution."
          : "Review the proposed plan. Execution is intentionally unavailable in viewer mode.",
      )
      setStatus("waiting")
    },
    [appendOutput, executionAvailable, finishStep, markStep, requestJson, runContextStep, setCurrentContext],
  )

  const start = useCallback(
    async (prompt: string, useGraphRag = true) => {
      reset()
      useGraphRagRef.current = useGraphRag
      const controller = new AbortController()
      abortRef.current = controller
      setStatus("running")

      try {
        await runPlanning(prompt, controller.signal)
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        setStatus("failed")
        const failedStepId = activeStepIdRef.current
        if (failedStepId) {
          try {
            await finishStep(failedStepId, "failed")
          } catch {
            markStep(failedStepId, "failed")
          }
        }
      }
    },
    [finishStep, markStep, reset, runPlanning],
  )

  const loadRun = useCallback(
    async (requestedJobId: string) => {
      const requested = requestedJobId.trim()
      if (!requested) return

      reset()
      setStatus("running")
      try {
        const snapshot = await requestJson<RunSnapshotResponse>(
          `/api/v1/runs/${encodeURIComponent(requested)}`,
        )
        jobIdRef.current = snapshot.job_id
        setJobId(snapshot.job_id)
        setIsLoadedRun(true)
        contextRef.current = snapshot.context
        setContext(snapshot.context)
        if (snapshot.context && typeof snapshot.context !== "string") {
          useGraphRagRef.current = snapshot.context.use_graphrag !== false
        }
        chosenParametersRef.current = snapshot.chosen_parameters
        setChosenParameters(snapshot.chosen_parameters)
        setDecisionEvidence(snapshot.decision_evidence ?? {})
        setEvaluationReport(snapshot.evaluation_report)
        setPlanningLLMUsage(snapshot.planning_llm_usage)
        setPostTrainingAssessment(snapshot.post_training_assessment)
        setAssessmentEligibility(snapshot.assessment_eligibility)
        setStepStatuses(Object.fromEntries(
          Object.entries(snapshot.steps).map(([id, step]) => [id, step.status]),
        ))
        stepStatusesRef.current = Object.fromEntries(
          Object.entries(snapshot.steps).map(([id, step]) => [id, step.status]),
        )
        setStepOutputs(Object.fromEntries(
          Object.entries(snapshot.steps).map(([id, step]) => [id, step.outputs]),
        ))
        setStepDurations(Object.fromEntries(
          Object.entries(snapshot.steps)
            .filter(([, step]) => step.duration_ms !== null)
            .map(([id, step]) => [id, step.duration_ms as number]),
        ))
        setArtifacts(snapshot.artifacts.map((artifact) => ({
          id: artifact.id,
          kind: artifact.kind,
          label: artifact.label,
          filename: artifact.filename,
          downloadUrl: artifact.download_url.startsWith("http")
            ? artifact.download_url
            : `${API_BASE}${artifact.download_url}`,
          description: artifact.description,
          standalone: artifact.standalone,
          requiredBaseModel: artifact.required_base_model,
          generatedOnDownload: artifact.generated_on_download,
        })))
        setStatus(snapshot.status)
        if (snapshot.errors) setError(summarizeValue("Persisted run errors", snapshot.errors))
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        setStatus("failed")
      }
    },
    [requestJson, reset],
  )

  const planRevision = useCallback(
    async (requiredChanges: string, preferences: string, scope: RevisionScope) => {
      const currentJobId = jobIdRef.current
      const currentContext = contextRef.current
      if (!currentJobId || !currentContext) return
      const controller = new AbortController()
      abortRef.current = controller
      setStatus("running")
      setError(null)
      try {
        const result = await requestJson<{ plan: RevisionPlan }>(
          "/api/v1/planning/plan-revision",
          {
            context: currentContext,
            job_id: currentJobId,
            required_changes: requiredChanges.trim(),
            preferences: preferences.trim(),
            requested_target: scope,
          },
          controller.signal,
        )
        setRevisionPlan(result.plan)
        setRevisionVerification(null)
        const snapshot = await requestJson<RunSnapshotResponse>(
          `/api/v1/runs/${encodeURIComponent(currentJobId)}`,
          undefined,
          controller.signal,
        )
        setPlanningLLMUsage(snapshot.planning_llm_usage)
        setStatus("waiting")
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return
        setError(err instanceof Error ? err.message : String(err))
        setStatus("waiting")
      }

    },
    [requestJson],
  )

  const applyRevision = useCallback(async () => {
    const currentJobId = jobIdRef.current
    const currentContext = contextRef.current
    if (!currentJobId || !currentContext || !revisionPlan) return
    const controller = new AbortController()
    abortRef.current = controller
    setStatus("running")
    setError(null)
    try {
      const activated = await requestJson<{ context: PipelineContext }>(
        "/api/v1/planning/activate-revision",
        { context: currentContext, plan: revisionPlan, job_id: currentJobId },
        controller.signal,
      )
      setCurrentContext(activated.context)
      const startId = revisionPlan.restart_from
      const startIndex = REVISION_STEPS.findIndex(([id]) => id === startId)
      const steps = REVISION_STEPS.slice(startIndex)
      setStepStatuses((previous) => ({
        ...previous,
        ...Object.fromEntries(steps.map(([id]) => [id, "pending" as StepStatus])),
      }))
      for (const [id, endpoint, label] of steps) {
        await runContextStep(id, endpoint, controller.signal, label, true)
      }
      const verification = await requestJson<RevisionVerification>(
        "/api/v1/planning/verify-revision",
        { context: contextRef.current },
        controller.signal,
      )
      setRevisionVerification(verification)
      setRevisionPlan(null)
      const usageSnapshot = await requestJson<RunSnapshotResponse>(
        `/api/v1/runs/${encodeURIComponent(currentJobId)}`,
        undefined,
        controller.signal,
      )
      setPlanningLLMUsage(usageSnapshot.planning_llm_usage)
      markStep("ask-change-requests", "running")
      appendOutput(
        "ask-change-requests",
        verification.satisfied
          ? "Revision applied and all required changes verified."
          : "Revision completed, but one or more required changes were not satisfied.",
      )
      setStatus("waiting")
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      setError(err instanceof Error ? err.message : String(err))
      setStatus("waiting")
    }
  }, [appendOutput, markStep, requestJson, revisionPlan, runContextStep, setCurrentContext])

  const cancelRevision = useCallback(() => {
    setRevisionPlan(null)
    setRevisionVerification(null)
  }, [])

  const updateRevisionStrength = useCallback(
    (changeId: string, strength: "required" | "preferred") => {
      setRevisionPlan((current) => current ? {
        ...current,
        changes: current.changes.map((change) =>
          change.id === changeId ? { ...change, strength } : change,
        ),
      } : null)
    },
    [],
  )

  const confirmPlan = useCallback(async () => {
    if (!executionAvailable) {
      await finishStep("ask-change-requests", "done")
      appendOutput("ask-change-requests", "Planning completed. No execution was started.")
      setStatus("done")
      return
    }
    const controller = new AbortController()
    abortRef.current = controller
    setStatus("running")
    setError(null)
    try {
      await runExecutionAndEvaluation(controller.signal)
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      setError(err instanceof Error ? err.message : String(err))
      setStatus("failed")
    }
  }, [appendOutput, executionAvailable, finishStep, runExecutionAndEvaluation])

  const requestAssessment = useCallback(async () => {
    const currentJobId = jobIdRef.current
    if (!currentJobId) return
    setStatus("running")
    setError(null)
    try {
      const result = await requestJson<{
        assessment: PostTrainingAssessment
        eligibility: AssessmentEligibility
      }>(`/api/v1/runs/${encodeURIComponent(currentJobId)}/assessment`, {}, undefined)
      setPostTrainingAssessment(result.assessment)
      setAssessmentEligibility(result.eligibility)
      setStatus("done")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStatus("done")
    }
  }, [requestJson])

  const redoRecommendation = useCallback(async () => {
    const currentJobId = jobIdRef.current
    if (!currentJobId || !postTrainingAssessment) return
    setStatus("running")
    setError(null)
    try {
      const result = await requestJson<{
        assessment: PostTrainingAssessment
        eligibility: AssessmentEligibility
      }>(`/api/v1/runs/${encodeURIComponent(currentJobId)}/assessment/recommendation`, {}, undefined)
      setPostTrainingAssessment(result.assessment)
      setAssessmentEligibility(result.eligibility)
      setStatus("done")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStatus("done")
    }
  }, [postTrainingAssessment, requestJson])

  const approveAssessment = useCallback(async () => {
    if (!capabilities.assessment_revision) {
      setError("Starting an improved execution run is unavailable in viewer mode.")
      return
    }
    const parentJobId = jobIdRef.current
    const plan = postTrainingAssessment?.recommended_plan
    if (!parentJobId || !postTrainingAssessment || !plan) return
    const controller = new AbortController()
    abortRef.current = controller
    setStatus("running")
    setError(null)
    try {
      const fork = await requestJson<{
        job_id: string
        context: PipelineContext
        plan: RevisionPlan
        restart_from: RevisionPlan["restart_from"]
        reused_steps: string[]
      }>("/api/v1/planning/fork-revision", {
        parent_job_id: parentJobId,
        assessment_id: postTrainingAssessment.assessment_id,
        plan,
      }, controller.signal)
      jobIdRef.current = fork.job_id
      setJobId(fork.job_id)
      setCurrentContext(fork.context)
      setIsLoadedRun(false)
      setArtifacts([])
      setEvaluationReport(null)
      setPlanningLLMUsage(null)
      setPostTrainingAssessment(null)
      setAssessmentEligibility(null)
      setDecisionEvidence({})
      setStepOutputs({})
      setStepDurations({})
      const reused = Object.fromEntries(fork.reused_steps.map((id) => [id, "done" as StepStatus]))
      setStepStatuses(reused)
      stepStatusesRef.current = reused
      const startIndex = REVISION_STEPS.findIndex(([id]) => id === fork.restart_from)
      for (const [id, endpoint, label] of REVISION_STEPS.slice(startIndex)) {
        await runContextStep(id, endpoint, controller.signal, label, true)
      }
      markStep("ask-change-requests", "running")
      await runExecutionAndEvaluation(controller.signal)
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      setError(err instanceof Error ? err.message : String(err))
      setStatus("failed")
    }
  }, [capabilities.assessment_revision, markStep, postTrainingAssessment, requestJson, runContextStep, runExecutionAndEvaluation, setCurrentContext])

  const continueRun = useCallback(
    async () => {
      const currentJobId = jobIdRef.current
      if (!currentJobId) return
      const controller = new AbortController()
      abortRef.current = controller
      setStatus("running")
      setError(null)

      try {
        if (executionAvailable) {
          await requestJson(
            `/api/v1/runs/${encodeURIComponent(currentJobId)}/errors`,
            undefined,
            controller.signal,
            "DELETE",
          )
        }
        // Re-read durable state: a backend worker may have completed after the UI
        // was stopped, in which case its validated artifact must be skipped.
        const snapshot = await requestJson<RunSnapshotResponse>(
          `/api/v1/runs/${encodeURIComponent(currentJobId)}`,
          undefined,
          controller.signal,
        )
        const statuses = Object.fromEntries(
          Object.entries(snapshot.steps).map(([id, step]) => [id, step.status]),
        )
        stepStatusesRef.current = statuses
        setStepStatuses(statuses)
        contextRef.current = snapshot.context
        setContext(snapshot.context)
        chosenParametersRef.current = snapshot.chosen_parameters
        setChosenParameters(snapshot.chosen_parameters)
        const index = STEP_ORDER.findIndex((stepId) => statuses[stepId] !== "done")
        if (index < 0) {
          setStatus("done")
          return
        }
        const stepId = STEP_ORDER[index]

        if (index <= STEP_ORDER.indexOf("choose-hyperparameters")) {
          const planningSteps = [
            ["task-interpretation", "/api/v1/planning/task-interpret", "Task interpretation"],
            ["check-data", "/api/v1/planning/check-data", "Data check"],
            ["model-selection", "/api/v1/planning/select-model", "Model selection"],
            ["dataset-selection", "/api/v1/planning/select-datasets", "Dataset selection"],
            ["choose-hyperparameters", "/api/v1/planning/choose-hyperparameters", "Hyperparameters"],
          ] as const
          for (const [id, path, label] of planningSteps.slice(index)) {
            await runContextStep(
              id,
              path,
              controller.signal,
              label,
              true,
            )
          }
          markStep("ask-change-requests", "running")
          setStatus("waiting")
          return
        }

        if (!executionAvailable) {
          if (stepId === "ask-change-requests") markStep("ask-change-requests", "done")
          setStatus("done")
          return
        }
        const executionStart = stepId === "ask-change-requests" ? "download-data" : stepId
        await runExecutionAndEvaluation(controller.signal, executionStart)
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return
        setError(err instanceof Error ? err.message : String(err))
        setStatus("failed")
        const failedStepId = activeStepIdRef.current
        if (failedStepId) markStep(failedStepId, "failed")
      }
    },
    [executionAvailable, markStep, requestJson, runContextStep, runExecutionAndEvaluation],
  )

  const getStepStatus = useCallback(
    (stepId: string): StepStatus => stepStatuses[stepId] ?? "pending",
    [stepStatuses],
  )

  const getRevealed = useCallback(
    (stepId: string) => (stepOutputs[stepId] ?? []).length,
    [stepOutputs],
  )

  const getStepDuration = useCallback(
    (stepId: string) => stepDurations[stepId],
    [stepDurations],
  )

  return {
    pipeline,
    capabilities,
    executionAvailable,
    status,
    isLoadedRun,
    activeStepId,
    clarification,
    error,
    jobId,
    context,
    chosenParameters,
    artifacts,
    evaluationReport,
    planningLLMUsage,
    postTrainingAssessment,
    assessmentEligibility,
    decisionEvidence,
    start,
    loadRun,
    reset,
    stop,
    revisionPlan,
    revisionVerification,
    planRevision,
    applyRevision,
    cancelRevision,
    updateRevisionStrength,
    confirmPlan,
    continueRun,
    requestAssessment,
    redoRecommendation,
    approveAssessment,
    getStepStatus,
    getRevealed,
    getStepDuration,
  }
}
