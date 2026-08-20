"use client"

import { useState } from "react"
import { AlertCircle, Boxes, DatabaseZap, FolderOpen, Plus, Play, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { PromptInput } from "@/components/prompt-input"
import { ExamplePrompts } from "@/components/example-prompts"
import { PipelineView } from "@/components/pipeline-view"
import { PipelineOutputs } from "@/components/pipeline-outputs"
import { InferenceSection } from "@/components/inference-section"
import { PostTrainingAssessmentSection } from "@/components/post-training-assessment"
import { EvaluationResults } from "@/components/evaluation-results"
import { PlanningPerformance } from "@/components/planning-performance"
import { usePipeline } from "@/hooks/use-pipeline"
import type { ExamplePrompt, RevisionScope } from "@/lib/pipeline"

type InferenceTask = "classification" | "detection" | "vqa"

function inferredTask(context: unknown): InferenceTask | null {
  if (!context || typeof context !== "object") return null
  const task = String((context as Record<string, unknown>).task ?? "").toLowerCase()
  if (task === "classification" || task === "detection" || task === "vqa") return task
  if (task === "visual question answering") return "vqa"
  return null
}

export default function Page() {
  const [prompt, setPrompt] = useState("")
  const [requiredChanges, setRequiredChanges] = useState("")
  const [preferences, setPreferences] = useState("")
  const [revisionScope, setRevisionScope] = useState<RevisionScope>("automatic")
  const [useGraphRag, setUseGraphRag] = useState(true)
  const [runToLoad, setRunToLoad] = useState("")
  const [selectedExampleJobIds, setSelectedExampleJobIds] = useState<ExamplePrompt["jobIds"]>()

  const {
    pipeline,
    status,
    isLoadedRun,
    activeStepId,
    clarification,
    error,
    jobId,
    chosenParameters,
    context,
    decisionEvidence,
    artifacts,
    evaluationReport,
    planningLLMUsage,
    postTrainingAssessment,
    assessmentEligibility,
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
  } = usePipeline()

  const running = status === "running"
  const cancelling = status === "cancelling"
  const waiting = status === "waiting"
  const done = status === "done"
  const task = inferredTask(context) ?? evaluationReport?.task ?? null
  const canEditPrompt = !running && !cancelling && !waiting

  const startPipeline = () => {
    if (!prompt.trim()) return
    setRequiredChanges("")
    setPreferences("")
    start(prompt.trim(), useGraphRag)
  }

  const startNewPipeline = () => {
    reset()
    setRunToLoad("")
    setRequiredChanges("")
    setPreferences("")
  }

  const toggleGraphRag = () => {
    const nextUseGraphRag = !useGraphRag
    if (selectedExampleJobIds) {
      const previousSuggestion = useGraphRag
        ? selectedExampleJobIds.withGraphRag
        : selectedExampleJobIds.withoutGraphRag
      const nextSuggestion = nextUseGraphRag
        ? selectedExampleJobIds.withGraphRag
        : selectedExampleJobIds.withoutGraphRag
      setRunToLoad((current) => current === previousSuggestion ? nextSuggestion : current)
    }
    setUseGraphRag(nextUseGraphRag)
  }

  return (
    <main
      suppressHydrationWarning
      className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-6 px-4 py-8 sm:px-6 lg:py-12"
    >
      <header className="surface-card flex flex-col gap-3 rounded-2xl border border-white/80 bg-white/82 p-6 sm:p-8">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary shadow-sm">
            <Boxes className="size-4.5" aria-hidden />
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">
              Adaptive Vision
            </span>
            <span className="size-1 rounded-full bg-primary/40" aria-hidden />
            <span className="text-xs text-muted-foreground">Model workspace</span>
          </div>
        </div>
        <h1 className="text-pretty text-2xl font-semibold leading-tight tracking-[-0.025em] sm:text-3xl">
          LLM-based Adaptive CV Model Learning Pipeline
        </h1>
        <p className="max-w-2xl text-pretty text-sm leading-6 text-muted-foreground">
          Describe the computer vision model you need in natural language. The
          pipeline plans the approach, trains a candidate model, evaluates it,
          and hands you a deployable model with a full report.
        </p>
      </header>

      {/* Configuration */}
      <section className="surface-card flex flex-col gap-4 rounded-2xl border border-white/80 bg-white/82 p-4 sm:p-5">
        <div className="grid gap-4 lg:grid-cols-2">
          <PromptInput value={prompt} onChange={setPrompt} disabled={!canEditPrompt} />
          <ExamplePrompts
            disabled={!canEditPrompt}
            onUse={(p) => {
              setPrompt(p.text)
              setSelectedExampleJobIds(p.jobIds)
              if (p.jobIds) {
                setRunToLoad(useGraphRag ? p.jobIds.withGraphRag : p.jobIds.withoutGraphRag)
              }
            }}
          />
        </div>

        <button
          type="button"
          role="switch"
          aria-checked={useGraphRag}
          disabled={!canEditPrompt}
          onClick={toggleGraphRag}
          className="flex items-center justify-between gap-4 rounded-lg border border-border bg-card px-3 py-2.5 text-left transition-colors hover:border-primary/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="flex items-center gap-3">
            <DatabaseZap className="size-4 text-primary" aria-hidden />
            <span>
              <span className="block text-sm font-medium">Use GraphRAG</span>
              <span className="ui-caption block">
                Ground model, datasets and hyperparameters selection in the knowledge graph.
              </span>
            </span>
          </span>
          <span
            aria-hidden
            className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${
              useGraphRag ? "bg-primary" : "bg-muted-foreground/30"
            }`}
          >
            <span
              className={`absolute top-0.5 size-5 rounded-full bg-white shadow-sm transition-transform ${
                useGraphRag ? "translate-x-5" : "translate-x-0.5"
              }`}
            />
          </span>
        </button>

        {clarification && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-950 dark:text-amber-100">
            <div className="mb-1 flex items-center gap-2 font-medium">
              <AlertCircle className="size-4" aria-hidden />
              Please clarify the request before starting the pipeline.
            </div>
            {clarification.reason && (
              <p className="text-sm leading-relaxed">{clarification.reason}</p>
            )}
            {!!clarification.suggestions?.length && (
              <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-sm">
                {clarification.suggestions.map((suggestion) => (
                  <li key={suggestion}>{suggestion}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <div className="mb-1 flex items-center gap-2 font-medium">
              <AlertCircle className="size-4" aria-hidden />
              Pipeline request failed.
            </div>
            <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
              {error}
            </pre>
          </div>
        )}

        <div className="flex items-center gap-3">
          {running || cancelling ? (
            jobId && context ? (
              <Button variant="outline" onClick={stop} disabled={cancelling} className="bg-transparent">
                <Square className="size-4" aria-hidden /> {cancelling ? "Stopping…" : "Stop run"}
              </Button>
            ) : (
              <Button disabled>Starting pipeline…</Button>
            )
          ) : isLoadedRun ? (
            <>
              {!done && (
                <Button onClick={continueRun} disabled={cancelling}>
                  <Play className="size-4" aria-hidden /> Continue pipeline
                </Button>
              )}
              <Button variant="outline" onClick={startNewPipeline} className="bg-transparent">
                <Plus className="size-4" aria-hidden /> New pipeline
              </Button>
            </>
          ) : (
            <Button onClick={startPipeline} disabled={!prompt.trim() || waiting}>
              <Play className="size-4" aria-hidden />
              {done ? "Run again" : "Start Pipeline"}
            </Button>
          )}
        </div>
      </section>

      {/* Existing run */}
      <section className="surface-card flex flex-col gap-4 rounded-2xl border border-white/80 bg-white/82 p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
        <div className="flex min-w-0 items-start gap-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-primary">
            <FolderOpen className="size-4" aria-hidden />
          </span>
          <div>
            <h2 className="ui-card-title">Load an existing run</h2>
            <p className="mt-1 text-[13px] font-normal leading-5 text-muted-foreground/85">
              Restore the saved decisions, rationales, step outputs, and results for a job ID.
            </p>
          </div>
        </div>

        <form
          className="flex shrink-0 flex-col gap-2 sm:flex-row sm:items-center"
          onSubmit={(event) => {
            event.preventDefault()
            loadRun(runToLoad)
          }}
        >
          <label className="sr-only" htmlFor="run-job-id">
            Job ID to load
          </label>
          <input
            id="run-job-id"
            value={runToLoad}
            onChange={(event) => setRunToLoad(event.target.value)}
            placeholder="Paste job ID"
            disabled={running || cancelling}
            spellCheck={false}
            autoComplete="off"
            className="h-9 w-full rounded-md border border-border bg-background px-3 font-mono text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 sm:w-64"
          />
          <Button
            type="submit"
            disabled={!runToLoad.trim() || running || cancelling}
            className="h-9"
          >
            <FolderOpen className="size-4" aria-hidden />
            Load run
          </Button>
        </form>
      </section>

      {/* Pipeline */}
      <PipelineView
        pipeline={pipeline}
        status={status}
        jobId={jobId}
        activeStepId={activeStepId}
        getStepStatus={getStepStatus}
        getRevealed={getRevealed}
        getStepDuration={getStepDuration}
        chosenParameters={chosenParameters}
        context={context}
        decisionEvidence={decisionEvidence}
        requiredChanges={requiredChanges}
        preferences={preferences}
        revisionScope={revisionScope}
        revisionPlan={revisionPlan}
        revisionVerification={revisionVerification}
        onRequiredChanges={setRequiredChanges}
        onPreferences={setPreferences}
        onRevisionScope={setRevisionScope}
        onInterpret={() => planRevision(requiredChanges, preferences, revisionScope)}
        onApplyRevision={applyRevision}
        onCancelRevision={cancelRevision}
        onRevisionStrength={updateRevisionStrength}
        onConfirm={confirmPlan}
      />

      <PlanningPerformance usage={planningLLMUsage} getStepDuration={getStepDuration} />

      <EvaluationResults report={evaluationReport} />

      {/* Deliverables */}
      <PipelineOutputs ready={done} artifacts={artifacts} />

      {/* Inference */}
      <InferenceSection task={task} jobId={jobId} enabled={done} />

      <PostTrainingAssessmentSection
        assessment={postTrainingAssessment}
        eligibility={assessmentEligibility}
        busy={running || cancelling}
        onAnalyze={requestAssessment}
        onRegenerate={redoRecommendation}
        onApprove={approveAssessment}
      />
    </main>
  )
}
