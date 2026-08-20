"use client"

import { useState } from "react"
import { Activity, Check, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import type {
  PipelineStage,
  RevisionPlan,
  RevisionScope,
  RevisionVerification,
  StepStatus,
} from "@/lib/pipeline"
import type { RunStatus } from "@/hooks/use-pipeline"
import { PipelineStep } from "@/components/pipeline-step"
import { DecisionEvidencePanel, type DecisionEvidence } from "@/components/decision-evidence"
import { DatasetSplitPlan } from "@/components/dataset-split-plan"

const STAGE_INDEX = ["01", "02", "03"]

export function PipelineView({
  pipeline,
  status,
  jobId,
  activeStepId,
  getStepStatus,
  getRevealed,
  getStepDuration,
  chosenParameters,
  context,
  decisionEvidence,
  requiredChanges,
  preferences,
  revisionScope,
  revisionPlan,
  revisionVerification,
  onRequiredChanges,
  onPreferences,
  onRevisionScope,
  onInterpret,
  onApplyRevision,
  onCancelRevision,
  onRevisionStrength,
  onConfirm,
  executionAvailable,
  planningRevisionAvailable,
}: {
  pipeline: PipelineStage[]
  status: RunStatus
  jobId: string | null
  activeStepId?: string
  getStepStatus: (id: string) => StepStatus
  getRevealed: (id: string) => number
  getStepDuration: (id: string) => number | undefined
  chosenParameters: unknown
  context: unknown
  decisionEvidence: Record<string, DecisionEvidence>
  requiredChanges: string
  preferences: string
  revisionScope: RevisionScope
  revisionPlan: RevisionPlan | null
  revisionVerification: RevisionVerification | null
  onRequiredChanges: (value: string) => void
  onPreferences: (value: string) => void
  onRevisionScope: (value: RevisionScope) => void
  onInterpret: () => void
  onApplyRevision: () => void
  onCancelRevision: () => void
  onRevisionStrength: (changeId: string, strength: "required" | "preferred") => void
  onConfirm: () => void
  executionAvailable: boolean
  planningRevisionAvailable: boolean
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <section className="surface-card overflow-hidden rounded-2xl border border-white/80 bg-card">
      <header className="flex flex-col gap-2 border-b border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-primary" aria-hidden />
          <h2 className="ui-card-title">Pipeline updates</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {jobId && (
            <span className="rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground">
              Job ID: <span className="text-foreground">{jobId}</span>
            </span>
          )}
          <span
            className={cn(
              "flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] uppercase tracking-wide",
              status === "running" && "border-primary/40 text-primary",
              status === "waiting" && "border-amber-500/40 text-amber-600",
              status === "done" && "border-primary/40 text-primary",
              status === "failed" && "border-destructive/40 text-destructive",
              status === "idle" && "border-border text-muted-foreground",
            )}
          >
            <span
              className={cn(
                "size-1.5 rounded-full",
                status === "running" && "animate-pulse bg-primary",
                status === "waiting" && "animate-pulse bg-amber-500",
                status === "done" && "bg-primary",
                status === "failed" && "bg-destructive",
                status === "idle" && "bg-muted-foreground",
              )}
              aria-hidden
            />
            {status === "running"
              ? "running"
              : status === "waiting"
                ? "waiting"
                : status === "done"
                  ? "complete"
                  : status === "failed"
                    ? "failed"
                    : "idle"}
          </span>
        </div>
      </header>

      <div className="flex flex-col gap-5 p-4">
        {pipeline.map((stage, si) => {
          const stageDone = stage.steps.every(
            (s) => getStepStatus(s.id) === "done",
          )
          const stageActive = stage.steps.some(
            (s) => getStepStatus(s.id) === "running",
          )
          return (
            <div key={stage.id} className="flex flex-col gap-2.5">
              <div className="flex items-center gap-3">
                <span
                  className={cn(
                    "flex size-7 items-center justify-center rounded-md border font-mono text-xs",
                    stageDone || stageActive
                      ? "border-primary/50 bg-primary/10 text-primary"
                      : "border-border bg-secondary text-muted-foreground",
                  )}
                >
                  {STAGE_INDEX[si]}
                </span>
                <div className="flex flex-col">
                  <h3 className="ui-subsection-title">
                    {stage.title}
                  </h3>
                  <p className="mt-0.5 text-[13px] font-normal leading-5 text-muted-foreground">
                    {stage.description}
                  </p>
                </div>
              </div>

              <div className="ml-3 flex flex-col gap-2 border-l border-border/60 pl-4">
                {stage.steps.map((step) => (
                  <PipelineStep
                    key={step.id}
                    step={step}
                    status={getStepStatus(step.id)}
                    expanded={expanded.has(step.id) || activeStepId === step.id}
                    revealedCount={getRevealed(step.id)}
                    durationMs={getStepDuration(step.id)}
                    onToggle={() => toggle(step.id)}
                    childrenBeforeOutputs={
                      step.id === "model-selection" ||
                      step.id === "dataset-selection" ||
                      step.id === "choose-hyperparameters"
                    }
                  >
                    {(step.id === "model-selection" ||
                      step.id === "dataset-selection" ||
                      step.id === "choose-hyperparameters") &&
                      decisionEvidence[step.id] && (
                        <DecisionEvidencePanel evidence={decisionEvidence[step.id]} />
                      )}
                    {step.id === "ask-change-requests" && chosenParameters != null && (
                      <div className="flex flex-col gap-3 rounded-md border border-border bg-card p-3">
                        <div className="flex flex-col gap-1">
                          <span className="ui-section-label">
                            {status === "waiting"
                              ? "Proposed Hyperparameters"
                              : "Accepted Final Configuration"}
                          </span>
                          <pre className="ui-code-block max-h-72 overflow-auto rounded-md border border-border bg-background p-3">
                            {JSON.stringify(chosenParameters, null, 2)}
                          </pre>
                        </div>

                        {status === "waiting" && planningRevisionAvailable && (
                          <>
                            <label className="flex flex-col gap-1.5 text-xs font-medium text-foreground">
                              Change scope
                              <select
                                value={revisionScope}
                                onChange={(event) => onRevisionScope(event.target.value as RevisionScope)}
                                className="rounded-md border border-border bg-background px-3 py-2 text-sm font-normal"
                              >
                                <option value="automatic">Automatic</option>
                                <option value="task-interpretation">Requirements</option>
                                <option value="model-selection">Model</option>
                                <option value="dataset-selection">Dataset</option>
                                <option value="choose-hyperparameters">Hyperparameters</option>
                              </select>
                            </label>
                            <label className="flex flex-col gap-1.5 text-xs font-medium text-foreground">
                              Required changes
                              <textarea
                                value={requiredChanges}
                                onChange={(event) => onRequiredChanges(event.target.value)}
                                placeholder="Must be applied exactly, for example: use YOLO11m and set batch size to 16."
                                rows={3}
                                className="resize-y rounded-md border border-border bg-background px-3 py-2 text-sm font-normal focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                              />
                            </label>
                            <label className="flex flex-col gap-1.5 text-xs font-medium text-foreground">
                              Preferences (optional)
                              <textarea
                                value={preferences}
                                onChange={(event) => onPreferences(event.target.value)}
                                placeholder="Applied when compatible, for example: prefer cosine scheduling."
                                rows={2}
                                className="resize-y rounded-md border border-border bg-background px-3 py-2 text-sm font-normal focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                              />
                            </label>

                            {revisionPlan && (
                              <div className="flex flex-col gap-2 rounded-md border border-primary/30 bg-primary/5 p-3 text-xs">
                                <span className="font-medium">{revisionPlan.summary}</span>
                                <span className="text-muted-foreground">
                                  Rerun from: {revisionPlan.restart_from}
                                </span>
                                <ul className="list-disc space-y-1 pl-5">
                                  {revisionPlan.changes.map((change) => (
                                    <li key={change.id} className="flex flex-wrap items-center gap-2">
                                      <select
                                        aria-label={`Priority for ${change.summary}`}
                                        value={change.strength}
                                        onChange={(event) => onRevisionStrength(
                                          change.id,
                                          event.target.value as "required" | "preferred",
                                        )}
                                        className="rounded border border-border bg-background px-1.5 py-1 capitalize"
                                      >
                                        <option value="required">Required</option>
                                        <option value="preferred">Preferred</option>
                                      </select>
                                      <span>{change.summary} <span className="text-muted-foreground">({change.target_step})</span></span>
                                    </li>
                                  ))}
                                </ul>
                                <div className="flex gap-2">
                                  <Button size="sm" onClick={onApplyRevision}>Apply revision</Button>
                                  <Button size="sm" variant="outline" onClick={onCancelRevision}>Cancel</Button>
                                </div>
                              </div>
                            )}

                            {revisionVerification && (
                              <div className={`rounded-md border p-3 text-xs ${revisionVerification.satisfied ? "border-emerald-500/40 bg-emerald-500/10" : "border-destructive/40 bg-destructive/10"}`}>
                                <span className="font-medium">
                                  {revisionVerification.satisfied
                                    ? "All required changes were verified."
                                    : "Some required changes were not satisfied."}
                                </span>
                                <ul className="mt-2 list-disc space-y-1 pl-5">
                                  {revisionVerification.checks.map((check) => (
                                    <li key={check.change_id}>
                                      {check.satisfied ? "Applied" : "Not applied"}: {check.summary}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            <div className="flex flex-col gap-2 sm:flex-row">
                              <Button onClick={onConfirm} disabled={revisionVerification?.satisfied === false}>
                                <Check className="size-4" aria-hidden />
                                {executionAvailable ? "Confirm and execute" : "Finish planning"}
                              </Button>
                              <Button
                                variant="outline"
                                onClick={onInterpret}
                                disabled={!!revisionPlan || (!requiredChanges.trim() && !preferences.trim())}
                                className="bg-transparent"
                              >
                                <X className="size-4" aria-hidden />
                                Interpret changes
                              </Button>
                            </div>
                          </>
                        )}
                        {status === "waiting" && !planningRevisionAvailable && (
                          <div className="flex flex-col gap-2">
                            <p className="text-xs leading-5 text-muted-foreground">
                              This hosted viewer can create and save the plan, but it cannot download data,
                              train models, evaluate models, or run inference.
                            </p>
                            <Button onClick={onConfirm}>
                              <Check className="size-4" aria-hidden /> Finish planning
                            </Button>
                          </div>
                        )}
                      </div>
                    )}
                    {step.id === "dataset-selection" && (
                      <DatasetSplitPlan context={context} />
                    )}
                  </PipelineStep>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
