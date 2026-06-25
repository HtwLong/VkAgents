"use client"

import { useEffect, useState } from "react"
import { Activity } from "lucide-react"
import { cn } from "@/lib/utils"
import type { PipelineStage, StepStatus } from "@/lib/pipeline"
import type { RunStatus } from "@/hooks/use-pipeline"
import { PipelineStep } from "@/components/pipeline-step"

const STAGE_INDEX = ["01", "02", "03"]

export function PipelineView({
  pipeline,
  status,
  activeStepId,
  getStepStatus,
  getRevealed,
}: {
  pipeline: PipelineStage[]
  status: RunStatus
  activeStepId?: string
  getStepStatus: (id: string) => StepStatus
  getRevealed: (id: string) => number
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  // auto-expand the active step as the run progresses
  useEffect(() => {
    if (activeStepId) {
      setExpanded((prev) => new Set(prev).add(activeStepId))
    }
  }, [activeStepId])

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <section className="rounded-lg border border-border bg-card">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-primary" aria-hidden />
          <h2 className="text-sm font-semibold">Pipeline updates</h2>
        </div>
        <span
          className={cn(
            "flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] uppercase tracking-wide",
            status === "running" && "border-primary/40 text-primary",
            status === "done" && "border-primary/40 text-primary",
            status === "idle" && "border-border text-muted-foreground",
          )}
        >
          <span
            className={cn(
              "size-1.5 rounded-full",
              status === "running" && "animate-pulse bg-primary",
              status === "done" && "bg-primary",
              status === "idle" && "bg-muted-foreground",
            )}
            aria-hidden
          />
          {status === "running" ? "running" : status === "done" ? "complete" : "idle"}
        </span>
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
                  <h3 className="text-sm font-semibold leading-tight">
                    {stage.title}
                  </h3>
                  <p className="text-xs text-muted-foreground">
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
                    expanded={expanded.has(step.id)}
                    revealedCount={getRevealed(step.id)}
                    onToggle={() => toggle(step.id)}
                  />
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
