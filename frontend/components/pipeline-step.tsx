"use client"

import { Check, ChevronRight, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import type { PipelineStep as Step, StepStatus } from "@/lib/pipeline"

function StatusDot({ status }: { status: StepStatus }) {
  if (status === "running") {
    return <Loader2 className="size-4 animate-spin text-primary" aria-hidden />
  }
  if (status === "done") {
    return (
      <span className="flex size-4 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Check className="size-3" aria-hidden />
      </span>
    )
  }
  return (
    <span
      className="size-4 rounded-full border border-border bg-transparent"
      aria-hidden
    />
  )
}

export function PipelineStep({
  step,
  status,
  expanded,
  revealedCount,
  onToggle,
}: {
  step: Step
  status: StepStatus
  expanded: boolean
  revealedCount: number
  onToggle: () => void
}) {
  const lines = step.outputs.slice(0, status === "done" ? step.outputs.length : revealedCount)

  return (
    <div className="rounded-md border border-border/70 bg-background/40">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left"
      >
        <ChevronRight
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform",
            expanded && "rotate-90",
          )}
          aria-hidden
        />
        <StatusDot status={status} />
        <span
          className={cn(
            "flex-1 text-sm font-medium",
            status === "pending" ? "text-muted-foreground" : "text-foreground",
          )}
        >
          {step.title}
        </span>
        {status === "running" && (
          <span className="rounded-sm bg-primary/15 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-primary">
            running
          </span>
        )}
      </button>

      {expanded && (
        <div className="border-t border-border/60 px-3 pb-3 pt-2">
          {lines.length === 0 ? (
            <p className="ml-6 font-mono text-xs text-muted-foreground/70">
              {status === "pending" ? "Waiting for previous steps…" : "Starting…"}
            </p>
          ) : (
            <ul className="ml-6 flex flex-col gap-1.5 border-l border-border/60 pl-3">
              {lines.map((line, i) => (
                <li
                  key={i}
                  className="font-mono text-xs leading-relaxed text-muted-foreground"
                >
                  <span className="mr-2 select-none text-primary/60">›</span>
                  {line}
                </li>
              ))}
              {status === "running" && lines.length < step.outputs.length && (
                <li className="ml-1 inline-block h-3 w-2 animate-pulse bg-primary/70" aria-hidden />
              )}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
