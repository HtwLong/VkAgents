"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { Check, ChevronRight, Loader2, X } from "lucide-react"
import { cn } from "@/lib/utils"
import type { PipelineStep as Step, StepStatus } from "@/lib/pipeline"

function tryParseJson(value: string) {
  try {
    return JSON.parse(value)
  } catch {
    return null
  }
}

function splitStructuredOutput(line: string) {
  const separatorIndex = line.indexOf(":\n")
  if (separatorIndex === -1) {
    const parsed = tryParseJson(line.trim())
    return parsed && typeof parsed === "object"
      ? { label: null, json: parsed }
      : { label: null, text: line }
  }

  const label = line.slice(0, separatorIndex)
  const body = line.slice(separatorIndex + 2).trim()
  const parsed = tryParseJson(body)
  return parsed && typeof parsed === "object"
    ? { label, json: parsed }
    : { label, text: body }
}

function formatDuration(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
  }

  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
}

function StepOutput({ line }: { line: string }) {
  const output = splitStructuredOutput(line)

  return (
    <li className="font-mono text-xs leading-5 text-muted-foreground">
      <span className="mr-2 select-none text-primary/60">›</span>
      {"json" in output ? (
        <span className="inline-flex w-[calc(100%-1.25rem)] flex-col gap-1 align-top">
          {output.label && (
            <span className="font-sans text-xs font-medium text-foreground">
              {output.label}
            </span>
          )}
          <pre className="ui-code-block overflow-x-auto rounded-md border border-border/70 bg-muted/35 p-3">
            {JSON.stringify(output.json, null, 2)}
          </pre>
        </span>
      ) : output.label ? (
        <span className="whitespace-pre-wrap">
          <span className="font-sans font-medium text-foreground">
            {output.label}:
          </span>
          {"\n"}
          {output.text}
        </span>
      ) : (
        <span className="whitespace-pre-wrap">{output.text}</span>
      )}
    </li>
  )
}

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
  if (status === "failed") {
    return (
      <span className="flex size-4 items-center justify-center rounded-full bg-destructive text-destructive-foreground">
        <X className="size-3" aria-hidden />
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
  durationMs,
  onToggle,
  children,
  childrenBeforeOutputs = false,
}: {
  step: Step
  status: StepStatus
  expanded: boolean
  revealedCount: number
  durationMs?: number
  onToggle: () => void
  children?: ReactNode
  childrenBeforeOutputs?: boolean
}) {
  const lines = step.outputs.slice(0, status === "done" ? step.outputs.length : revealedCount)
  const [elapsedMs, setElapsedMs] = useState(durationMs ?? 0)
  const startedAtRef = useRef<number | null>(null)

  useEffect(() => {
    if (status === "running") {
      startedAtRef.current = Date.now()
      window.setTimeout(() => setElapsedMs(0), 0)

      const interval = window.setInterval(() => {
        if (startedAtRef.current !== null) {
          setElapsedMs(Date.now() - startedAtRef.current)
        }
      }, 1000)

      return () => window.clearInterval(interval)
    }

    if (startedAtRef.current !== null) {
      setElapsedMs(durationMs ?? Date.now() - startedAtRef.current)
      startedAtRef.current = null
      return
    }

    if (status === "pending") {
      window.setTimeout(() => setElapsedMs(0), 0)
    } else if (durationMs !== undefined) {
      window.setTimeout(() => setElapsedMs(durationMs), 0)
    }
  }, [durationMs, status])

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
            "flex-1 text-sm font-semibold tracking-tight",
            status === "pending" ? "text-muted-foreground" : "text-foreground",
            status === "failed" && "text-destructive",
          )}
        >
          {step.title}
        </span>
        <div className="flex items-center gap-1.5">
          {status !== "pending" && (
            <span className="rounded-sm border border-border/70 bg-muted/70 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
              {formatDuration(elapsedMs)}
            </span>
          )}
          {status === "running" && (
            <span className="rounded-sm bg-primary/15 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-primary">
              running
            </span>
          )}
          {status === "failed" && (
            <span className="rounded-sm bg-destructive/15 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-destructive">
              failed
            </span>
          )}
        </div>
      </button>

      {expanded && (
        <div className="max-h-[28rem] overflow-y-auto border-t border-border/60 px-3 pb-3 pt-2">
          {childrenBeforeOutputs && children && (
            <div className="ml-6 mb-3">{children}</div>
          )}
          {lines.length === 0 ? (
            <p className="ml-6 font-mono text-xs text-muted-foreground/70">
              {status === "pending" ? "Waiting for previous steps…" : "Starting…"}
            </p>
          ) : (
            <ul className="ml-6 flex flex-col gap-1.5 border-l border-border/60 pl-3">
              {lines.map((line, i) => (
                <StepOutput key={i} line={line} />
              ))}
              {status === "running" && lines.length < step.outputs.length && (
                <li className="ml-1 inline-block h-3 w-2 animate-pulse bg-primary/70" aria-hidden />
              )}
            </ul>
          )}
          {!childrenBeforeOutputs && children && <div className="ml-6 mt-3">{children}</div>}
        </div>
      )}
    </div>
  )
}
