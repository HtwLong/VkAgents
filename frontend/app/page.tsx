"use client"

import { useState } from "react"
import { Boxes, Play, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { TaskSelector } from "@/components/task-selector"
import { PromptInput } from "@/components/prompt-input"
import { ExamplePrompts } from "@/components/example-prompts"
import { PipelineView } from "@/components/pipeline-view"
import { PipelineOutputs } from "@/components/pipeline-outputs"
import { InferenceSection } from "@/components/inference-section"
import { FeedbackBar } from "@/components/feedback-bar"
import { usePipeline } from "@/hooks/use-pipeline"
import type { TaskType } from "@/lib/pipeline"

export default function Page() {
  const [task, setTask] = useState<TaskType>("automatic")
  const [prompt, setPrompt] = useState("")

  const {
    pipeline,
    status,
    activeStepId,
    start,
    reset,
    getStepStatus,
    getRevealed,
  } = usePipeline(task)

  const running = status === "running"
  const done = status === "done"

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-6 px-4 py-8 sm:px-6 lg:py-12">
      <header className="flex flex-col gap-3 border-b border-border pb-6">
        <div className="flex items-center gap-2.5">
          <span className="flex size-8 items-center justify-center rounded-md border border-primary/40 bg-primary/10 text-primary">
            <Boxes className="size-4.5" aria-hidden />
          </span>
          <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            v0 · vision-ops
          </span>
        </div>
        <h1 className="text-pretty text-2xl font-semibold tracking-tight sm:text-3xl">
          LLM-based Adaptive CV Model Learning Pipeline
        </h1>
        <p className="max-w-2xl text-pretty text-sm leading-relaxed text-muted-foreground">
          Describe the computer vision model you need in natural language. The
          pipeline plans the approach, trains a candidate model, evaluates it,
          and hands you a deployable model with a full report.
        </p>
      </header>

      {/* Configuration */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-4 lg:flex-row">
          <div className="flex flex-1 flex-col gap-4 sm:flex-row">
            <div className="sm:w-56">
              <TaskSelector value={task} onChange={setTask} disabled={running} />
            </div>
            <PromptInput value={prompt} onChange={setPrompt} disabled={running} />
          </div>
          <ExamplePrompts
            disabled={running}
            onUse={(p) => {
              setTask(p.task)
              setPrompt(p.text)
            }}
          />
        </div>

        <div className="flex items-center gap-3">
          {running ? (
            <Button variant="outline" onClick={reset} className="bg-transparent">
              <Square className="size-4" aria-hidden /> Stop run
            </Button>
          ) : (
            <Button onClick={start} disabled={!prompt.trim()}>
              <Play className="size-4" aria-hidden />
              {done ? "Run again" : "Start pipeline"}
            </Button>
          )}
          {!prompt.trim() && !running && (
            <span className="text-xs text-muted-foreground">
              Enter or drag in a prompt to begin.
            </span>
          )}
        </div>
      </section>

      {/* Pipeline */}
      <PipelineView
        pipeline={pipeline}
        status={status}
        activeStepId={activeStepId}
        getStepStatus={getStepStatus}
        getRevealed={getRevealed}
      />

      {/* Deliverables */}
      <PipelineOutputs ready={done} />

      {/* Inference */}
      <InferenceSection task={task} enabled={done} />

      {/* Feedback */}
      <FeedbackBar onRetry={start} />
    </main>
  )
}
