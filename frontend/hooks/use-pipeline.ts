"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  buildPipeline,
  type PipelineStage,
  type StepStatus,
  type TaskType,
} from "@/lib/pipeline"

export type RunStatus = "idle" | "running" | "done"

interface FlatStep {
  stageId: string
  stepId: string
  total: number
}

/** Drives a simulated, streaming run through the planning/execution/evaluation stages. */
export function usePipeline(task: TaskType) {
  const pipeline = useMemo<PipelineStage[]>(() => buildPipeline(task), [task])

  const flat = useMemo<FlatStep[]>(
    () =>
      pipeline.flatMap((stage) =>
        stage.steps.map((step) => ({
          stageId: stage.id,
          stepId: step.id,
          total: step.outputs.length,
        })),
      ),
    [pipeline],
  )

  const [status, setStatus] = useState<RunStatus>("idle")
  // index of the currently running step within `flat`; -1 when idle/finished
  const [cursor, setCursor] = useState(-1)
  // number of output lines revealed per step id
  const [revealed, setRevealed] = useState<Record<string, number>>({})

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearTimer = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  const reset = useCallback(() => {
    clearTimer()
    setStatus("idle")
    setCursor(-1)
    setRevealed({})
  }, [clearTimer])

  const start = useCallback(() => {
    clearTimer()
    setRevealed({})
    setStatus("running")
    setCursor(0)
  }, [clearTimer])

  // streaming engine
  useEffect(() => {
    if (status !== "running" || cursor < 0) return
    const step = flat[cursor]
    if (!step) {
      setStatus("done")
      setCursor(-1)
      return
    }

    const current = revealed[step.stepId] ?? 0
    if (current < step.total) {
      timer.current = setTimeout(
        () => {
          setRevealed((prev) => ({
            ...prev,
            [step.stepId]: (prev[step.stepId] ?? 0) + 1,
          }))
        },
        // first line lands quickly, subsequent lines stream a bit slower
        current === 0 ? 420 : 620,
      )
    } else {
      // step finished — advance
      timer.current = setTimeout(() => {
        setCursor((c) => c + 1)
      }, 300)
    }

    return clearTimer
  }, [status, cursor, revealed, flat, clearTimer])

  useEffect(() => clearTimer, [clearTimer])

  const getStepStatus = useCallback(
    (stepId: string): StepStatus => {
      const idx = flat.findIndex((f) => f.stepId === stepId)
      if (idx === -1) return "pending"
      if (status === "done") return "done"
      if (cursor === -1) return "pending"
      if (idx < cursor) return "done"
      if (idx === cursor) {
        const total = flat[idx].total
        return (revealed[stepId] ?? 0) >= total ? "done" : "running"
      }
      return "pending"
    },
    [flat, status, cursor, revealed],
  )

  const getRevealed = useCallback(
    (stepId: string) => revealed[stepId] ?? 0,
    [revealed],
  )

  const activeStepId = cursor >= 0 ? flat[cursor]?.stepId : undefined

  return {
    pipeline,
    status,
    activeStepId,
    start,
    reset,
    getStepStatus,
    getRevealed,
  }
}
