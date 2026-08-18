"use client"

import { useEffect, useState } from "react"
import { createPortal } from "react-dom"
import { Activity, BarChart3, Database, Maximize2, Target, X } from "lucide-react"
import Image from "next/image"
import type { EvaluationReport } from "@/lib/pipeline"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  loss: "Test loss",
  macro_precision: "Macro precision",
  macro_recall: "Macro recall",
  macro_f1: "Macro F1",
  micro_precision: "Micro precision",
  micro_recall: "Micro recall",
  micro_f1: "Micro F1",
  top5_acc: "Top-5 accuracy",
  map: "mAP@0.5:0.95",
  map50: "mAP@0.5",
  map75: "mAP@0.75",
  precision: "Precision",
  recall: "Recall",
}

function formatMetric(name: string, value: number) {
  return name === "loss" ? value.toFixed(4) : `${(value * 100).toFixed(1)}%`
}

function TrainingChart({ history }: { history: EvaluationReport["training_history"] }) {
  if (history.length < 2) return null
  const width = 600
  const height = 180
  const values = history.flatMap((row) => [row.train_loss, row.val_loss]).filter(Number.isFinite)
  if (!values.length) return null
  const max = Math.max(...values, 0.001)
  const points = (key: string) => history.map((row, index) => {
    const x = 12 + (index / (history.length - 1)) * (width - 24)
    const y = height - 12 - ((row[key] ?? 0) / max) * (height - 24)
    return `${x},${y}`
  }).join(" ")

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold tracking-tight">Training history</h3>
        <div className="flex gap-4 text-xs text-muted-foreground">
          <span><i className="mr-1.5 inline-block size-2 rounded-full bg-primary" />Train loss</span>
          <span><i className="mr-1.5 inline-block size-2 rounded-full bg-amber-500" />Validation loss</span>
        </div>
      </div>
      <div className="overflow-hidden rounded-lg border bg-muted/20 p-2">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-44 w-full" role="img" aria-label="Training and validation loss by epoch">
          <polyline points={points("train_loss")} fill="none" stroke="currentColor" strokeWidth="3" className="text-primary" />
          <polyline points={points("val_loss")} fill="none" stroke="currentColor" strokeWidth="3" className="text-amber-500" />
        </svg>
      </div>
    </div>
  )
}

function CurveChart({ title, curve }: { title: string; curve: { x: number[]; y: number[] } }) {
  if (curve.x.length < 2 || curve.x.length !== curve.y.length) return null
  const points = curve.x.map((x, index) => `${12 + x * 276},${108 - curve.y[index] * 96}`).join(" ")
  return <div className="rounded-lg border p-3">
    <h4 className="mb-2 text-xs font-medium">{title}</h4>
    <svg viewBox="0 0 300 120" className="h-36 w-full" role="img" aria-label={title}>
      <path d="M12 12V108H288" fill="none" stroke="currentColor" className="text-muted-foreground" />
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="3" className="text-primary" />
    </svg>
  </div>
}

export function EvaluationResults({ report }: { report: EvaluationReport | null }) {
  const [selectedVisualization, setSelectedVisualization] = useState<{ name: string; src: string } | null>(null)

  useEffect(() => {
    if (!selectedVisualization) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedVisualization(null)
    }
    window.addEventListener("keydown", closeOnEscape)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener("keydown", closeOnEscape)
    }
  }, [selectedVisualization])

  if (!report) return null

  return (
    <section className="surface-card flex flex-col gap-6 rounded-2xl border border-white/80 bg-white/82 p-4 sm:p-6">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
        <div>
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-primary">Evaluation results</span>
          <h2 className="mt-1 text-xl font-semibold tracking-tight">{report.model.name}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {report.task === "classification" ? "Image classification" : "Object detection"} · {report.classes.length} classes
            {report.model.training_mode ? ` · ${report.model.training_mode.replaceAll("_", " ")}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-lg border bg-primary/5 px-3 py-2 text-sm text-primary">
          <Target className="size-4" /> Evaluated on the held-out test split
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {Object.entries(report.metrics).map(([name, value], index) => {
          const Icon = index % 2 ? Activity : BarChart3
          return (
            <div key={name} className="rounded-xl border bg-card p-4">
              <Icon className="mb-3 size-4 text-primary" />
              <div className="text-2xl font-semibold tracking-tight tabular-nums">{formatMetric(name, value)}</div>
              <div className="mt-1 text-xs text-muted-foreground">{METRIC_LABELS[name] ?? name}</div>
            </div>
          )
        })}
      </div>

      <TrainingChart history={report.training_history} />

      {!!report.per_class.length && (
        <div>
          <h3 className="mb-3 text-sm font-semibold tracking-tight">Performance by class</h3>
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left text-xs text-muted-foreground">
                <tr><th className="p-3">Class</th>{report.task === "detection" && <><th>AP50–95</th><th>AP50</th></>}<th>Precision</th><th>Recall</th><th>F1</th><th>Instances</th></tr>
              </thead>
              <tbody>
                {report.per_class.map((row) => (
                  <tr key={row.class_name} className="border-t">
                    <td className="p-3 font-medium">{row.class_name}</td>
                    {report.task === "detection" && <><td>{row.ap == null ? "—" : formatMetric("map", row.ap)}</td><td>{row.ap50 == null ? "—" : formatMetric("map50", row.ap50)}</td></>}
                    <td>{row.precision == null ? "—" : formatMetric("precision", row.precision)}</td>
                    <td>{row.recall == null ? "—" : formatMetric("recall", row.recall)}</td>
                    <td>{row.f1 == null ? "—" : formatMetric("f1", row.f1)}</td>
                    <td>{row.support ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {report.curves && Object.keys(report.curves).length > 0 && <div>
        <h3 className="mb-3 text-sm font-semibold tracking-tight">Threshold analysis</h3>
        <div className="grid gap-3 md:grid-cols-2">
          {Object.entries(report.curves).map(([name, curve]) => <CurveChart key={name} title={name.replaceAll("_", " ")} curve={curve} />)}
        </div>
      </div>}

      {report.size_metrics && Object.keys(report.size_metrics).length > 0 && <div>
        <h3 className="mb-3 text-sm font-semibold tracking-tight">Performance by object size</h3>
        <div className="grid gap-3 sm:grid-cols-3">{Object.entries(report.size_metrics).map(([name, value]) =>
          <div key={name} className="rounded-lg border p-3"><div className="font-semibold">{formatMetric(name, value)}</div><div className="text-xs text-muted-foreground">{name.replaceAll("_", " ")}</div></div>
        )}</div>
      </div>}

      {!!report.visualizations?.length && <div>
        <h3 className="mb-3 text-sm font-semibold tracking-tight">Evaluation visualizations</h3>
        <div className="grid gap-4 md:grid-cols-2">{report.visualizations.map((item) => {
          const src = `${API_BASE}${item.url ?? `/${item.path}`}`
          return <button key={item.path} type="button" onClick={() => setSelectedVisualization({ name: item.name, src })} className="group overflow-hidden rounded-lg border bg-muted/20 text-left transition hover:border-primary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
            <span className="relative block"><Image src={src} alt={item.name} width={1200} height={800} className="h-auto w-full" unoptimized /><span className="absolute right-2 top-2 rounded-md bg-background/85 p-2 opacity-0 shadow-sm transition group-hover:opacity-100 group-focus-visible:opacity-100"><Maximize2 className="size-4" /></span></span>
            <span className="block p-2 text-xs text-muted-foreground">{item.name} · Click to enlarge</span>
          </button>
        })}</div>
      </div>}

      <div className="grid gap-4 border-t pt-5 md:grid-cols-2">
        <div>
          <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold tracking-tight"><Database className="size-4 text-primary" />Dataset splits</h3>
          <div className="flex gap-2">
            {Object.entries(report.dataset.splits).map(([split, count]) => (
              <div key={split} className="flex-1 rounded-lg bg-muted/40 p-3"><div className="text-lg font-semibold">{count}</div><div className="text-xs capitalize text-muted-foreground">{split}</div></div>
            ))}
          </div>
        </div>
        <div>
          <h3 className="mb-3 text-sm font-semibold tracking-tight">Configuration</h3>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
            {Object.entries(report.configuration).map(([key, value]) => (
              <div key={key} className="flex justify-between gap-2 border-b pb-1"><dt className="text-muted-foreground">{key.replaceAll("_", " ")}</dt><dd className="font-medium">{String(value)}</dd></div>
            ))}
          </dl>
        </div>
      </div>

      {selectedVisualization && createPortal(<div role="dialog" aria-modal="true" aria-label={selectedVisualization.name} className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-3 sm:p-8" onClick={() => setSelectedVisualization(null)}>
        <div className="relative flex max-h-full max-w-[96vw] flex-col overflow-hidden rounded-xl bg-background shadow-2xl" onClick={(event) => event.stopPropagation()}>
          <div className="flex items-center justify-between border-b px-4 py-3"><h3 className="font-semibold">{selectedVisualization.name}</h3><button type="button" onClick={() => setSelectedVisualization(null)} aria-label="Close enlarged image" className="rounded-md p-2 hover:bg-muted"><X className="size-5" /></button></div>
          <div className="overflow-auto p-2"><Image src={selectedVisualization.src} alt={selectedVisualization.name} width={2400} height={1600} className="h-auto max-h-[82vh] w-auto max-w-none" unoptimized priority /></div>
        </div>
      </div>, document.body)}
    </section>
  )
}
