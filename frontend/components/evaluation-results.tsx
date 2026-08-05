"use client"

import { Activity, BarChart3, Database, Target } from "lucide-react"
import type { EvaluationReport } from "@/lib/pipeline"

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

export function EvaluationResults({ report }: { report: EvaluationReport | null }) {
  if (!report) return null
  const maxConfusion = Math.max(...report.confusion_matrix.flat(), 1)

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
                <tr><th className="p-3">Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test examples</th></tr>
              </thead>
              <tbody>
                {report.per_class.map((row) => (
                  <tr key={row.class_name} className="border-t">
                    <td className="p-3 font-medium">{row.class_name}</td>
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

      {!!report.confusion_matrix.length && report.confusion_matrix.length <= 12 && (
        <div>
          <h3 className="mb-1 text-sm font-semibold tracking-tight">Confusion matrix</h3>
          <p className="mb-3 text-xs text-muted-foreground">Rows are actual classes; columns are predictions.</p>
          <div className="overflow-x-auto">
            <div className="grid min-w-fit gap-1" style={{ gridTemplateColumns: `7rem repeat(${report.classes.length}, 3rem)` }}>
              <span />
              {report.classes.map((name) => <span key={name} className="truncate text-center text-[10px] text-muted-foreground" title={name}>{name}</span>)}
              {report.confusion_matrix.flatMap((row, rowIndex) => [
                <span key={`label-${rowIndex}`} className="truncate self-center text-xs" title={report.classes[rowIndex]}>{report.classes[rowIndex]}</span>,
                ...row.map((value, columnIndex) => (
                  <span key={`${rowIndex}-${columnIndex}`} className="flex size-12 items-center justify-center rounded text-xs tabular-nums" style={{ backgroundColor: `color-mix(in srgb, var(--primary) ${(value / maxConfusion) * 75}%, transparent)` }}>{value}</span>
                )),
              ])}
            </div>
          </div>
        </div>
      )}

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
    </section>
  )
}
