"use client"

import { Clock3, Coins, Cpu, Gauge, MessageSquareText } from "lucide-react"
import type { PlanningLLMUsage, PlanningLLMUsageBucket } from "@/lib/pipeline"

const ROWS = [
  { id: "task-interpretation", label: "Task interpretation", operations: ["completeness_check", "task_interpretation", "synonym_check"] },
  { id: "check-data", label: "Data check", operations: [] },
  { id: "model-selection", label: "Model selection", operations: ["model_selection"] },
  { id: "dataset-selection", label: "Dataset selection", operations: ["dataset_selection"] },
  { id: "choose-hyperparameters", label: "Hyperparameter planning", operations: ["hpo_optimizer", "hpo_evaluator"] },
] as const

function emptyUsage(): PlanningLLMUsageBucket {
  return { requests: 0, input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 0, calculated_cost_usd: "0" }
}

function combine(usage: PlanningLLMUsage, operations: readonly string[]) {
  const result = emptyUsage()
  let cost = 0
  for (const operation of operations) {
    const bucket = usage.operations[operation]
    if (!bucket) continue
    result.requests += bucket.requests
    result.input_tokens += bucket.input_tokens
    result.cached_input_tokens += bucket.cached_input_tokens
    result.output_tokens += bucket.output_tokens
    result.reasoning_tokens += bucket.reasoning_tokens
    result.total_tokens += bucket.total_tokens
    if (bucket.calculated_cost_usd == null) result.calculated_cost_usd = null
    else cost += Number(bucket.calculated_cost_usd)
  }
  if (result.calculated_cost_usd !== null) result.calculated_cost_usd = cost.toFixed(8)
  return result
}

function duration(value?: number) {
  if (value == null) return "—"
  if (value < 1000) return `${value} ms`
  return `${(value / 1000).toFixed(value < 10_000 ? 2 : 1)} s`
}

function cost(value: string | null) {
  if (value == null) return "Unavailable"
  const amount = Number(value)
  return `$${amount.toFixed(amount < 0.01 ? 5 : 3)}`
}

export function PlanningPerformance({
  usage,
  getStepDuration,
}: {
  usage: PlanningLLMUsage | null
  getStepDuration: (id: string) => number | undefined
}) {
  if (!usage) return null
  const models = Object.keys(usage.models)
  const totalDuration = ROWS.reduce((sum, row) => sum + (getStepDuration(row.id) ?? 0), 0)
  const revisionUsage = usage.operations.planning_revision

  return <section className="surface-card rounded-2xl border border-white/80 bg-white/82 p-4 sm:p-6">
    <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
      <div><span className="text-xs font-semibold uppercase tracking-[0.12em] text-primary">Planning performance</span><h2 className="mt-1 text-xl font-semibold tracking-tight">Time and LLM consumption</h2></div>
    </div>

    <div className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <div className="rounded-xl border bg-card p-4"><Clock3 className="mb-3 size-4 text-primary" /><div className="text-xl font-semibold tabular-nums">{duration(totalDuration)}</div><div className="mt-1 text-xs text-muted-foreground">Total planning time</div></div>
      <div className="rounded-xl border bg-card p-4"><Coins className="mb-3 size-4 text-primary" /><div className="text-xl font-semibold tabular-nums">{cost(usage.totals.calculated_cost_usd)}</div><div className="mt-1 text-xs text-muted-foreground">Calculated planning cost</div></div>
      <div className="rounded-xl border bg-card p-4"><MessageSquareText className="mb-3 size-4 text-primary" /><div className="text-xl font-semibold tabular-nums">{usage.totals.total_tokens.toLocaleString()}</div><div className="mt-1 text-xs text-muted-foreground">Total tokens · {usage.totals.requests} requests</div></div>
      <div className="rounded-xl border bg-card p-4"><Cpu className="mb-3 size-4 text-primary" /><div className="text-base font-semibold">{models.join(", ") || "Unknown"}</div><div className="mt-1 text-xs text-muted-foreground">Planning LLM model</div></div>
    </div>

    <div className="overflow-x-auto rounded-xl border">
      <table className="w-full min-w-[780px] text-sm">
        <thead className="bg-muted/50 text-left text-xs text-muted-foreground"><tr><th className="p-3">Planning step</th><th>Time</th><th>Requests</th><th>Input tokens</th><th>Output tokens</th><th>Cost</th><th>Model</th></tr></thead>
        <tbody>{ROWS.map((row) => {
          const bucket = combine(usage, row.operations)
          const hasLLM = row.operations.length > 0
          return <tr key={row.id} className="border-t"><td className="p-3 font-medium">{row.label}</td><td className="tabular-nums">{duration(getStepDuration(row.id))}</td><td className="tabular-nums">{bucket.requests}</td><td className="tabular-nums">{bucket.input_tokens.toLocaleString()}</td><td className="tabular-nums">{bucket.output_tokens.toLocaleString()}</td><td className="tabular-nums">{cost(bucket.calculated_cost_usd)}</td><td>{hasLLM ? models.join(", ") : "No LLM"}</td></tr>
        })}</tbody>
      </table>
    </div>

    {revisionUsage && <p className="mt-3 text-xs text-muted-foreground">Planning revision requests are included in the total: {revisionUsage.requests} request(s), {revisionUsage.total_tokens.toLocaleString()} tokens, {cost(revisionUsage.calculated_cost_usd)}.</p>}
    {usage.usage_notes.length > 0 && <p className="mt-2 text-xs text-muted-foreground">{usage.usage_notes.join(" ")}</p>}
  </section>
}
