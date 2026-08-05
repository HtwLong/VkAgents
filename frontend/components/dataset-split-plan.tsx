"use client"

type Allocation = {
  split: "train" | "validation" | "test"
  count: number
  assignment_type: "official_split" | "derived_from_train"
}

type Assignment = {
  class_name: string
  sources: Array<{ dataset_name: string; allocations: Allocation[] }>
}

const SPLITS: Allocation["split"][] = ["train", "validation", "test"]

function readAssignments(context: unknown): Assignment[] {
  if (!context || typeof context !== "object" || !("selected_data" in context)) return []
  const selectedData = (context as { selected_data?: unknown }).selected_data
  if (!Array.isArray(selectedData)) return []
  return selectedData.filter((item): item is Assignment => (
    !!item && typeof item === "object" && typeof item.class_name === "string" && Array.isArray(item.sources)
  ))
}

export function DatasetSplitPlan({ context }: { context: unknown }) {
  const assignments = readAssignments(context)
  if (!assignments.length) return null

  const totals = Object.fromEntries(SPLITS.map((split) => [split, 0])) as Record<Allocation["split"], number>
  const official = Object.fromEntries(SPLITS.map((split) => [split, 0])) as Record<Allocation["split"], number>
  const derived = Object.fromEntries(SPLITS.map((split) => [split, 0])) as Record<Allocation["split"], number>
  for (const assignment of assignments) {
    for (const source of assignment.sources) {
      for (const allocation of source.allocations ?? []) {
        if (!SPLITS.includes(allocation.split) || !Number.isFinite(allocation.count)) continue
        totals[allocation.split] += allocation.count
        const target = allocation.assignment_type === "derived_from_train" ? derived : official
        target[allocation.split] += allocation.count
      }
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-md border border-border bg-card p-3">
      <div>
        <p className="ui-section-label">
          Planned split assignments
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Official validation and test samples stay in their source split. Derived holdouts come only from training sources.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {SPLITS.map((split) => (
          <div key={split} className="rounded-md border border-border/70 bg-background p-2">
            <p className="text-xs font-medium capitalize text-foreground">{split}</p>
            <p className="font-mono text-sm text-primary">{totals[split]}</p>
            <p className="text-[11px] text-muted-foreground">
              {official[split]} official · {derived[split]} derived
            </p>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded-md border border-border/70">
        <table className="w-full min-w-[34rem] text-left text-xs">
          <thead className="bg-muted/60 text-muted-foreground">
            <tr>
              <th className="px-2.5 py-2 font-medium">Class</th>
              <th className="px-2.5 py-2 font-medium">Dataset source</th>
              <th className="px-2.5 py-2 font-medium">Assignments</th>
            </tr>
          </thead>
          <tbody>
            {assignments.flatMap((assignment) => assignment.sources.map((source) => (
              <tr key={`${assignment.class_name}:${source.dataset_name}`} className="border-t border-border/60">
                <td className="px-2.5 py-2 font-medium text-foreground">{assignment.class_name}</td>
                <td className="px-2.5 py-2 font-mono text-muted-foreground">{source.dataset_name}</td>
                <td className="px-2.5 py-2 text-muted-foreground">
                  {(source.allocations ?? []).map((allocation) => (
                    `${allocation.split}: ${allocation.count} (${allocation.assignment_type === "official_split" ? "official" : "derived"})`
                  )).join(" · ")}
                </td>
              </tr>
            ))) }
          </tbody>
        </table>
      </div>
    </div>
  )
}
