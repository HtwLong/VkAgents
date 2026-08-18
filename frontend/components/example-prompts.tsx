"use client"

import { GripVertical } from "lucide-react"
import { cn } from "@/lib/utils"
import { DOMAIN_LABELS, EXAMPLE_PROMPTS, TASK_LABELS, type ExamplePrompt } from "@/lib/pipeline"

export function ExamplePrompts({
  onUse,
  disabled,
}: {
  onUse: (prompt: ExamplePrompt) => void
  disabled?: boolean
}) {
  return (
    <div className="flex w-full flex-col">
      <span className="ui-subsection-title mb-1.5 block">
        Example prompts
      </span>
      <div className="flex max-h-[17.5rem] flex-col gap-2 overflow-y-auto rounded-md border border-border bg-card/50 p-2 lg:max-h-none lg:h-44">
        {EXAMPLE_PROMPTS.map((p) => (
          <button
            key={p.id}
            type="button"
            draggable={!disabled}
            onDragStart={(e) => {
              e.dataTransfer.setData("text/plain", p.text)
              e.dataTransfer.effectAllowed = "copy"
            }}
            onClick={() => !disabled && onUse(p)}
            className={cn(
              "group flex cursor-grab items-start gap-2 rounded-md border border-border/60 bg-background/60 p-2.5 text-left transition-colors",
              "hover:border-primary/50 hover:bg-accent/40 active:cursor-grabbing",
              disabled && "cursor-not-allowed opacity-50",
            )}
            title="Drag into the prompt box or click to use"
          >
            <GripVertical
              className="mt-0.5 size-4 shrink-0 text-muted-foreground group-hover:text-primary"
              aria-hidden
            />
            <span className="flex flex-col gap-1">
              <span className="text-sm font-medium text-foreground">
                {p.name}
              </span>
              <span className="flex flex-wrap gap-1">
                <span className="w-fit rounded-sm border border-sky-500/30 bg-sky-500/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sky-700 dark:text-sky-300">
                  Task: {TASK_LABELS[p.task]}
                </span>
                <span className="w-fit rounded-sm bg-primary/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary">
                  {DOMAIN_LABELS[p.domain]}
                </span>
              </span>
              <span className="ui-caption line-clamp-3">
                {p.text}
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
