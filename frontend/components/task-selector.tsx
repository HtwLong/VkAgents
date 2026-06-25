"use client"

import { useEffect, useRef, useState } from "react"
import { Check, ChevronDown, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"
import { TASK_LABELS, type TaskType } from "@/lib/pipeline"

const ORDER: TaskType[] = ["automatic", "classification", "detection", "vqa"]

const DESCRIPTIONS: Record<TaskType, string> = {
  automatic: "Let the pipeline infer the best task",
  classification: "Assign a label to each image",
  detection: "Locate objects with bounding boxes",
  vqa: "Answer questions about an image",
}

export function TaskSelector({
  value,
  onChange,
  disabled,
}: {
  value: TaskType
  onChange: (t: TaskType) => void
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false)
    }
    document.addEventListener("mousedown", onClick)
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onClick)
      document.removeEventListener("keydown", onKey)
    }
  }, [open])

  return (
    <div ref={ref} className="relative">
      <span className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Task
      </span>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          "flex w-full min-w-56 items-center justify-between gap-3 rounded-md border border-border bg-card px-3 py-2.5 text-left text-sm transition-colors",
          "hover:border-primary/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          disabled && "cursor-not-allowed opacity-50",
        )}
      >
        <span className="flex items-center gap-2">
          {value === "automatic" && (
            <Sparkles className="size-4 text-primary" aria-hidden />
          )}
          <span className="font-medium">{TASK_LABELS[value]}</span>
        </span>
        <ChevronDown
          className={cn(
            "size-4 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      {open && (
        <ul
          role="listbox"
          className="absolute z-30 mt-2 w-full overflow-hidden rounded-md border border-border bg-popover p-1 shadow-xl"
        >
          {ORDER.map((task) => {
            const active = task === value
            return (
              <li key={task}>
                <button
                  type="button"
                  role="option"
                  aria-selected={active}
                  onClick={() => {
                    onChange(task)
                    setOpen(false)
                  }}
                  className={cn(
                    "flex w-full items-start gap-2 rounded-sm px-2.5 py-2 text-left transition-colors hover:bg-accent",
                    active && "bg-accent/60",
                  )}
                >
                  <Check
                    className={cn(
                      "mt-0.5 size-4 shrink-0 text-primary",
                      !active && "opacity-0",
                    )}
                    aria-hidden
                  />
                  <span className="flex flex-col">
                    <span className="flex items-center gap-1.5 text-sm font-medium text-popover-foreground">
                      {task === "automatic" && (
                        <Sparkles className="size-3.5 text-primary" aria-hidden />
                      )}
                      {TASK_LABELS[task]}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {DESCRIPTIONS[task]}
                    </span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
