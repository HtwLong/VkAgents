"use client"

import { useState } from "react"
import type { DragEvent } from "react"
import { cn } from "@/lib/utils"

export function PromptInput({
  value,
  onChange,
  disabled,
}: {
  value: string
  onChange: (v: string) => void
  disabled?: boolean
}) {
  const [dragOver, setDragOver] = useState(false)

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    if (disabled) return
    e.preventDefault()
    e.dataTransfer.dropEffect = "copy"
    setDragOver(true)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    if (disabled) return
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)

    const text = e.dataTransfer.getData("text/plain")
    if (text) {
      onChange(text)
    }
  }

  return (
    <div className="flex flex-1 flex-col">
      <label
        htmlFor="prompt"
        className="ui-subsection-title mb-1.5 block"
      >
        Prompt
      </label>
      <div
        className={cn(
          "relative flex-1 rounded-md border bg-card transition-colors",
          dragOver ? "border-primary ring-2 ring-primary/40" : "border-border",
        )}
        onDragOverCapture={handleDragOver}
        onDragLeave={() => setDragOver(false)}
        onDropCapture={handleDrop}
      >
        <textarea
          id="prompt"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Describe the computer vision model you want to build — the data, the goal, and any constraints. Or drag in an example prompt."
          className={cn(
            "h-[312] w-full resize-none rounded-md bg-transparent px-3.5 py-3 text-sm leading-relaxed text-foreground",
            "placeholder:text-muted-foreground/70 focus:outline-none",
            "overflow-y-auto",
            disabled && "cursor-not-allowed opacity-60",
          )}
        />
        {dragOver && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-md bg-primary/5 text-sm font-medium text-primary">
            Drop to replace prompt
          </div>
        )}
      </div>
    </div>
  )
}
