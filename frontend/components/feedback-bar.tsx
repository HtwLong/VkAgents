"use client"

import { useState } from "react"
import { Check, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"

export function FeedbackBar({ onRetry }: { onRetry: () => void }) {
  const [feedback, setFeedback] = useState("")
  const [sent, setSent] = useState(false)

  const submit = () => {
    if (!feedback.trim()) return
    // simulated submission to the backend
    console.log("[v0] feedback submitted:", feedback)
    setSent(true)
    setFeedback("")
    setTimeout(() => setSent(false), 2500)
  }

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <div className="flex flex-1 flex-col gap-2">
          <label
            htmlFor="feedback"
            className="text-xs font-medium uppercase tracking-wider text-muted-foreground"
          >
            Feedback to the pipeline
          </label>
          <div className="flex gap-2">
            <input
              id="feedback"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="e.g. Prioritize recall over precision, or add more augmentation."
              className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            <Button onClick={submit} disabled={!feedback.trim()} className="shrink-0">
              {sent ? (
                <>
                  <Check className="size-4" aria-hidden /> Sent
                </>
              ) : (
                "Send feedback"
              )}
            </Button>
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:pt-[1.625rem]">
          <Button
            variant="outline"
            onClick={onRetry}
            className="bg-transparent"
          >
            <RotateCcw className="size-4" aria-hidden /> Retry run
          </Button>
        </div>
      </div>
    </section>
  )
}
