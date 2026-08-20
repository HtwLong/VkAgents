"use client"

import { BrainCircuit, CheckCircle2, RefreshCw, RotateCcw, TriangleAlert } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { AssessmentEligibility, PostTrainingAssessment } from "@/lib/pipeline"

export function PostTrainingAssessmentSection({
  assessment,
  eligibility,
  busy,
  onAnalyze,
  onRegenerate,
  onApprove,
  allowRevisionApproval,
}: {
  assessment: PostTrainingAssessment | null
  eligibility: AssessmentEligibility | null
  busy: boolean
  onAnalyze: () => void
  onRegenerate: () => void
  onApprove: () => void
  allowRevisionApproval: boolean
}) {
  if (!eligibility?.eligible) {
    return (
      <section className="surface-card rounded-2xl border border-white/80 bg-card p-4 sm:p-5">
        <h2 className="ui-card-title">Post-training assessment</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          {eligibility?.reason ?? "Complete the evaluation to assess this run."}
        </p>
      </section>
    )
  }

  if (!assessment) {
    return (
      <section className="surface-card rounded-2xl border border-white/80 bg-card p-4 sm:p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="ui-card-title">Post-training assessment</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Let the LLM compare the evaluation evidence with the original request.
            </p>
          </div>
          <Button onClick={onAnalyze} disabled={busy}>
            {busy ? <RefreshCw className="size-4 animate-spin" aria-hidden /> : <BrainCircuit className="size-4" aria-hidden />}
            Analyze results
          </Button>
        </div>
      </section>
    )
  }

  const met = assessment.verdict === "satisfied"
  return (
    <section className="surface-card rounded-2xl border border-white/80 bg-card p-4 sm:p-5">
      <div className="flex items-start gap-3">
        {met ? <CheckCircle2 className="mt-0.5 size-5 text-emerald-600" aria-hidden /> : <TriangleAlert className="mt-0.5 size-5 text-amber-600" aria-hidden />}
        <div className="min-w-0 flex-1">
          <h2 className="ui-card-title">Post-training assessment</h2>
          <p className="mt-1 text-sm font-medium capitalize">{assessment.verdict.replaceAll("_", " ")}</p>
          <p className="mt-1 text-sm text-muted-foreground">{assessment.summary}</p>
        </div>
      </div>

      {assessment.requirements.length > 0 && (
        <div className="mt-4 grid gap-2">
          {assessment.requirements.map((item, index) => (
            <div key={`${item.requirement}-${index}`} className="rounded-lg border border-border bg-background/60 p-3 text-sm">
              <div className="flex justify-between gap-3">
                <span className="font-medium">{item.requirement}</span>
                <span className="shrink-0 capitalize text-muted-foreground">{item.status.replaceAll("_", " ")}</span>
              </div>
              <p className="mt-1 text-muted-foreground">{item.explanation}</p>
              {item.evidence.length > 0 && <p className="mt-1 text-xs text-muted-foreground">Evidence: {item.evidence.join(" · ")}</p>}
            </div>
          ))}
        </div>
      )}

      {assessment.recommended_plan && (
        <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 p-3">
          <p className="text-sm font-medium">Recommended improvement</p>
          <p className="mt-1 text-sm text-muted-foreground">{assessment.recommended_plan.summary}</p>
          <p className="mt-2 text-xs text-muted-foreground">Restart planning at: {assessment.recommended_plan.restart_from}</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {assessment.recommended_plan.changes.map((change) => <li key={change.id}>{change.summary}</li>)}
          </ul>
          <div className="mt-3 flex flex-wrap gap-2">
            {allowRevisionApproval && (
              <Button onClick={onApprove} disabled={busy || !eligibility.can_create_revision}>
                <RotateCcw className="size-4" aria-hidden /> Approve and start improved run
              </Button>
            )}
            <Button variant="outline" className="bg-transparent" onClick={onRegenerate} disabled={busy}>
              <RefreshCw className={`size-4 ${busy ? "animate-spin" : ""}`} aria-hidden /> Redo recommendation
            </Button>
          </div>
          {!eligibility.can_create_revision && <p className="mt-2 text-xs text-amber-700">{eligibility.revision_reason}</p>}
          {!allowRevisionApproval && (
            <p className="mt-2 text-xs text-muted-foreground">
              The recommendation is informational in viewer mode; starting a training run is disabled.
            </p>
          )}
        </div>
      )}

      {!assessment.recommended_plan && (
        <Button variant="outline" className="mt-4 bg-transparent" onClick={onRegenerate} disabled={busy}>
          <RefreshCw className={`size-4 ${busy ? "animate-spin" : ""}`} aria-hidden /> Find improvement recommendation
        </Button>
      )}

      {assessment.limitations.length > 0 && <p className="mt-3 text-xs text-muted-foreground">Limitations: {assessment.limitations.join(" · ")}</p>}
    </section>
  )
}
