"use client"

import { Calculator, FileCode2, GitBranch, ListChecks, ShieldCheck, Sparkles } from "lucide-react"

export interface RetrievedFact {
  id: string
  type: string
  statement: string
  evidence_ids: string[]
  support_type: string
  confidence?: string | null
  derivation?: { method?: string; inputs?: Record<string, unknown> } | null
}

export interface EvidenceSource {
  id: string
  title: string
  url?: string | null
  reference?: string | null
  source_type?: string | null
  source_owner?: string | null
  year?: string | number | null
  claim_supported?: string | null
  locator_type?: "external_url" | "repository_path"
}

export interface DecisionEvidence {
  decision_type: "model_selection" | "dataset_selection" | "hyperparameter_selection"
  decision: unknown
  rationale: string
  retrieved_facts: RetrievedFact[]
  evidence_sources: EvidenceSource[]
  grounded: boolean
  evidence_backed?: boolean
  match_scope?: "exact" | "family_variant" | null
  grounding?: {
    status: string
    fact_count: number
    support_counts: Record<string, number>
    evidence_coverage: number
  }
  field_provenance?: Record<string, {
    source: string
    source_id: string
    reason: string
    support_type: string
    evidence_ids: string[]
    applied_policy_ids?: string[]
  }>
  policy_guidance?: {
    used_policies: Array<{
      id: string
      fields: string[]
      influenced_fields: string[]
      guidance: string
      evidence_inputs: string[]
      authority: "advisory" | string
    }>
    profile: Record<string, unknown>
  }
}

const SUPPORT_LABELS: Record<string, string> = {
  direct_evidence: "Directly supported",
  derived: "Calculated",
  inferred: "Inferred",
  heuristic: "Heuristic",
  user_constraint: "User constraint",
  system_policy: "System policy",
  schema_default: "Schema default",
  llm_judgment: "LLM judgment",
  internal_assertion: "Internal source",
}

const HIDDEN_EVIDENCE_SOURCE_TYPES = new Set(["LocalCode", "CourseNotes"])

function SupportIcon({ type }: { type: string }) {
  if (type === "derived") return <Calculator className="size-3" aria-hidden />
  if (type === "inferred") return <GitBranch className="size-3" aria-hidden />
  if (type === "llm_judgment" || type === "heuristic") return <Sparkles className="size-3" aria-hidden />
  return <ShieldCheck className="size-3" aria-hidden />
}

function SourceBadge({ source, number }: { source: EvidenceSource; number: number }) {
  const badge = (
    <span
      className="inline-flex size-6 shrink-0 items-center justify-center rounded-full border border-primary/50 bg-primary/10 font-mono text-[11px] font-semibold text-primary transition-colors group-hover:bg-primary group-hover:text-primary-foreground"
      aria-hidden
    >
      {number}
    </span>
  )

  if (!source.url) {
    return (
      <span
        className="inline-flex size-6 items-center justify-center rounded-md border border-border bg-muted text-muted-foreground"
        title={`Internal reference: ${source.reference || source.title}`}
        aria-label={`Internal source: ${source.title}`}
      >
        <FileCode2 className="size-3.5" aria-hidden />
      </span>
    )
  }
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group inline-flex"
      aria-label={`Open source ${number}: ${source.title} in a new tab`}
      title={source.title}
    >
      {badge}
    </a>
  )
}

export function DecisionEvidencePanel({ evidence }: { evidence: DecisionEvidence }) {
  const visibleEvidenceSources = evidence.evidence_sources.filter(
    (source) => !source.source_type || !HIDDEN_EVIDENCE_SOURCE_TYPES.has(source.source_type),
  )
  const visibleEvidenceSourceIds = new Set(visibleEvidenceSources.map((source) => source.id))
  const visibleRetrievedFacts = evidence.retrieved_facts.filter((fact) =>
    fact.evidence_ids.some((id) => visibleEvidenceSourceIds.has(id)),
  )

  const sourceNumber = new Map<string, number>()
  let nextSourceNumber = 1
  for (const source of visibleEvidenceSources) {
    if (source.url) sourceNumber.set(source.id, nextSourceNumber++)
  }

  return (
    <div className="flex flex-col gap-4 rounded-md border border-primary/20 bg-primary/[0.03] p-3">
      <div className="flex flex-col gap-1">
        <span className="ui-section-label">
          Decision
        </span>
        <p className="text-sm leading-relaxed text-foreground">
          {evidence.rationale || "No rationale was returned."}
        </p>
        {evidence.grounding && (
          <p className="text-xs text-muted-foreground">
            {evidence.grounding.status.replaceAll("_", " ")} · {evidence.grounding.fact_count} facts · {Math.round(evidence.grounding.evidence_coverage * 100)}% source coverage
          </p>
        )}
        <details className="mt-1 rounded-md border border-border/70 bg-background/70">
          <summary className="cursor-pointer px-2.5 py-2 text-xs font-medium text-foreground">
            Selected decision
          </summary>
          <pre className="ui-code-block max-h-64 overflow-auto border-t border-border/70 p-2.5">
            {JSON.stringify(evidence.decision, null, 2)}
          </pre>
        </details>
      </div>

      {!!visibleRetrievedFacts.length && (
        <div className="flex flex-col gap-2">
          <span className="ui-section-label">
            Retrieved graph facts
          </span>
          {evidence.match_scope === "family_variant" && (
            <p className="text-xs leading-relaxed text-muted-foreground">
              Evidence comes from a retrieved variant in the selected model family. Variant-specific
              benchmark and memory values are shown as contextual evidence, not as exact values for every variant.
            </p>
          )}
          <ul className="flex flex-col gap-2">
            {visibleRetrievedFacts.map((fact) => (
              <li key={fact.id} className="rounded-md border border-border/70 bg-background/70 p-2.5">
                <span className="mb-1.5 inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                  <SupportIcon type={fact.support_type} />
                  {SUPPORT_LABELS[fact.support_type] || fact.support_type}
                </span>
                <div className="flex items-start justify-between gap-3">
                  <p className="text-xs leading-relaxed text-foreground">{fact.statement}</p>
                  <div className="flex shrink-0 flex-wrap gap-1">
                    {fact.evidence_ids.map((id) => {
                      const number = sourceNumber.get(id)
                      const source = visibleEvidenceSources.find((item) => item.id === id)
                      return source ? (
                        <SourceBadge key={id} source={source} number={number ?? 0} />
                      ) : null
                    })}
                  </div>
                </div>
                <span className="mt-1 block font-mono text-[11px] text-muted-foreground">
                  {fact.type} · {fact.id}
                </span>
                {fact.derivation?.method && (
                  <details className="mt-1 text-[11px] text-muted-foreground">
                    <summary className="cursor-pointer">Calculation details</summary>
                    <p className="mt-1 whitespace-pre-wrap">{fact.derivation.method}</p>
                  </details>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {!!visibleEvidenceSources.length && (
        <div className="flex flex-col gap-2">
          <span className="ui-section-label">
            Evidence sources
          </span>
          <ol className="flex flex-col gap-2">
            {visibleEvidenceSources.map((source) => (
              <li key={source.id} className="flex items-start gap-2.5">
                <SourceBadge source={source} number={sourceNumber.get(source.id) ?? 0} />
                <div className="min-w-0 text-xs leading-relaxed">
                  <p className="font-medium text-foreground">{source.title}</p>
                  <p className="text-muted-foreground">
                    {[source.source_owner, source.year, source.source_type].filter(Boolean).join(" · ")}
                  </p>
                  {source.claim_supported && (
                    <p className="mt-0.5 text-muted-foreground">{source.claim_supported}</p>
                  )}
                  {!source.url && source.reference && (
                    <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                      Internal reference: {source.reference}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}

      {!!evidence.policy_guidance?.used_policies.length && (
        <div className="flex flex-col gap-2 rounded-md border border-amber-500/25 bg-amber-500/[0.04] p-3">
          <div className="flex items-start gap-2">
            <ListChecks className="mt-0.5 size-4 shrink-0 text-amber-700" aria-hidden />
            <div>
              <span className="ui-section-label">Policies used</span>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Advisory decision rules that guided LLM-selected values. Policies are not retrieved
                facts and cannot override recipe, runtime, or schema constraints.
              </p>
            </div>
          </div>
          <ul className="flex flex-col gap-2">
            {evidence.policy_guidance.used_policies.map((policy) => (
              <li key={policy.id} className="rounded-md border border-border/70 bg-background/80 p-2.5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="font-mono text-xs font-semibold text-foreground">{policy.id}</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                      {policy.guidance}
                    </p>
                  </div>
                  <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-800">
                    {policy.authority}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {policy.influenced_fields.map((field) => (
                    <span key={field} className="rounded-sm border border-border bg-muted/60 px-1.5 py-0.5 font-mono text-[10px] text-foreground">
                      {field}
                    </span>
                  ))}
                </div>
                {!!policy.evidence_inputs.length && (
                  <details className="mt-2 text-[11px] text-muted-foreground">
                    <summary className="cursor-pointer">Inputs considered</summary>
                    <p className="mt-1 font-mono">{policy.evidence_inputs.join(" · ")}</p>
                  </details>
                )}
              </li>
            ))}
          </ul>
          {Object.keys(evidence.policy_guidance.profile).length > 0 && (
            <details className="rounded-md border border-border/70 bg-background/80">
              <summary className="cursor-pointer px-2.5 py-2 text-xs font-medium text-foreground">
                Derived training profile used by policies
              </summary>
              <pre className="ui-code-block max-h-64 overflow-auto border-t border-border/70 p-2.5">
                {JSON.stringify(evidence.policy_guidance.profile, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      {evidence.field_provenance && Object.keys(evidence.field_provenance).length > 0 && (
        <details className="rounded-md border border-border/70 bg-background/70">
          <summary className="cursor-pointer px-2.5 py-2 text-xs font-medium text-foreground">
            Hyperparameter field provenance
          </summary>
          <ul className="max-h-80 overflow-auto border-t border-border/70">
            {Object.entries(evidence.field_provenance).map(([field, provenance]) => (
              <li key={field} className="border-b border-border/50 p-2.5 last:border-b-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-mono text-xs font-medium text-foreground">{field}</p>
                    <p className="text-xs leading-relaxed text-muted-foreground">{provenance.reason}</p>
                    {!!provenance.applied_policy_ids?.length && (
                      <p className="mt-1 font-mono text-[10px] text-amber-700">
                        {provenance.applied_policy_ids.join(" · ")}
                      </p>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                      <SupportIcon type={provenance.support_type} />
                      {SUPPORT_LABELS[provenance.support_type] || provenance.support_type}
                    </span>
                    {provenance.evidence_ids.map((id) => {
                      const source = visibleEvidenceSources.find((item) => item.id === id)
                      return source ? (
                        <SourceBadge
                          key={id}
                          source={source}
                          number={sourceNumber.get(id) ?? 0}
                        />
                      ) : null
                    })}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </details>
      )}

      {!evidence.grounded && (
        <p className="text-xs text-muted-foreground">
          GraphRAG was disabled or no matching graph facts were retrieved for this decision.
        </p>
      )}
      {evidence.grounded && !evidence.evidence_backed && (
        <p className="text-xs text-muted-foreground">
          Graph facts were retrieved, but none of their source references resolve to a source in this response.
        </p>
      )}
    </div>
  )
}
