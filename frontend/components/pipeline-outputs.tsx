"use client"

import { Download, FileText, Package } from "lucide-react"
import { cn } from "@/lib/utils"
import type { DeliverableArtifact } from "@/lib/pipeline"

function OutputCard({
  icon,
  title,
  meta,
  ready,
  onDownload,
}: {
  icon: React.ReactNode
  title: string
  meta: string
  ready: boolean
  onDownload: () => void
}) {
  return (
    <button
      type="button"
      disabled={!ready}
      onClick={onDownload}
      className={cn(
        "group flex flex-1 items-center gap-3 rounded-md border p-3.5 text-left transition-colors",
        ready
          ? "cursor-pointer border-primary/40 bg-primary/5 hover:border-primary hover:bg-primary/10"
          : "cursor-not-allowed border-border bg-muted/30 opacity-60",
      )}
    >
      <span
        className={cn(
          "flex size-10 shrink-0 items-center justify-center rounded-md",
          ready ? "bg-primary/15 text-primary" : "bg-secondary text-muted-foreground",
        )}
      >
        {icon}
      </span>
      <span className="flex flex-1 flex-col">
        <span
          className={cn(
            "text-sm font-medium",
            ready ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {title}
        </span>
        <span className="font-mono text-xs text-muted-foreground">
          {ready ? meta : "available when run completes"}
        </span>
      </span>
      <Download
        className={cn(
          "size-4",
          ready ? "text-primary" : "text-muted-foreground",
        )}
        aria-hidden
      />
    </button>
  )
}

export function PipelineOutputs({
  ready,
  artifacts,
}: {
  ready: boolean
  artifacts: DeliverableArtifact[]
}) {
  const download = (url?: string) => {
    if (!url) return
    const a = document.createElement("a")
    a.href = url
    a.download = ""
    a.click()
  }

  const displayedArtifacts = artifacts.length > 0
    ? artifacts
    : [
        {
          id: "model-placeholder",
          kind: "full_model",
          label: "Model artifact",
          filename: "model artifact",
        },
      ]

  return (
    <section className="surface-card flex flex-col gap-3 rounded-2xl border border-white/80 bg-white/82 p-4 sm:p-5">
      <h2 className="ui-card-title">Deliverables</h2>
      <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
        {displayedArtifacts.map((artifact) => (
          <OutputCard
            key={artifact.id}
            icon={["configuration", "provenance_audit"].includes(artifact.kind)
              ? <FileText className="size-5" aria-hidden />
              : <Package className="size-5" aria-hidden />}
            title={artifact.label}
            meta={artifact.description ?? artifact.filename}
            ready={ready && !!artifact.downloadUrl}
            onDownload={() => download(artifact.downloadUrl)}
          />
        ))}
      </div>
    </section>
  )
}
