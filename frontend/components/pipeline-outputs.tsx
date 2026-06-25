"use client"

import { Download, FileText, Package } from "lucide-react"
import { cn } from "@/lib/utils"

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

export function PipelineOutputs({ ready }: { ready: boolean }) {
  const download = (name: string, content: string) => {
    const blob = new Blob([content], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Deliverables
      </span>
      <div className="flex flex-col gap-2.5 sm:flex-row">
        <OutputCard
          icon={<Package className="size-5" aria-hidden />}
          title="Trained model"
          meta="model.safetensors · 112 MB"
          ready={ready}
          onDownload={() =>
            download(
              "model.safetensors.txt",
              "Placeholder model artifact exported by the Adaptive CV Pipeline.",
            )
          }
        />
        <OutputCard
          icon={<FileText className="size-5" aria-hidden />}
          title="PDF summary"
          meta="report.pdf · 1.4 MB"
          ready={ready}
          onDownload={() =>
            download(
              "report.txt",
              "Adaptive CV Pipeline — Training & Evaluation Summary.\nTop-1 accuracy: 0.947 · Latency: 8.3ms/img.",
            )
          }
        />
      </div>
    </div>
  )
}
