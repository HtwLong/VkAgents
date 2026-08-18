"use client"

import { useEffect, useRef, useState } from "react"
import { ImageUp, Loader2, Send, Sparkles, X } from "lucide-react"
import { cn } from "@/lib/utils"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend"

type InferenceTask = "classification" | "detection" | "vqa"

interface ClassificationResult {
  predicted_class: string
  confidence: number
  probabilities: Record<string, number>
}

interface Detection {
  box: [number, number, number, number]
  confidence: number
  class_id: number
  label: string
}

interface DetectionResult {
  detections_count: number
  image_width: number
  image_height: number
  detections: Detection[]
}

interface ChatMessage {
  role: "user" | "model"
  text: string
}

const TASK_LABELS: Record<InferenceTask, string> = {
  classification: "Image classification",
  detection: "Object detection",
  vqa: "Visual question answering",
}

async function errorMessage(response: Response) {
  try {
    const body = await response.json()
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body)
  } catch {
    return `${response.status} ${response.statusText}`
  }
}

export function InferenceSection({
  task,
  jobId,
  enabled,
}: {
  task: InferenceTask | null
  jobId: string | null
  enabled: boolean
}) {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [classification, setClassification] = useState<ClassificationResult | null>(null)
  const [detection, setDetection] = useState<DetectionResult | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl)
    }
  }, [imageUrl])

  const clearImage = () => {
    setImageFile(null)
    setImageUrl(null)
    setClassification(null)
    setDetection(null)
    setMessages([])
    setQuestion("")
    setError(null)
    if (fileRef.current) fileRef.current.value = ""
  }

  const infer = async (file: File) => {
    if (!jobId || !task || task === "vqa") return
    setLoading(true)
    setError(null)
    try {
      const loadResponse = await fetch(`${API_BASE}/api/v1/load-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId }),
      })
      if (!loadResponse.ok) throw new Error(await errorMessage(loadResponse))

      const body = new FormData()
      body.append("file", file)
      const response = await fetch(
        `${API_BASE}/api/v1/infer?job_id=${encodeURIComponent(jobId)}`,
        { method: "POST", body },
      )
      if (!response.ok) throw new Error(await errorMessage(response))
      const payload = await response.json()
      if (task === "classification") setClassification(payload.result)
      else setDetection(payload.result)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setLoading(false)
    }
  }

  const handleFile = (file?: File) => {
    if (!file) return
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
    setClassification(null)
    setDetection(null)
    setMessages([])
    setError(null)
    void infer(file)
  }

  const askQuestion = () => {
    const text = question.trim()
    if (!text || !imageFile) return
    setMessages((current) => [
      ...current,
      { role: "user", text },
      {
        role: "model",
        text: "VQA model inference is not implemented by the backend yet.",
      },
    ])
    setQuestion("")
  }

  return (
    <section className="surface-card overflow-hidden rounded-2xl border border-white/80 bg-card">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" aria-hidden />
          <h2 className="ui-card-title">Inference</h2>
        </div>
        {task && (
          <span className="rounded-sm bg-secondary px-2 py-0.5 font-mono text-[11px] text-secondary-foreground">
            {TASK_LABELS[task]}
          </span>
        )}
      </header>

      {!enabled || !jobId || !task ? (
        <div className="px-4 py-10 text-center text-sm text-muted-foreground">
          Complete the pipeline to run inference with the fine-tuned model.
        </div>
      ) : (
        <div className="grid gap-4 p-4 md:grid-cols-2">
          <div className="flex flex-col">
            <span className="ui-section-label mb-1.5">
              Input image
            </span>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="sr-only"
              onChange={(event) => handleFile(event.target.files?.[0])}
            />
            {imageUrl ? (
              <div className="relative overflow-hidden rounded-md border border-border bg-black/5">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={imageUrl} alt="Uploaded for inference" className="h-auto w-full" />
                {task === "detection" && detection && (
                  <DetectionOverlay result={detection} />
                )}
                {loading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background/70 backdrop-blur-sm">
                    <Loader2 className="size-6 animate-spin text-primary" aria-label="Running inference" />
                  </div>
                )}
                <button
                  type="button"
                  onClick={clearImage}
                  className="absolute right-2 top-2 flex size-7 items-center justify-center rounded-md bg-background/85 text-foreground shadow-sm"
                  aria-label="Remove image"
                >
                  <X className="size-4" aria-hidden />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="flex aspect-square w-full flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-background/40 text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
              >
                <ImageUp className="size-7" aria-hidden />
                <span className="text-sm font-medium">Upload an image</span>
                <span className="text-xs">PNG or JPG</span>
              </button>
            )}
          </div>

          <div className="flex flex-col">
            <span className="ui-section-label mb-1.5">
              {task === "vqa" ? "Conversation" : "Output"}
            </span>
            <div className="flex-1 rounded-md border border-border bg-background/40 p-3">
              {error ? (
                <p className="text-sm text-destructive">{error}</p>
              ) : !imageUrl ? (
                <p className="flex h-full min-h-40 items-center justify-center text-center text-sm text-muted-foreground">
                  Upload an image to {task === "vqa" ? "start a conversation" : "run inference"}.
                </p>
              ) : task === "classification" ? (
                <ClassificationOutput result={classification} loading={loading} />
              ) : task === "detection" ? (
                <DetectionOutput result={detection} loading={loading} />
              ) : (
                <VqaChat
                  messages={messages}
                  question={question}
                  setQuestion={setQuestion}
                  onAsk={askQuestion}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

function ClassificationOutput({ result, loading }: { result: ClassificationResult | null; loading: boolean }) {
  if (loading || !result) return <Waiting />
  const probabilities = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5)
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm">
        Predicted: <span className="font-mono font-medium text-primary">{result.predicted_class}</span>
        <span className="ml-2 text-xs text-muted-foreground">{(result.confidence * 100).toFixed(1)}%</span>
      </p>
      <ul className="flex flex-col gap-2">
        {probabilities.map(([label, score]) => (
          <li key={label} className="flex flex-col gap-1">
            <div className="flex justify-between font-mono text-xs">
              <span className="text-muted-foreground">{label}</span>
              <span>{(score * 100).toFixed(1)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
              <div className="h-full rounded-full bg-primary" style={{ width: `${score * 100}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function DetectionOverlay({ result }: { result: DetectionResult }) {
  return result.detections.map((item, index) => {
    const [x1, y1, x2, y2] = item.box
    return (
      <div
        key={`${item.class_id}-${index}`}
        className="absolute border-2 border-primary"
        style={{
          left: `${(x1 / result.image_width) * 100}%`,
          top: `${(y1 / result.image_height) * 100}%`,
          width: `${((x2 - x1) / result.image_width) * 100}%`,
          height: `${((y2 - y1) / result.image_height) * 100}%`,
        }}
      >
        <span className="absolute left-0 top-0 -translate-y-full whitespace-nowrap bg-primary px-1 py-0.5 font-mono text-[10px] text-primary-foreground">
          {item.label} {(item.confidence * 100).toFixed(0)}%
        </span>
      </div>
    )
  })
}

function DetectionOutput({ result, loading }: { result: DetectionResult | null; loading: boolean }) {
  if (loading || !result) return <Waiting />
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-muted-foreground">
        {result.detections_count === 0 ? "No objects detected." : `${result.detections_count} objects detected:`}
      </p>
      <ul className="flex flex-col gap-2 font-mono text-xs">
        {result.detections.map((item, index) => (
          <li key={`${item.class_id}-${index}`} className="flex justify-between rounded-sm border border-border bg-card px-2.5 py-1.5">
            <span className="text-primary">{item.label}</span>
            <span className="text-muted-foreground">{(item.confidence * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Waiting() {
  return (
    <p className="flex min-h-40 items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" aria-hidden /> Running inference…
    </p>
  )
}

function VqaChat({
  messages,
  question,
  setQuestion,
  onAsk,
}: {
  messages: ChatMessage[]
  question: string
  setQuestion: (value: string) => void
  onAsk: () => void
}) {
  return (
    <div className="flex h-full min-h-44 flex-col">
      <div className="flex-1 space-y-2 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="flex h-full items-center justify-center text-center text-sm text-muted-foreground">
            Ask a question or talk about the image.
          </p>
        ) : messages.map((message, index) => (
          <div key={index} className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}>
            <span className={cn(
              "max-w-[85%] rounded-md px-2.5 py-1.5 text-sm",
              message.role === "user" ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground",
            )}>
              {message.text}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && onAsk()}
          placeholder="Ask about the image…"
          className="flex-1 rounded-md border border-border bg-background px-2.5 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <button
          type="button"
          onClick={onAsk}
          disabled={!question.trim()}
          className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground disabled:opacity-50"
          aria-label="Send question"
        >
          <Send className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  )
}
