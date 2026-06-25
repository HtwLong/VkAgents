"use client"

import { useRef, useState } from "react"
import { ImageUp, Send, Sparkles, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { TASK_LABELS, type TaskType } from "@/lib/pipeline"

interface ChatMessage {
  role: "user" | "model"
  text: string
}

const DETECTION_BOXES = [
  { label: "hard-hat", conf: 0.97, top: "12%", left: "30%", w: "34%", h: "26%" },
  { label: "safety-vest", conf: 0.91, top: "44%", left: "24%", w: "46%", h: "44%" },
]

export function InferenceSection({
  task,
  enabled,
}: {
  task: TaskType
  enabled: boolean
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState("")
  const fileRef = useRef<HTMLInputElement>(null)

  // for automatic, resolve a concrete output type to render
  const resolved: Exclude<TaskType, "automatic"> =
    task === "automatic" ? "classification" : task

  const handleFile = (file?: File) => {
    if (!file) return
    setImageUrl(URL.createObjectURL(file))
    setMessages([])
  }

  const askQuestion = () => {
    const q = question.trim()
    if (!q) return
    setMessages((m) => [
      ...m,
      { role: "user", text: q },
      {
        role: "model",
        text: "Based on the image, the answer is likely affirmative — the model is 88% confident in this response.",
      },
    ])
    setQuestion("")
  }

  return (
    <section className="rounded-lg border border-border bg-card">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" aria-hidden />
          <h2 className="text-sm font-semibold">Inference</h2>
        </div>
        <span className="rounded-sm bg-secondary px-2 py-0.5 font-mono text-[11px] text-secondary-foreground">
          {TASK_LABELS[task]}
          {task === "automatic" && ` → ${TASK_LABELS[resolved]}`}
        </span>
      </header>

      {!enabled ? (
        <div className="px-4 py-10 text-center text-sm text-muted-foreground">
          Run the pipeline to unlock inference with your trained model.
        </div>
      ) : (
        <div className="grid gap-4 p-4 md:grid-cols-2">
          {/* Left: image upload */}
          <div className="flex flex-col">
            <span className="mb-1.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Input image
            </span>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="sr-only"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            {imageUrl ? (
              <div className="relative overflow-hidden rounded-md border border-border">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageUrl || "/placeholder.svg"}
                  alt="Uploaded for inference"
                  className="aspect-square w-full object-cover"
                />
                {resolved === "detection" &&
                  DETECTION_BOXES.map((b) => (
                    <div
                      key={b.label}
                      className="absolute rounded-sm border-2 border-primary"
                      style={{ top: b.top, left: b.left, width: b.w, height: b.h }}
                    >
                      <span className="absolute -top-5 left-0 whitespace-nowrap rounded-sm bg-primary px-1 py-0.5 font-mono text-[10px] text-primary-foreground">
                        {b.label} {b.conf}
                      </span>
                    </div>
                  ))}
                <button
                  type="button"
                  onClick={() => {
                    setImageUrl(null)
                    setMessages([])
                    if (fileRef.current) fileRef.current.value = ""
                  }}
                  className="absolute right-2 top-2 flex size-7 items-center justify-center rounded-md bg-background/80 text-foreground backdrop-blur transition-colors hover:bg-background"
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
                <span className="text-xs text-muted-foreground">
                  PNG or JPG
                </span>
              </button>
            )}
          </div>

          {/* Right: task-dependent output */}
          <div className="flex flex-col">
            <span className="mb-1.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Output
            </span>
            <div className="flex-1 rounded-md border border-border bg-background/40 p-3">
              {!imageUrl ? (
                <p className="flex h-full min-h-40 items-center justify-center text-center text-sm text-muted-foreground">
                  Upload an image to see results.
                </p>
              ) : resolved === "classification" ? (
                <ClassificationOutput />
              ) : resolved === "detection" ? (
                <DetectionOutput />
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

function ClassificationOutput() {
  const preds = [
    { label: "category_04", score: 0.947 },
    { label: "category_09", score: 0.038 },
    { label: "category_01", score: 0.011 },
  ]
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm">
        Predicted:{" "}
        <span className="font-mono font-medium text-primary">category_04</span>
      </p>
      <ul className="flex flex-col gap-2">
        {preds.map((p) => (
          <li key={p.label} className="flex flex-col gap-1">
            <div className="flex justify-between font-mono text-xs">
              <span className="text-muted-foreground">{p.label}</span>
              <span>{(p.score * 100).toFixed(1)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${p.score * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function DetectionOutput() {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-muted-foreground">2 objects detected:</p>
      <ul className="flex flex-col gap-2 font-mono text-xs">
        {DETECTION_BOXES.map((b) => (
          <li
            key={b.label}
            className="flex items-center justify-between rounded-sm border border-border bg-card px-2.5 py-1.5"
          >
            <span className="text-primary">{b.label}</span>
            <span className="text-muted-foreground">conf {b.conf}</span>
          </li>
        ))}
      </ul>
    </div>
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
  setQuestion: (v: string) => void
  onAsk: () => void
}) {
  return (
    <div className="flex h-full min-h-44 flex-col">
      <div className="flex-1 space-y-2 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="flex h-full items-center justify-center text-center text-sm text-muted-foreground">
            Ask a question about the image.
          </p>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={cn(
                "flex",
                m.role === "user" ? "justify-end" : "justify-start",
              )}
            >
              <span
                className={cn(
                  "max-w-[85%] rounded-md px-2.5 py-1.5 text-sm",
                  m.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground",
                )}
              >
                {m.text}
              </span>
            </div>
          ))
        )}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAsk()}
          placeholder="e.g. How many people are in the image?"
          className="flex-1 rounded-md border border-border bg-background px-2.5 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <button
          type="button"
          onClick={onAsk}
          className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground transition-opacity hover:opacity-90"
          aria-label="Send question"
        >
          <Send className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  )
}
