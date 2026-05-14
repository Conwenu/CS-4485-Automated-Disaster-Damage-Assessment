import { useState, useEffect, useRef, useMemo } from "react";
import "./FloatingChatbot.css";

type ChatRole = "user" | "assistant" | "error";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  contentHtml: string;
  suggestions?: string[];
  timestamp: string;
  isStreaming?: boolean;
};


type StreamMetadata = {
  suggestions?: string[];
  ui_actions?: UiAction[];
  requires_clarification?: boolean;
  pending_clarification?: Record<string, any> | null;
};

type UiAction = {
  type: string;
  [key: string]: any;
};

const STARTER_QUESTIONS = [
  {
    label: "Overview",
    text: "Summarize damage in the affected region",
  },
  {
    label: "Buildings",
    text: "Which buildings are classified as severe damage?",
  },
  {
    label: "Model",
    text: "How is the current model performing on recent assessments?",
  },
];

function newSessionId(): string {
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function loadOrCreateSessionId(): string {
  let id = sessionStorage.getItem("damage_assist_session_id");
  if (!id) {
    id = newSessionId();
    sessionStorage.setItem("damage_assist_session_id", id);
  }
  return id;
}

function timeNow(): string {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function unescapeHtml(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");
}

function renderInline(text: string): string {
  const codeSpans: string[] = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => {
    codeSpans.push(code);
    return `\x01C${codeSpans.length - 1}\x01`;
  });

  text = text.replace(
    /\[([^\]]+)\]\(([^\s)]+)(?:\s+"([^"]*)")?\)/g,
    (_, label, url, title) => {
      const safe = /^(https?:|mailto:|\/|#|\.\/)/i.test(url) ? url : "#";
      const t = title ? ` title="${title}"` : "";
      return `<a href="${safe}" target="_blank" rel="noopener noreferrer"${t}>${label}</a>`;
    },
  );

  text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
  text = text.replace(
    /(^|[^*\w])\*([^*\s][^*\n]*?[^*\s]|\S)\*(?!\w)/g,
    "$1<em>$2</em>",
  );
  text = text.replace(
    /(^|[^_\w])_([^_\s][^_\n]*?[^_\s]|\S)_(?!\w)/g,
    "$1<em>$2</em>",
  );
  text = text.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  text = text.replace(
    /\x01C(\d+)\x01/g,
    (_, i) => `<code>${codeSpans[+i]}</code>`,
  );

  return text;
}

function renderMarkdown(src: string): string {
  if (!src) return "";

  const codeBlocks: Array<{ lang: string; code: string }> = [];
  src = src.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    codeBlocks.push({ lang: lang.trim(), code: code.replace(/\n$/, "") });
    return `\x02CB${codeBlocks.length - 1}\x02`;
  });

  src = escapeHtml(src);

  const lines = src.split("\n");
  const out: string[] = [];
  let i = 0;
  const isBlank = (s: string) => /^\s*$/.test(s);

  while (i < lines.length) {
    const line = lines[i];

    const cbMatch = line.match(/^\x02CB(\d+)\x02\s*$/);
    if (cbMatch) {
      const cb = codeBlocks[+cbMatch[1]];
      const langAttr = cb.lang ? ` data-lang="${escapeHtml(cb.lang)}"` : "";
      out.push(`<pre${langAttr}><code>${escapeHtml(cb.code)}</code></pre>`);
      i++;
      continue;
    }

    const hMatch = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (hMatch) {
      const level = hMatch[1].length;
      out.push(`<h${level}>${renderInline(hMatch[2])}</h${level}>`);
      i++;
      continue;
    }

    if (
      /^\s{0,3}(-\s*){3,}$|^\s{0,3}(\*\s*){3,}$|^\s{0,3}(_\s*){3,}$/.test(line)
    ) {
      out.push("<hr>");
      i++;
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quoted: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        quoted.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      out.push(
        `<blockquote>${renderMarkdown(unescapeHtml(quoted.join("\n")))}</blockquote>`,
      );
      continue;
    }

    if (/^\s{0,3}[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s{0,3}[-*+]\s+/.test(lines[i])) {
        let item = lines[i].replace(/^\s{0,3}[-*+]\s+/, "");
        i++;
        while (
          i < lines.length &&
          !isBlank(lines[i]) &&
          !/^\s{0,3}[-*+]\s+/.test(lines[i]) &&
          !/^\s{0,3}\d+[.)]\s+/.test(lines[i]) &&
          /^\s{2,}|^\s*[^#>`\x02]/.test(lines[i])
        ) {
          if (/^\s{0,3}#{1,6}\s/.test(lines[i])) break;
          item += " " + lines[i].trim();
          i++;
        }
        items.push(`<li>${renderInline(item)}</li>`);
      }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\s{0,3}\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s{0,3}\d+[.)]\s+/.test(lines[i])) {
        let item = lines[i].replace(/^\s{0,3}\d+[.)]\s+/, "");
        i++;
        while (
          i < lines.length &&
          !isBlank(lines[i]) &&
          !/^\s{0,3}\d+[.)]\s+/.test(lines[i]) &&
          !/^\s{0,3}[-*+]\s+/.test(lines[i]) &&
          /^\s{2,}|^\s*[^#>`\x02]/.test(lines[i])
        ) {
          if (/^\s{0,3}#{1,6}\s/.test(lines[i])) break;
          item += " " + lines[i].trim();
          i++;
        }
        items.push(`<li>${renderInline(item)}</li>`);
      }
      out.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (isBlank(line)) {
      i++;
      continue;
    }

    const paraLines: string[] = [];
    while (
      i < lines.length &&
      !isBlank(lines[i]) &&
      !/^#{1,6}\s/.test(lines[i]) &&
      !/^\s*>/.test(lines[i]) &&
      !/^\s{0,3}[-*+]\s+/.test(lines[i]) &&
      !/^\s{0,3}\d+[.)]\s+/.test(lines[i]) &&
      !/^\x02CB\d+\x02\s*$/.test(lines[i]) &&
      !/^\s{0,3}(-\s*){3,}$|^\s{0,3}(\*\s*){3,}$|^\s{0,3}(_\s*){3,}$/.test(
        lines[i],
      )
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      const joined = paraLines
        .map((l, idx) =>
          idx < paraLines.length - 1 && /  $/.test(l)
            ? l.trimEnd() + "\x03"
            : l,
        )
        .join(" ")
        .replace(/\x03 /g, "<br>");
      out.push(`<p>${renderInline(joined)}</p>`);
    }
  }

  return out.join("");
}

/**
 * Wraps the trailing characters of rendered HTML in animated spans
 * for a character-by-character blur-in streaming effect.
 *
 * Only wraps the last `tailLength` characters of visible text so that
 * already-rendered content doesn't re-animate on each update.
 */
function withStreamingTail(html: string, tailLength: number): string {
  if (!html || tailLength <= 0) return html;

  // Find the last text node region — walk backwards through the HTML,
  // counting visible characters (skipping tags) until we hit tailLength.
  let chars = 0;
  let cutIdx = html.length;
  let inTag = false;
  let inEntity = false;

  for (let i = html.length - 1; i >= 0; i--) {
    const c = html[i];
    if (c === ">") {
      inTag = true;
      continue;
    }
    if (c === "<") {
      inTag = false;
      continue;
    }
    if (inTag) continue;

    // Handle HTML entities (&amp;, &lt;, etc.) — count them as 1 char
    if (c === ";") {
      inEntity = true;
      continue;
    }
    if (inEntity) {
      if (c === "&") {
        chars++;
        inEntity = false;
        cutIdx = i;
        if (chars >= tailLength) break;
      }
      continue;
    }

    chars++;
    cutIdx = i;
    if (chars >= tailLength) break;
  }

  if (cutIdx >= html.length) return html;

  const head = html.slice(0, cutIdx);
  const tail = html.slice(cutIdx);

  // Wrap each character in tail in an animated span, but skip tag boundaries.
  let wrapped = "";
  let i = 0;
  while (i < tail.length) {
    const c = tail[i];

    // Skip tag passthrough — don't wrap HTML tags
    if (c === "<") {
      const end = tail.indexOf(">", i);
      if (end === -1) {
        wrapped += tail.slice(i);
        break;
      }
      wrapped += tail.slice(i, end + 1);
      i = end + 1;
      continue;
    }

    // Skip HTML entities — wrap the entire entity as one unit
    if (c === "&") {
      const end = tail.indexOf(";", i);
      if (end !== -1 && end - i < 10) {
        wrapped += `<span class="fc-stream-char">${tail.slice(i, end + 1)}</span>`;
        i = end + 1;
        continue;
      }
    }

    // Wrap regular character with staggered animation delay
    const delay = Math.min((i / tail.length) * 200, 200);
    const spaceAttr = c === " " ? ` data-space="true"` : "";
    wrapped += `<span class="fc-stream-char" style="animation-delay:${delay.toFixed(0)}ms"${spaceAttr}>${c}</span>`;
    i++;
  }

  return head + wrapped;
}

export function FloatingChatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>(loadOrCreateSessionId);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const renderTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const messageIndices = useMemo(() => {
    let userCount = 0;
    return messages.map((m) => {
      if (m.role === "user") userCount++;
      return m.role === "error"
        ? "!!"
        : String(Math.max(1, userCount)).padStart(2, "0");
    });
  }, [messages]);

  const exchangeCount = useMemo(
    () => messages.filter((m) => m.role === "user").length,
    [messages],
  );

  const hasMessages = messages.length > 0;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (isOpen) {
      const t = setTimeout(() => inputRef.current?.focus(), 320);
      return () => clearTimeout(t);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isOpen]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height =
        Math.min(inputRef.current.scrollHeight, 200) + "px";
    }
  }, [input]);

  const handleSend = async (questionOverride?: string) => {
    const query = (questionOverride ?? input).trim();
    if (!query || isLoading) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content: query,
      contentHtml: renderMarkdown(query),
      timestamp: timeNow(),
    };

    const history = messages
      .filter((m) => m.role !== "error")
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Create the assistant message immediately with empty content.
    // The streaming loop will update it in place.
    const assistantId = `msg-${Date.now()}-assistant`;
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      contentHtml: "",
      suggestions: [],
      timestamp: timeNow(),
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const apiBase = import.meta.env.VITE_API_URL ?? "";
      const response = await fetch(`${apiBase}/query/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          history,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? ""; // keep the last incomplete event

        for (const event of events) {
          if (!event.trim()) continue;

          const lines = event.split("\n");
          const eventType = lines
            .find((l) => l.startsWith("event:"))
            ?.replace("event:", "")
            .trim();
          const dataLine = lines
            .find((l) => l.startsWith("data:"))
            ?.replace("data:", "")
            .trim();

          if (!dataLine) continue;

          let parsed: any;
          try {
            parsed = JSON.parse(dataLine);
          } catch {
            continue;
          }

          if (eventType === "token") {
            const tokenData = parsed as { text: string };
            const prevLength = accumulated.length;
            accumulated += tokenData.text;
            const newChars = accumulated.length - prevLength;

            if (renderTimer.current) clearTimeout(renderTimer.current);
            renderTimer.current = setTimeout(() => {
              const baseHtml = renderMarkdown(accumulated);
              // Wrap the last `newChars + buffer` characters in animated spans.
              // The extra buffer ensures characters from the previous batch
              // that haven't finished animating still look smooth.
              const html = withStreamingTail(baseHtml, newChars + 20);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: accumulated,
                        contentHtml: html,
                        isStreaming: true,
                      }
                    : m,
                ),
              );
            }, 80);
          } else if (eventType === "metadata") {
            // Attach suggestions and ui_actions when text is complete
            const meta = parsed as StreamMetadata;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      suggestions: meta.suggestions ?? [],
                      isStreaming: false,
                    }
                  : m,
              ),
            );
          } else if (eventType === "done") {
            if (renderTimer.current) {
              clearTimeout(renderTimer.current);
              renderTimer.current = null;
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: accumulated,
                      contentHtml: renderMarkdown(accumulated), // no wrapping
                      isStreaming: false,
                    }
                  : m,
              ),
            );
          }
        }
      }
    } catch (error) {
      // Remove the empty assistant message and show error instead
      setMessages((prev) => {
        const withoutEmpty = prev.filter((m) => m.id !== assistantId);
        const errText =
          error instanceof Error ? error.message : "Request failed.";
        return [
          ...withoutEmpty,
          {
            id: `msg-${Date.now()}-error`,
            role: "error" as ChatRole,
            content: errText,
            contentHtml: escapeHtml(errText),
            timestamp: timeNow(),
          },
        ];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setInput("");
    const newId = newSessionId();
    sessionStorage.setItem("damage_assist_session_id", newId);
    setSessionId(newId);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = (content: string, el: HTMLButtonElement) => {
    navigator.clipboard.writeText(content).then(() => {
      const span = el.querySelector("span");
      if (!span) return;
      el.classList.add("copied");
      span.textContent = "Copied";
      setTimeout(() => {
        el.classList.remove("copied");
        span.textContent = "Copy";
      }, 1400);
    });
  };

  return (
    <>
      <button
        type="button"
        className={`fc-fab${isOpen ? " hidden" : ""}`}
        onClick={() => setIsOpen(true)}
        aria-label="Open Damage Assist chatbot"
      >
        <span className="fc-fab-icon" aria-hidden="true">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </span>
        <span className="fc-fab-label">Ask Damage Assist</span>
      </button>

      <div
        className={`fc-backdrop${isOpen ? " open" : ""}`}
        onClick={() => setIsOpen(false)}
        aria-hidden="true"
      />

      <aside
        className={`fc-panel${isOpen ? " open" : ""}`}
        role="dialog"
        aria-label="Damage Assist chatbot"
        aria-hidden={!isOpen}
      >
        <header className="fc-topbar">
          <div className="fc-brand">
            <div className="fc-brand-mark">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="2.4"
                strokeLinecap="round"
              >
                <circle cx="12" cy="12" r="3" fill="white" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
            </div>
            <span className="fc-brand-name">Damage Assist</span>
          </div>
          <div className="fc-topbar-right">
            {hasMessages && (
              <button
                type="button"
                className="fc-clear"
                onClick={handleClear}
                aria-label="Clear chat history"
                title="Clear chat"
              >
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M3 6h18" />
                  <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                </svg>
                <span>Clear</span>
              </button>
            )}
            <button
              type="button"
              className="fc-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </header>

        <main className="fc-chat">
          {messages.length === 0 && !isLoading ? (
            <div className="fc-empty">
              <div className="fc-empty-eyebrow">Damage Assist · Ready</div>
              <h2 className="fc-empty-headline">
                Ask about <em>damage</em>,<br />
                models, or a building.
              </h2>
              <p className="fc-empty-sub">
                Query the damage assessment index for structural status, model
                confidence, or specific parcels.
              </p>
              <div className="fc-starter-label">Try asking</div>
              <div className="fc-starter-grid">
                {STARTER_QUESTIONS.map((q) => (
                  <button
                    key={q.label}
                    type="button"
                    className="fc-starter-card"
                    onClick={() => handleSend(q.text)}
                  >
                    <div className="fc-starter-card-label">{q.label}</div>
                    <div className="fc-starter-card-text">{q.text}</div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => {
                const indexStr = messageIndices[idx];
                const roleLabel =
                  msg.role === "user"
                    ? "You"
                    : msg.role === "error"
                      ? "Error"
                      : "Assist";

                return (
                  <div key={msg.id} className={`fc-msg ${msg.role}`}>
                    <div className="fc-msg-head">
                      <span className="fc-msg-head-index">#{indexStr}</span>
                      <span className="fc-msg-head-role">{roleLabel}</span>
                      <span className="fc-msg-head-line" />
                      <span className="fc-msg-head-time">{msg.timestamp}</span>
                    </div>

                    <div
                      className="fc-bubble"
                      dangerouslySetInnerHTML={{
                        __html:
                          msg.contentHtml ||
                          (msg.isStreaming
                            ? `<div class="fc-loading-dots">
                              <div class="fc-dots"><span/><span/><span/></div>
                              <span class="fc-loading-label">Thinking</span>
                            </div>`
                            : ""),
                      }}
                    />

                    {msg.role === "assistant" && (
                      <div className="fc-msg-actions">
                        <button
                          type="button"
                          className="fc-action-btn"
                          onClick={(e) =>
                            handleCopy(msg.content, e.currentTarget)
                          }
                        >
                          <svg
                            width="11"
                            height="11"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <rect
                              x="9"
                              y="9"
                              width="13"
                              height="13"
                              rx="2"
                              ry="2"
                            />
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                          </svg>
                          <span>Copy</span>
                        </button>
                      </div>
                    )}

                    {msg.suggestions && msg.suggestions.length > 0 && (
                      <div className="fc-suggestions">
                        <div className="fc-suggestions-label">
                          <span className="fc-suggestions-label-icon">
                            <svg
                              width="11"
                              height="11"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2.4"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <path d="M7 17L17 7" />
                              <path d="M7 7h10v10" />
                            </svg>
                          </span>
                          <span>Related queries</span>
                        </div>
                        {msg.suggestions.map((s, i) => (
                          <button
                            key={i}
                            type="button"
                            className="fc-chip"
                            onClick={() => handleSend(s)}
                          >
                            <span className="fc-chip-fill" />
                            <span
                              className="fc-chip-num"
                              data-num={String(i + 1).padStart(2, "0")}
                            />
                            <span className="fc-chip-text">{s}</span>
                            <span className="fc-chip-arrow">→</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}

              {isLoading && !messages.some((m) => m.isStreaming) && (
                <div className="fc-msg assistant">
                  <div className="fc-msg-head">
                    <span className="fc-msg-head-index">
                      #{String(exchangeCount).padStart(2, "0")}
                    </span>
                    <span className="fc-msg-head-role">Assist</span>
                    <span className="fc-msg-head-line" />
                  </div>
                  <div className="fc-bubble">
                    <div className="fc-loading-dots">
                      <div className="fc-dots">
                        <span />
                        <span />
                        <span />
                      </div>
                      <span className="fc-loading-label">Thinking</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </main>

        <div className="fc-input-area">
          <div className={`fc-input-frame${isLoading ? " sending" : ""}`}>
            <div className="fc-input-row">
              <textarea
                ref={inputRef}
                className="fc-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about damage, buildings, or model performance…"
                rows={1}
                disabled={isLoading}
              />
              <button
                type="button"
                className="fc-send-btn"
                onClick={() => handleSend()}
                disabled={isLoading || !input.trim()}
                aria-label="Send message"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M5 12h14M13 6l6 6-6 6"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
            <div className="fc-input-meta">
              <span className="fc-kbd">⇧</span>
              <span className="fc-kbd">↵</span>
              <span>newline ·</span>
              <span className="fc-kbd">↵</span>
              <span>send</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

export default FloatingChatbot;
