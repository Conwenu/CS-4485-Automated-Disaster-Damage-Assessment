import { useMemo, useState, useEffect } from "react";
import "./App.css";
import summaryData from "../../Backend/data/santa_rosa/building_summary.json"

type DamageLevel = "noDamage" | "minorDamage" | "severeDamage";

type PropertyPoint = {
  id: string;
  damageLevel: DamageLevel;
  row: number;
  col: number;
};

const PROPERTIES: PropertyPoint[] = [
  {
    id: "p1",
    damageLevel: "noDamage",
    row: 0,
    col: 0,
  },
  {
    id: "p2",
    damageLevel: "minorDamage",
    row: 0,
    col: 1,
  },
  {
    id: "p3",
    damageLevel: "severeDamage",
    row: 0,
    col: 2,
  },
  {
    id: "p4",
    damageLevel: "minorDamage",
    row: 1,
    col: 0,
  },
  {
    id: "p5",
    damageLevel: "noDamage",
    row: 1,
    col: 1,
  },
  {
    id: "p6",
    damageLevel: "severeDamage",
    row: 1,
    col: 2,
  },
];

type ChatMessage = {
  id: number;
  sender: "user" | "assistant";
  text: string;
};

type BoundingBox = {
  building_id: string;
  subtype: string;
  bbox: [number, number, number, number]; 
}

type VlmDemoResult = {
  model: {
    damage_level?: string;
    confidence?: number | null;
    reasoning?: string;
  };
  inputs?: {
    pre_image?: string;
    post_image?: string;
  };
};

const EVALUATION_LABELS = [
  "no-damage",
  "minor-damage",
  "major-damage",
  "destroyed",
  "un-classified",
] as const;

type EvaluationLabel = (typeof EVALUATION_LABELS)[number];

function App() {
  const [damageFilter, setDamageFilter] = useState<{
    noDamage: boolean;
    minorDamage: boolean;
    severeDamage: boolean;
  }>({
    noDamage: true,
    minorDamage: true,
    severeDamage: true,
  });
  const [mapZoom, setMapZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [activeImageTab, setActiveImageTab] = useState<"before" | "after">("after");
  const [selectedPropertyId, setSelectedPropertyId] = useState<string>(
    PROPERTIES[2].id,
  );
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      sender: "assistant",
      text: "Ask me about damage patterns, affected areas, or overall impact.",
    },
  ]);
  const [demoPreImage, setDemoPreImage] = useState<File | null>(null);
  const [demoPostImage, setDemoPostImage] = useState<File | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState("");
  const [demoResult, setDemoResult] = useState<VlmDemoResult | null>(null);

  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([]);
  useEffect(() => {
  fetch("http://localhost:8000/api/bounding-boxes/?image_id=santa-rosa-00000000")
    .then((r) => r.json())
    .then((data) => {
      console.log("Bounding boxes:", data);
      setBoundingBoxes(data);
    })
    .catch((err) => console.error("Failed to fetch bounding boxes:", err));
  }, []);

  const selectedProperty = useMemo(
    () => PROPERTIES.find((p) => p.id === selectedPropertyId) ?? PROPERTIES[0],
    [selectedPropertyId],
  );

  const filteredProperties = useMemo(
    () =>
      PROPERTIES.filter((p) => {
        if (p.damageLevel === "noDamage") return damageFilter.noDamage;
        if (p.damageLevel === "minorDamage") return damageFilter.minorDamage;
        return damageFilter.severeDamage;
      }),
    [damageFilter],
  );

  const handleToggleDamage = (level: DamageLevel) => {
    setDamageFilter((prev) => {
      const next = { ...prev };
      if (level === "noDamage") next.noDamage = !prev.noDamage;
      if (level === "minorDamage") next.minorDamage = !prev.minorDamage;
      if (level === "severeDamage") next.severeDamage = !prev.severeDamage;
      if (!next.noDamage && !next.minorDamage && !next.severeDamage) {
        return prev;
      }
      return next;
    });
  };

  const handleSendChat = () => {
    const trimmed = chatInput.trim();
    if (!trimmed) return;

    const nextId = chatMessages.length
      ? chatMessages[chatMessages.length - 1].id + 1
      : 1;

    const userMessage: ChatMessage = {
      id: nextId,
      sender: "user",
      text: trimmed,
    };

    const assistantMessage: ChatMessage = {
      id: nextId + 1,
      sender: "assistant",
      text: `This is a placeholder analysis for "${trimmed}". Once the model is connected, this area will summarize damage patterns and affected properties for the selected disaster.`,
    };

    setChatMessages((prev) => [...prev, userMessage, assistantMessage]);
    setChatInput("");
  };

  const handleVlmDemoSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!demoPreImage || !demoPostImage) {
      setDemoError("Please upload both a pre-disaster image and a post-disaster image.");
      return;
    }

    const formData = new FormData();
    formData.append("pre_image", demoPreImage);
    formData.append("post_image", demoPostImage);

    setDemoLoading(true);
    setDemoError("");
    setDemoResult(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/vlm/assess-demo`,  {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = (await response.json()) as VlmDemoResult;
      setDemoResult(data);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "The VLM evaluation request failed.";
      setDemoError(message);
    } finally {
      setDemoLoading(false);
    }
  };

  const renderHouse = (property: PropertyPoint) => {
    const isVisible = filteredProperties.some((p) => p.id === property.id);
    if (!isVisible) {
      return <div key={property.id} className="map-house empty-slot" />;
    }

    const damageClass =
      property.damageLevel === "noDamage"
        ? "no-damage"
        : property.damageLevel === "minorDamage"
          ? "minor-damage"
          : "severe-damage";

    return (
      <button
        key={property.id}
        type="button"
        className={`map-house ${damageClass}${
          property.id === selectedPropertyId ? " selected" : ""
        }`}
        onClick={() => setSelectedPropertyId(property.id)}
        aria-label={`Property – ${damageClass.replace("-", " ")}`}
      />
    );
  };

  const row0 = PROPERTIES.filter((p) => p.row === 0).sort(
    (a, b) => a.col - b.col,
  );
  const row1 = PROPERTIES.filter((p) => p.row === 1).sort(
    (a, b) => a.col - b.col,
  );
  const groundTruthCounts = summaryData.ground_truth_counts as Partial<
    Record<EvaluationLabel, number>
  >;
  const predictionCounts = summaryData.prediction_counts as Partial<
    Record<EvaluationLabel, number>
  >;
  const confusionCounts = summaryData.confusion_counts as Record<string, number>;
  const evaluationMatrix = useMemo(
    () =>
      EVALUATION_LABELS.map((groundTruth) => ({
        groundTruth,
        predictions: EVALUATION_LABELS.map(
          (prediction) => confusionCounts[`${groundTruth} -> ${prediction}`] ?? 0,
        ),
      })),
    [confusionCounts],
  );
  const topConfusions = useMemo(
    () =>
      Object.entries(confusionCounts)
        .filter(([key, count]) => {
          const [groundTruth, prediction] = key.split(" -> ");
          return groundTruth !== prediction && count > 0;
        })
        .sort(([, a], [, b]) => b - a)
        .slice(0, 4),
    [confusionCounts],
  );
  const evaluatedPct = ((summaryData.matched / summaryData.total) * 100).toFixed(2);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1 className="app-title">Disaster Assessment Dashboard</h1>
          <p className="app-subtitle">Vision–Language Model Powered</p>
        </div>
        <div className="app-header-right">
          <span className="header-disaster-pill">Santa Rosa Wildfire Disaster</span>
        </div>
      </header>

      <main className="app-main">
        <section className="left-column">
          <section className="panel imagery-panel">
            <div className="imagery-tabs">
              <button
                type="button"
                className={`imagery-tab${activeImageTab === "before" ? " active" : ""}`}
                onClick={() => setActiveImageTab("before")}
              >
                Before
              </button>
              <button
                type="button"
                className={`imagery-tab${activeImageTab === "after" ? " active" : ""}`}
                onClick={() => setActiveImageTab("after")}
              >
                After
              </button>
            </div>
            <div className="imagery-content">
              {activeImageTab === "before" ? (
                <img
                  src="/santa-rosa-wildfire_00000000_pre_disaster.png"
                  alt="Before"
                  className="imagery-img"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              ) : (
                <img
                  src="/santa-rosa-wildfire_00000000_post_disaster.png"
                  alt="After disaster"
                  className="imagery-img"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              )}
              <div className="imagery-label">
                {activeImageTab === "before" ? "Before – pre-disaster" : "After – post-disaster"}
              </div>
            </div>
            <div className="imagery-details">
            <div className="imagery-pill">
        </div>
        <div className="imagery-meta">
          <div className="imagery-meta-title">Selected property</div>
          <div className="imagery-meta-damage">
            Predicted damage:{" "}
            <span className={`damage-tag ${selectedProperty.damageLevel}`}>
              {selectedProperty.damageLevel === "noDamage" && "No Damage"}
              {selectedProperty.damageLevel === "minorDamage" && "Minor Damage"}
              {selectedProperty.damageLevel === "severeDamage" && "Severe Damage"}
            </span>
          </div>
        </div>
      </div>
      
          </section>

          <section className="panel legend-panel">
            <div className="panel-header">
              <h2>Damage Filters</h2>
            </div>
            <ul className="legend-list">
              <li
                className={`legend-toggle${
                  damageFilter.noDamage ? "" : " inactive"
                }`}
                onClick={() => handleToggleDamage("noDamage")}
              >
                <span className="legend-dot no-damage" />
                <span>No Damage</span>
              </li>
              <li
                className={`legend-toggle${
                  damageFilter.minorDamage ? "" : " inactive"
                }`}
                onClick={() => handleToggleDamage("minorDamage")}
              >
                <span className="legend-dot minor-damage" />
                <span>Minor Damage</span>
              </li>
              <li
                className={`legend-toggle${
                  damageFilter.severeDamage ? "" : " inactive"
                }`}
                onClick={() => handleToggleDamage("severeDamage")}
              >
                <span className="legend-dot severe-damage" />
                <span>Severe Damage</span>
              </li>
            </ul>
          </section>
          <section className="panel summary-panel">
            <div className="panel-header">
              <h2>Summary</h2>
            </div>
            <div className="summary-grid">
              <div className="summary-card severe">
                <div className="summary-num">
                  {(summaryData.prediction_counts["destroyed"] + summaryData.prediction_counts["major-damage"]).toLocaleString()}
                </div>
                <div className="summary-label">Severe</div>
              </div>
              <div className="summary-card minor">
                <div className="summary-num">
                  {summaryData.prediction_counts["minor-damage"].toLocaleString()}
                </div>
                <div className="summary-label">Minor</div>
              </div>
              <div className="summary-card none">
                <div className="summary-num">
                  {summaryData.prediction_counts["no-damage"].toLocaleString()}
                </div>
                <div className="summary-label">No Damage</div>
              </div>
              <div className="summary-card total">
                <div className="summary-num">
                  {summaryData.total.toLocaleString()}
                </div>
                <div className="summary-label">Total Buildings</div>
              </div>
            </div>
          </section>
        </section>

        

        <section className="right-column">
          <section className="panel map-panel">
            <div className="panel-header">
              <h2>Property Damage Map</h2>
            </div>
            <div className="map-container">
              <div className="map-toolbar">
                <button className="map-button" type="button" onClick={() => setMapZoom(z => Math.min(z + 0.2, 3))}>
                  +
                </button>
                <button className="map-button" type="button" onClick={() => setMapZoom(z => Math.max(z - 0.2, 0.5))}>
                  −
                </button>
              </div>
              <div
                className="map-placeholder"
                style={{ overflow: "hidden", position: "relative", cursor: isDragging ? "grabbing" : "grab" }}
                onMouseDown={(e) => {
                  setIsDragging(true);
                  setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
                }}
                onMouseMove={(e) => {
                  if (!isDragging) return;
                  setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
                }}
                onMouseUp={() => setIsDragging(false)}
                onMouseLeave={() => setIsDragging(false)}
              >
                <img
                  src={activeImageTab === "before"
                    ? "/santa-rosa-wildfire_00000000_pre_disaster.png"
                    : "/santa-rosa-wildfire_00000000_post_disaster.png"}
                  alt="Satellite view"
                  draggable={false}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    transform: `scale(${mapZoom}) translate(${pan.x / mapZoom}px, ${pan.y / mapZoom}px)`,
                    transformOrigin: "center center",
                    transition: isDragging ? "none" : "transform 0.2s ease",
                    display: "block",
                    userSelect: "none",
                  }}
                />
                <svg
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    transform: `scale(${mapZoom}) translate(${pan.x / mapZoom}px, ${pan.y / mapZoom}px)`,
                    transformOrigin: "center center",
                    transition: isDragging ? "none" : "transform 0.2s ease",
                    overflow: "visible",
                    pointerEvents: "none",
                  }}
                  viewBox="0 0 1024 1024"
                  preserveAspectRatio="xMidYMid meet"
                >
                  {boundingBoxes
                    .filter((box) => {
                      if (box.subtype === "no-damage") return damageFilter.noDamage;
                      if (box.subtype === "minor-damage") return damageFilter.minorDamage;
                      return damageFilter.severeDamage;
                    })
                    .map(({ building_id, bbox }) => {
                      const [minX, minY, maxX, maxY] = bbox;

                      return (
                        <rect
                          key={building_id}
                          x={minX}
                          y={minY}
                          width={maxX - minX}
                          height={maxY - minY}
                          fill="none"
                          stroke="black"
                          strokeWidth={2}
                          strokeDasharray="4 2"
                        />
                      );
                    })}
                </svg>
                <div style={{ position: "relative", zIndex: 1 }}>
                  <div className="map-houses-row">
                    {row0.map((property) => renderHouse(property))}
                  </div>
                  <div className="map-houses-row">
                    {row1.map((property) => renderHouse(property))}
                  </div>
                </div>
              </div>
              <div className="map-legend-floating">
                <div className="legend-title">Damage</div>
                <div className="legend-row">
                  <span className="legend-dot no-damage" />
                  <span>No Damage</span>
                </div>
                <div className="legend-row">
                  <span className="legend-dot minor-damage" />
                  <span>Minor Damage</span>
                </div>
                <div className="legend-row">
                  <span className="legend-dot severe-damage" />
                  <span>Severe Damage</span>
                </div>
              </div>
              <div className="map-legend-bottom">
                <span className="legend-item">
                  <span className="legend-dot no-damage" />
                  No Damage
                </span>
                <span className="legend-item">
                  <span className="legend-dot minor-damage" />
                  Minor Damage
                </span>
                <span className="legend-item">
                  <span className="legend-dot severe-damage" />
                  Severe Damage
                </span>
              </div>
            </div>
          </section>
          <section className="panel vlm-demo-panel">
            <div className="panel-header">
              <h2>VLM Evaluation</h2>
            </div>
            <p className="vlm-demo-copy">
              Upload a pre-disaster and post-disaster image pair to evaluate damage level.
            </p>
            <form className="vlm-demo-form" onSubmit={handleVlmDemoSubmit}>
              <label className="vlm-demo-field">
                <span>Pre-disaster image</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={(event) =>
                    setDemoPreImage(event.target.files?.[0] ?? null)
                  }
                />
              </label>
              <label className="vlm-demo-field">
                <span>Post-disaster image</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={(event) =>
                    setDemoPostImage(event.target.files?.[0] ?? null)
                  }
                />
              </label>
              <button
                className="vlm-demo-submit"
                type="submit"
                disabled={demoLoading}
              >
                {demoLoading ? "Evaluating..." : "Run VLM Evaluation"}
              </button>
            </form>
            {demoError ? <p className="vlm-demo-error">{demoError}</p> : null}
            {demoResult ? (
              <div className="vlm-demo-result">
                <div className="vlm-demo-result-row">
                  <span className="vlm-demo-label">Damage level</span>
                  <span className="vlm-demo-value">
                    {demoResult.model.damage_level ?? "Unavailable"}
                  </span>
                </div>
                <div className="vlm-demo-result-row">
                  <span className="vlm-demo-label">Confidence</span>
                  <span className="vlm-demo-value">
                    {demoResult.model.confidence ?? "Unavailable"}
                  </span>
                </div>
                <div className="vlm-demo-reasoning">
                  <span className="vlm-demo-label">Reasoning</span>
                  <p>{demoResult.model.reasoning ?? "No reasoning returned."}</p>
                </div>
              </div>
            ) : null}
          </section>
          <section className="panel evaluation-panel">
            <div className="panel-header">
              <h2>Evaluation Metrics</h2>
            </div>
            <div className="evaluation-metrics-grid">
              <div className="evaluation-metric-card">
                <span className="evaluation-metric-label">Accuracy</span>
                <strong className="evaluation-metric-value">
                  {(summaryData.accuracy * 100).toFixed(2)}%
                </strong>
              </div>
              <div className="evaluation-metric-card">
                <span className="evaluation-metric-label">Correct Predictions</span>
                <strong className="evaluation-metric-value">
                  {summaryData.matched.toLocaleString()}
                </strong>
              </div>
              <div className="evaluation-metric-card">
                <span className="evaluation-metric-label">Dataset Size</span>
                <strong className="evaluation-metric-value">
                  {summaryData.total.toLocaleString()}
                </strong>
              </div>
              <div className="evaluation-metric-card">
                <span className="evaluation-metric-label">Match Rate</span>
                <strong className="evaluation-metric-value">{evaluatedPct}%</strong>
              </div>
            </div>

            <div className="evaluation-stats-grid">
              <div className="evaluation-stats-card">
                <h3>Ground Truth Counts</h3>
                <ul className="evaluation-stats-list">
                  {EVALUATION_LABELS.map((label) => (
                    <li key={`gt-${label}`}>
                      <span>{label}</span>
                      <strong>
                        {(groundTruthCounts[label] ?? 0).toLocaleString()}
                      </strong>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="evaluation-stats-card">
                <h3>Prediction Counts</h3>
                <ul className="evaluation-stats-list">
                  {EVALUATION_LABELS.map((label) => (
                    <li key={`pred-${label}`}>
                      <span>{label}</span>
                      <strong>
                        {(predictionCounts[label] ?? 0).toLocaleString()}
                      </strong>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="evaluation-matrix-wrap">
              <table className="evaluation-matrix">
                <thead>
                  <tr>
                    <th>Actual \ Predicted</th>
                    {EVALUATION_LABELS.map((label) => (
                      <th key={`head-${label}`}>{label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {evaluationMatrix.map((row) => (
                    <tr key={row.groundTruth}>
                      <th>{row.groundTruth}</th>
                      {row.predictions.map((value, index) => (
                        <td key={`${row.groundTruth}-${EVALUATION_LABELS[index]}`}>
                          {value.toLocaleString()}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="evaluation-stats-card evaluation-confusions-card">
              <h3>Most Common Errors</h3>
              <ul className="evaluation-stats-list">
                {topConfusions.map(([key, count]) => (
                  <li key={key}>
                    <span>{key}</span>
                    <strong>{count.toLocaleString()}</strong>
                  </li>
                ))}
              </ul>
            </div>
          </section>
          <section className="chat-container">
            <div className="chat-header">
              <span className="chat-title">AI Assistant</span>
            </div>
            <div className="chat-messages">
              {chatMessages.map((message) => (
                <div
                  key={message.id}
                  className={`message-wrapper ${message.sender === "user" ? "user-wrapper" : "assistant-wrapper"}`}
                >
                  {/* Only show the AI icon for assistant messages */}
                  {message.sender === "assistant" && (
                    <div className="assistant-icon" title="AI Assistant">AI</div>
                  )}
                  
                  <div className={`chat-bubble ${message.sender}`}>
                    {message.text}
                  </div>
                </div>
              ))}
            </div>

            <div className="chat-input-area">
              {/* Refined Quick Actions to help with Capstone testing */}
              <div className="quick-actions">
                <button type="button" onClick={() => setChatInput("Analyze severe damage areas")}>
                  Analyze Severe Damage
                </button>
                <button type="button" onClick={() => setChatInput("Show overall building damage")}>
                  Show Overall Building Damage
                </button>
                <button type="button" onClick={() => setChatInput("Number of buildings affected")}>
                  Show Number of Buildings Affected
                </button>
              </div>

              <div className="chat-input-row">
                <input
                  className="chat-input"
                  type="text"
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Ask about damage patterns..."
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      handleSendChat();
                    }
                  }}
                />
                <button
                  className="chat-send-button"
                  type="button"
                  onClick={handleSendChat}
                >
                  Send
                </button>
              </div>
            </div>
          </section>
        </section>
      </main>

    
    </div>
  );
}

export default App;