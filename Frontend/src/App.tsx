import { useMemo, useState, useEffect } from "react";
import "./App.css";
import 'leaflet/dist/leaflet.css';
import Map from "./Map";
import summaryData from "../../Backend/data/santa_rosa/building_summary.json";
import { FloatingChatbot } from "./FloatingChatbot";

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

type BoundingBox = {
  building_id: string;
  subtype: string;
  bbox: [number, number, number, number];
};

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
  const [activeImageTab, setActiveImageTab] = useState<"before" | "after">(
    "after",
  );

  const [demoPreImage, setDemoPreImage] = useState<File | null>(null);
  const [demoPostImage, setDemoPostImage] = useState<File | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState("");
  const [demoResult, setDemoResult] = useState<VlmDemoResult | null>(null);

  const [mapLayer, setMapLayer] = useState<"pre" | "post">("pre");
  const [selectedTile, setSelectedTile] = useState<string>("00000000")

  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([]);
  useEffect(() => {
    fetch("/bounding_boxes.json")
      .then((r) => r.json())
      .then((data) => {
        console.log("all bounding boxes:", data.length);
        setBoundingBoxes(data);
      })
      .catch((err) => console.error("Failed to fetch bounding boxes:", err));
  }, []);

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

  const handleVlmDemoSubmit = async (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();

    if (!demoPreImage || !demoPostImage) {
      setDemoError(
        "Please upload both a pre-disaster image and a post-disaster image.",
      );
      return;
    }

    const formData = new FormData();
    formData.append("pre_image", demoPreImage);
    formData.append("post_image", demoPostImage);

    setDemoLoading(true);
    setDemoError("");
    setDemoResult(null);

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/vlm/assess-demo`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = (await response.json()) as VlmDemoResult;
      setDemoResult(data);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "The VLM evaluation request failed.";
      setDemoError(message);
    } finally {
      setDemoLoading(false);
    }
  };

  const groundTruthCounts = summaryData.ground_truth_counts as Partial<
    Record<EvaluationLabel, number>
  >;
  const predictionCounts = summaryData.prediction_counts as Partial<
    Record<EvaluationLabel, number>
  >;
  const confusionCounts = summaryData.confusion_counts as Record<
    string,
    number
  >;
  const evaluationMatrix = useMemo(
    () =>
      EVALUATION_LABELS.map((groundTruth) => ({
        groundTruth,
        predictions: EVALUATION_LABELS.map(
          (prediction) =>
            confusionCounts[`${groundTruth} -> ${prediction}`] ?? 0,
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
  const evaluatedPct = (
    (summaryData.matched / summaryData.total) *
    100
  ).toFixed(2);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1 className="app-title">Disaster Assessment Dashboard</h1>
          <p className="app-subtitle">Vision–Language Model Powered</p>
        </div>
        <div className="app-header-right">
          <span className="header-disaster-pill">
            Santa Rosa Wildfire Disaster
          </span>
        </div>
      </header>

      <main className="app-main">
        <FloatingChatbot />
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
                  src={`https://amzn-santa-rosa-wildfire-images.s3.us-east-1.amazonaws.com/datatsetCapstone/santa-rosa-wildfire_${selectedTile}_pre_disaster.png`}
                  alt="Before"
                  className="imagery-img"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              ) : (
                <img
                  src={`https://amzn-santa-rosa-wildfire-images.s3.us-east-1.amazonaws.com/datatsetCapstone/santa-rosa-wildfire_${selectedTile}_post_disaster.png`}
                  alt="After disaster"
                  className="imagery-img"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              )}
              <div className="imagery-label">
              </div>
            </div>
            <div className="imagery-details">
              <div className="imagery-meta">
                <div className="imagery-meta-title">Selected property</div>
                <div className="imagery-meta-damage">
                  Predicted damage:{" "}
                  <span>
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
                  {(
                    summaryData.prediction_counts["destroyed"] +
                    summaryData.prediction_counts["major-damage"]
                  ).toLocaleString()}
                </div>
                <div className="summary-label">Severe</div>
              </div>
              <div className="summary-card minor">
                <div className="summary-num">
                  {summaryData.prediction_counts[
                    "minor-damage"
                  ].toLocaleString()}
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
              <div className="map-layer-toggles">
                <label>
                  <input
                    type="checkbox"
                    checked={mapLayer == "pre"}
                    onChange={() => setMapLayer("pre")}
                  />
                  Pre-Disaster
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={mapLayer == "post"}
                    onChange={() => setMapLayer("post")}
                    />
                    Post-Disaster
                </label>
              </div>
              <Map mapLayer={mapLayer} boundingBoxes={boundingBoxes} damageFilter={damageFilter} onTileClick={(tileID) => setSelectedTile(tileID)}></Map>
            </div>
          </section>
          <section className="panel vlm-demo-panel">
            <div className="panel-header">
              <h2>VLM Evaluation</h2>
            </div>
            <p className="vlm-demo-copy">
              Upload a pre-disaster and post-disaster image pair to evaluate
              damage level.
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
                  <p>
                    {demoResult.model.reasoning ?? "No reasoning returned."}
                  </p>
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
                <span className="evaluation-metric-label">
                  Correct Predictions
                </span>
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
                <strong className="evaluation-metric-value">
                  {evaluatedPct}%
                </strong>
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
                        <td
                          key={`${row.groundTruth}-${EVALUATION_LABELS[index]}`}
                        >
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
        </section>
      </main>
    </div>
  );
}

export default App;
