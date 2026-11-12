"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  Send,
  XCircle,
  Info,
  Server,
} from "lucide-react";
import {
  predict,
  useApiStatus,
  API_BASE_URL,
  type ApiDecision,
} from "@/lib/api";
import { type DataProvenance } from "@/lib/data";

const SYNTHETIC_FRAME = Array.from({ length: 16 }, () => Math.random() * 0.5 + 0.3);

export default function LiveDemoPage() {
  const apiStatus = useApiStatus();
  const [decision, setDecision] = useState<ApiDecision | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-fetch one prediction on mount (if API is online)
  useEffect(() => {
    if (apiStatus === true && !decision && !loading) {
      runPrediction();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiStatus]);

  async function runPrediction() {
    setLoading(true);
    setError(null);
    const result = await predict({
      sensor_type: "eo",
      frame_data: SYNTHETIC_FRAME,
      frame_shape: [4, 4],
      own_state: { lat: 0.0, lon: 0.0, heading_deg: 0.0, speed_kn: 10.0 },
    });
    if (result) {
      setDecision(result);
    } else {
      setError("API call failed — backend did not return a valid response. Check that the FastAPI service is running and reachable.");
    }
    setLoading(false);
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div>
        <h1 className="h-page">Live Pipeline Demo</h1>
        <p className="body-secondary mt-1">
          Executes the full UQ-AI Naval pipeline against the live FastAPI backend.
          Responses include the actual decision, audit trail, and collision
          probability bound produced by the running service.
        </p>
      </div>

      {/* Backend status */}
      <div className="surface p-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Status indicator */}
          <div>
            <div className="label mb-2">Backend Status</div>
            <div className="flex items-center gap-2">
              {apiStatus === null ? (
                <>
                  <Activity size={14} className="text-navy-400 animate-pulse" />
                  <span className="text-sm text-navy-500">Checking…</span>
                </>
              ) : apiStatus ? (
                <>
                  <CheckCircle2 size={14} className="text-green-600" />
                  <span className="text-sm text-green-700 font-medium">Online</span>
                </>
              ) : (
                <>
                  <XCircle size={14} className="text-amber-600" />
                  <span className="text-sm text-amber-700 font-medium">Offline</span>
                </>
              )}
            </div>
          </div>

          {/* Endpoint */}
          <div>
            <div className="label mb-2">Configured Endpoint</div>
            <div className="flex items-center gap-2">
              <Server size={14} className="text-navy-400" />
              <code className="text-xs text-navy-600 font-mono">{API_BASE_URL}</code>
            </div>
          </div>

          {/* Action */}
          <div className="flex flex-col items-end justify-center">
            <button
              onClick={runPrediction}
              disabled={apiStatus !== true || loading}
              className="btn-primary disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Send size={13} />
              {loading ? "Running…" : "Run Prediction"}
            </button>
          </div>
        </div>

        {/* Offline guidance */}
        {apiStatus === false && (
          <div className="mt-4 pt-4 border-t border-navy-100">
            <div className="rounded-md bg-amber-50 border border-amber-200 px-3 py-2.5 text-xs text-amber-800 flex items-start gap-2">
              <Info size={13} className="mt-0.5 flex-shrink-0" />
              <div className="leading-relaxed">
                <span className="font-semibold">Backend not reachable.</span> The dashboard
                cannot execute live pipeline predictions. To enable live predictions, start
                the FastAPI service:
                <pre className="mt-1.5 p-2 bg-white rounded text-[10px] font-mono text-navy-700 overflow-x-auto">
{`cd uq-naval
python scripts/serve.py --port 8000`}
                </pre>
                Then refresh this page. Until then, the{" "}
                <a href="/pipeline" className="underline">Pipeline Demo</a> page provides
                an explicitly labelled illustrative scenario.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 flex items-start gap-2">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-semibold mb-0.5">Request Failed</div>
            <div className="text-xs leading-relaxed">{error}</div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="surface p-8 flex items-center justify-center">
          <div className="flex items-center gap-2 text-navy-500 text-sm">
            <Activity size={14} className="animate-pulse" />
            Executing pipeline on backend…
          </div>
        </div>
      )}

      {/* Decision display */}
      {decision && !loading && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <DecisionCard decision={decision} />
            <UncertaintyCard decision={decision} />
            <DetectionsCard decision={decision} />
          </div>

          <div className="surface">
            <div className="px-6 py-3.5 border-b border-navy-100 flex items-center justify-between">
              <h2 className="h-card">Audit Trail</h2>
              <ProvenanceChip provenance="measured" label="Backend response" />
            </div>
            <div className="p-5">
              <pre className="text-[11px] font-mono text-navy-600 bg-navy-50 rounded-md p-4 overflow-x-auto leading-relaxed">
{JSON.stringify(decision.audit_trail, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Idle state with no decision yet */}
      {!decision && !loading && !error && apiStatus !== true && (
        <div className="surface">
          <div className="empty-state py-12">
            <div className="empty-state-icon">
              <Crosshair size={20} />
            </div>
            <div className="text-sm text-navy-600 font-medium mb-1">No live prediction yet</div>
            <div className="text-xs text-navy-400 max-w-sm">
              Start the backend service and click <strong>Run Prediction</strong> to
              execute the pipeline and inspect a real decision with its audit trail.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DecisionCard({ decision }: { decision: ApiDecision }) {
  const color =
    decision.action === "EXECUTE"
      ? "#3F8A5E"
      : decision.action === "REJECT_HUMAN"
      ? "#E0A53D"
      : "#B5443A";
  return (
    <div className="surface p-5">
      <div className="label mb-2">Decision</div>
      <div className="text-2xl font-bold mb-2" style={{ color }}>
        {decision.action}
      </div>
      {decision.reject_reason && (
        <div className="text-xs text-navy-500 mt-1 leading-relaxed">
          {decision.reject_reason}
        </div>
      )}
      <div className="mt-4 pt-3 border-t border-navy-50">
        <div className="label mb-1">Collision Probability Bound</div>
        <div className="font-mono text-sm font-bold text-navy-700">
          {decision.collision_probability_bound.toExponential(2)}
        </div>
        <div className="text-[10px] text-navy-400 mt-1 leading-snug">
          Statistical bound from Conformal Risk Control — conditional on calibration exchangeability.
        </div>
      </div>
    </div>
  );
}

function UncertaintyCard({ decision }: { decision: ApiDecision }) {
  const value = decision.mean_epistemic_uncertainty;
  return (
    <div className="surface p-5">
      <div className="label mb-2">Mean Epistemic Uncertainty</div>
      <div className="font-mono text-3xl font-bold text-navy-700">
        {value.toFixed(4)}
      </div>
      <div className="mt-3 h-1.5 rounded-full bg-navy-100 overflow-hidden">
        <div
          className="h-full rounded-full bg-navy-700"
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
      <div className="text-[10px] text-navy-400 mt-2 leading-snug">
        Epistemic uncertainty is the model&apos;s ignorance — reducible with more training data.
        Above 0.70 it triggers the reject option.
      </div>
    </div>
  );
}

function DetectionsCard({ decision }: { decision: ApiDecision }) {
  return (
    <div className="surface p-5">
      <div className="label mb-2">Detections</div>
      <div className="text-3xl font-bold text-navy-700">{decision.n_detections}</div>
      <div className="text-xs text-navy-500 mt-1">obstacles detected in frame</div>
    </div>
  );
}

function ProvenanceChip({
  provenance,
  label,
}: {
  provenance: DataProvenance;
  label?: string;
}) {
  const styles: Record<DataProvenance, string> = {
    "measured": "chip-measured",
    "illustrative": "chip-illustrative",
    "proposed": "chip-proposed",
    "not-evaluated": "chip-not-evaluated",
  };
  return (
    <span className={`chip border ${styles[provenance]}`}>
      {label || provenance}
    </span>
  );
}
