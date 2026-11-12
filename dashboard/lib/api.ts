/**
 * API client for the UQ-AI Naval dashboard.
 *
 * Fetches live data from the FastAPI backend at http://localhost:8000.
 * Falls back to static data from lib/data.ts when the API is unavailable
 * (e.g., during static deployment or when the API is not running).
 */

import { useEffect, useState } from "react";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ApiStatus = {
  status: string;
  version: string;
  uq_methods: string[];
  pipeline_loaded: boolean;
};

export type ApiDecision = {
  action: string;
  collision_probability_bound: number;
  reject_reason: string | null;
  audit_trail: Record<string, unknown>;
  n_detections: number;
  mean_epistemic_uncertainty: number;
};

export type ApiAuditEntry = {
  entries: Array<Record<string, unknown>>;
};

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function getApiStatus(): Promise<ApiStatus | null> {
  return fetchJson<ApiStatus>(`${API_BASE_URL}/`);
}

export async function getUqMethods(): Promise<string[] | null> {
  const data = await fetchJson<{ methods: string[] }>(`${API_BASE_URL}/uq-methods`);
  return data?.methods ?? null;
}

export async function getAuditLog(): Promise<ApiAuditEntry | null> {
  return fetchJson<ApiAuditEntry>(`${API_BASE_URL}/audit-log`);
}

export async function predict(input: {
  sensor_type?: string;
  frame_data: number[];
  frame_shape: number[];
  own_state?: Record<string, number>;
}): Promise<ApiDecision | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sensor_type: input.sensor_type ?? "eo",
        frame_data: input.frame_data,
        frame_shape: input.frame_shape,
        own_state: input.own_state ?? {
          lat: 0.0,
          lon: 0.0,
          heading_deg: 0.0,
          speed_kn: 10.0,
        },
      }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) return null;
    return (await res.json()) as ApiDecision;
  } catch {
    return null;
  }
}

/**
 * Hook that checks if the API is reachable.
 * Returns null while loading, true/false once checked.
 */
export function useApiStatus(): boolean | null {
  const [status, setStatus] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    getApiStatus().then((s) => {
      if (mounted) setStatus(s?.status === "ok");
    });
    return () => {
      mounted = false;
    };
  }, []);

  return status;
}
