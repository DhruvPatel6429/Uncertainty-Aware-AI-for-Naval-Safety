"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import { uqMethods } from "@/lib/data";

const dimensions = [
  { key: "coverage",    label: "Coverage" },
  { key: "calibration", label: "Calibration" },
  { key: "compute",     label: "Compute Eff." },
  { key: "ood",         label: "OOD Robust" },
  { key: "scalability", label: "Scalability" },
  { key: "rigor",       label: "Rigor" },
];

export default function UQRadarChart({ highlight }: { highlight?: string }) {
  const data = dimensions.map((d) => {
    const row: Record<string, number | string> = { dimension: d.label };
    uqMethods.forEach((m) => {
      // @ts-ignore — dynamic key access on metrics
      row[m.shortName] = m.metrics[d.key];
    });
    return row;
  });

  return (
    <div className="w-full h-[440px]">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} outerRadius="72%">
          <PolarGrid stroke="#E1E8F2" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fill: "#3A5680", fontSize: 11, fontWeight: 500 }}
          />
          <PolarRadiusAxis
            domain={[0, 5]}
            tick={{ fill: "#8FA3BF", fontSize: 10 }}
            stroke="#C2D0E2"
          />
          {uqMethods.map((m) => {
            const dim = highlight && highlight !== m.id ? 0.15 : 1;
            return (
              <Radar
                key={m.id}
                name={m.shortName}
                dataKey={m.shortName}
                stroke={m.color}
                fill={m.color}
                fillOpacity={0.12 * dim}
                strokeOpacity={dim}
                strokeWidth={2}
              />
            );
          })}
          <Legend
            wrapperStyle={{
              fontSize: 11,
              paddingTop: 8,
            }}
          />
          <Tooltip
            contentStyle={{
              fontSize: 11,
              borderRadius: 6,
              border: "1px solid #E1E8F2",
              fontFamily: "Inter, sans-serif",
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
