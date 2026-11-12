"use client";

import { ShieldCheck, Wifi, WifiOff, Loader2 } from "lucide-react";
import { useApiStatus, API_BASE_URL } from "@/lib/api";

export default function Header() {
  const apiStatus = useApiStatus();

  return (
    <header className="h-12 bg-white border-b border-navy-100 flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center gap-2">
        <ShieldCheck size={15} className="text-gold-500" />
        <span className="text-xs text-navy-600 font-medium">
          UQ-AI Naval — Research Dashboard for DRDO/NRB Review
        </span>
      </div>
      <div className="flex items-center gap-4 text-[11px] text-navy-500">
        {/* Backend status — honest representation */}
        <div className="flex items-center gap-1.5">
          {apiStatus === null ? (
            <>
              <Loader2 size={11} className="animate-spin text-navy-300" />
              <span className="text-navy-400">checking backend…</span>
            </>
          ) : apiStatus ? (
            <>
              <Wifi size={11} className="text-green-600" />
              <span className="text-green-700">Backend online</span>
            </>
          ) : (
            <>
              <WifiOff size={11} className="text-amber-600" />
              <span className="text-amber-700">Backend offline</span>
            </>
          )}
        </div>
        <span className="text-navy-300">|</span>
        <span>Self-funded · Single PI</span>
        <span className="px-2 py-0.5 bg-gold-50 text-gold-700 rounded text-[10px] font-medium border border-gold-200">
          For Review
        </span>
      </div>
    </header>
  );
}
