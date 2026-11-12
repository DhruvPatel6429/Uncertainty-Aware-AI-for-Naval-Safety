"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Network,
  BrainCircuit,
  Workflow,
  AlertTriangle,
  FileCheck,
  Anchor,
  Radio,
  FlaskConical,
} from "lucide-react";
import { navItems } from "@/lib/data";

const iconMap: Record<string, React.ComponentType<{ size?: number | string; className?: string }>> = {
  LayoutDashboard,
  Network,
  BrainCircuit,
  Workflow,
  AlertTriangle,
  FileCheck,
  Radio,
  FlaskConical,
};

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-60 bg-navy-700 text-white flex-shrink-0 flex flex-col h-screen sticky top-0">
      {/* Brand */}
      <div className="px-5 py-5 border-b border-navy-600">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-md bg-gold-400 flex items-center justify-center flex-shrink-0">
            <Anchor size={18} className="text-navy-700" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-bold tracking-tight leading-tight">UQ-AI Naval</div>
            <div className="text-[10px] text-navy-300 uppercase tracking-[0.18em] mt-0.5">
              NRB · DRDO
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = iconMap[item.icon] ?? LayoutDashboard;
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-5 py-2 text-sm transition-colors relative group ${
                active
                  ? "bg-navy-800 text-white"
                  : "text-navy-200 hover:bg-navy-600 hover:text-white"
              }`}
            >
              {active && (
                <span className="absolute left-0 top-0 bottom-0 w-0.5 bg-gold-400" aria-hidden />
              )}
              <Icon size={16} className={active ? "text-gold-400" : "text-navy-300 group-hover:text-navy-100"} />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer status */}
      <div className="px-5 py-3 border-t border-navy-600 text-[10px] text-navy-300">
        <div className="text-navy-100 font-semibold uppercase tracking-wider mb-1">Project</div>
        <div>Proposal v1.0</div>
        <div className="mt-0.5 text-navy-400">Self-funded · 12 months</div>
      </div>
    </aside>
  );
}
