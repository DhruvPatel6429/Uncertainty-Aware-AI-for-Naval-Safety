import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";

export const metadata: Metadata = {
  title: "UQ-AI Naval — Research Dashboard for DRDO/NRB Review",
  description:
    "Uncertainty-Aware AI for Safety-Critical Naval Applications. Interactive research dashboard for NRB/DRDO review.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex">
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0">
            <Header />
            <main className="flex-1 p-6 md:p-8 overflow-x-hidden">{children}</main>
            <footer className="px-6 md:px-8 py-3 border-t border-navy-100 text-[10px] text-navy-400 flex items-center justify-between bg-white">
              <span>UQ-AI Naval · Uncertainty-Aware AI for Safety-Critical Naval Applications</span>
              <span>Self-funded research · NRB/DRDO submission · v1.0</span>
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
