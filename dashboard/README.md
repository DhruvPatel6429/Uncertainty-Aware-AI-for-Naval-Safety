# UQ-AI Naval Dashboard

Interactive dashboard for **"Uncertainty-Aware AI for Safety-Critical Naval Applications"** — a software-based research proposal submitted to the Naval Research Board (NRB), DRDO.

## What this is

A Next.js 14 + Tailwind CSS dashboard that lets DRDO/NRB reviewers interactively explore the proposed framework. The dashboard mirrors the content of the accompanying 32-page proposal document (`UQ_Naval_Proposal.docx` / `.pdf`) but presents it as clickable, visual, and explorable.

## Features

7 pages covering the full proposal:

| Page | Purpose |
|---|---|
| **Overview** | Hero, key facts, problem & solution, deliverables |
| **Architecture** | End-to-end pipeline diagram + 4 stage detail cards |
| **UQ Methods** | Interactive radar chart comparing all 4 UQ techniques + per-method strengths/weaknesses/key papers + side-by-side comparison table |
| **Pipeline Demo** | Interactive maritime perception scene — click detections to inspect calibrated confidence, epistemic uncertainty, conformal coverage, reject option |
| **Roadmap** | 12-month Gantt chart with 5 milestones + deliverable acceptance criteria table |
| **Risk Matrix** | Interactive likelihood × impact heat map of 8 risks + complete risk register |
| **DRDO Submission** | Package contents, 4-step adoption pathway, benefits one-pager |

## Tech Stack

- **Next.js 14** (App Router)
- **React 18** + **TypeScript 5**
- **Tailwind CSS 3** with custom navy + gold theme matching the proposal
- **Recharts** for radar chart
- **lucide-react** for icons

## Quick Start

```bash
# Install dependencies
npm install

# Run development server (http://localhost:3000)
npm run dev

# Build production bundle
npm run build

# Start production server
npm start
```

## Project Structure

```
uq-naval-dashboard/
├── app/
│   ├── layout.tsx              # Root layout (sidebar + header)
│   ├── page.tsx                # Overview page
│   ├── globals.css             # Tailwind + custom styles
│   ├── architecture/page.tsx   # System architecture
│   ├── uq-methods/page.tsx     # 4 UQ methods comparison
│   ├── pipeline/page.tsx       # Interactive perception demo
│   ├── roadmap/page.tsx        # 12-month Gantt
│   ├── risks/page.tsx          # Risk matrix
│   └── submission/page.tsx     # DRDO submission package
├── components/
│   ├── Sidebar.tsx
│   ├── Header.tsx
│   ├── ArchitectureDiagram.tsx
│   ├── UQRadarChart.tsx
│   ├── GanttChart.tsx
│   ├── RiskMatrix.tsx
│   └── PerceptionDemo.tsx
└── lib/
    └── data.ts                 # Shared data (UQ methods, phases, risks, deliverables)
```

## Color Palette

| Token | Hex | Use |
|---|---|---|
| Navy 700 | `#0F2845` | Primary headings, sidebar |
| Navy 600 | `#1E4275` | Sub-headings, secondary navy |
| Gold 400 | `#C9963A` | Accent, dividers, milestones |
| Gold 500 | `#B07E26` | Strong accent |
| Green 600 | `#3F8A5E` | Success / low-risk indicators |
| Red 600 | `#B5443A` | Warning / high-risk indicators |

## License

Apache 2.0 — see proposal document for full IP terms.

## Author

Principal Investigator — Self-funded research effort, single investigator.

---

**Companion documents:** `UQ_Naval_Proposal.docx` and `UQ_Naval_Proposal.pdf` (32 pages, ~11,600 words).
