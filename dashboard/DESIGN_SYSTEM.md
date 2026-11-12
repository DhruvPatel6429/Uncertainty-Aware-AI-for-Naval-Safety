# UQ-AI Naval Dashboard — Design System Reference

## Colour tokens (preserved from existing palette)

| Token | Hex | Usage |
|---|---|---|
| `navy-50` | `#F3F6FA` | Page background |
| `navy-100` | `#E1E8F2` | Borders, subtle dividers |
| `navy-200` | `#C2D0E2` | Disabled, secondary dividers |
| `navy-300` | `#8FA3BF` | Placeholder, secondary icons |
| `navy-400` | `#5C78A0` | Secondary text |
| `navy-500` | `#3A5680` | Sub-heading text |
| `navy-600` | `#1E4275` | Primary button hover |
| `navy-700` | `#0F2845` | Sidebar, primary identity |
| `navy-800` | `#0A1E36` | Hover state |
| `gold-400` | `#C9963A` | Accent, dividers, highlights |
| `gold-500` | `#B07E26` | Strong accent |
| `green-600` | `#3F8A5E` | Low-risk, success |
| `amber-600` | `#E0A53D` | Medium-risk, warning |
| `red-600`  | `#B5443A` | High-risk, danger |

## Typography scale

| Token | Size | Weight | Line | Usage |
|---|---|---|---|---|
| `text-display` | 32px | 700 | 1.15 | Hero title |
| `text-h1` | 24px | 700 | 1.25 | Page title |
| `text-h2` | 18px | 600 | 1.35 | Section title |
| `text-h3` | 15px | 600 | 1.4 | Card title |
| `text-body` | 14px | 400 | 1.5 | Body text |
| `text-small` | 13px | 400 | 1.5 | Secondary text |
| `text-mono-sm` | 12px | 500 | 1.4 | Mono, metrics |
| `text-label` | 11px | 600 | 1.3 | Uppercase labels |

## Spacing scale (4px base)

2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64

## Border radius

- `sm`: 4px — chips, badges
- `md`: 6px — buttons, inputs
- `lg`: 8px — cards, panels
- `xl`: 12px — large containers

## Scientific data provenance tags

Every metric or measured value must carry one of:
- **MEASURED** — produced by running the evaluation harness
- **ILLUSTRATIVE** — synthetic demonstration data
- **NOT EVALUATED** — proposed but not yet run

These tags must be visually distinct and never absent on metric displays.
