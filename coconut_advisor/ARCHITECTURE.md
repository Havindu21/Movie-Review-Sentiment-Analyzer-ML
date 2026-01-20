# 📊 SYSTEM ARCHITECTURE DIAGRAM

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SME OWNER                                     │
│              "What product should I invest in?"                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WEB BROWSER / API CLIENT                          │
│              http://localhost:8000/recommendations                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI REST API (app.py)                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ GET /recommendations                                          │  │
│  │ GET /top-products                                             │  │
│  │ GET /product/{code}                                           │  │
│  │ POST /ask                                                     │  │
│  └─────────────────────────┬─────────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ANALYSIS ENGINE (analyzer.py)                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ExportAnalyzer Class                                          │  │
│  │  ├── prepare_data()         (CAGR, growth %, volatility)     │  │
│  │  ├── calculate_metrics()    (23 products × 8 metrics)         │  │
│  │  ├── score_products()       (5-component scoring)             │  │
│  │  ├── categorize_products()  (Growth/Stable/Declining)         │  │
│  │  ├── generate_recommendations() (Action items)                │  │
│  │  └── get_product_details()  (Deep dive + forecast)            │  │
│  └─────────────────────────┬─────────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ML MODELS (scikit-learn)                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Linear Regression     → Trend detection (R² score)            │  │
│  │ StandardScaler        → Normalize metrics for fair scoring    │  │
│  │ Statistical Functions → CAGR, volatility, mean, std dev       │  │
│  └─────────────────────────┬─────────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ exports.csv (pandas DataFrame)                                │  │
│  │  Columns:                                                     │  │
│  │   - year (2000-2025)                                          │  │
│  │   - product_code (S.030XXX)                                   │  │
│  │   - product_name                                              │  │
│  │   - quantity                                                  │  │
│  │   - quantity_unit (Kg, No, M2, L)                             │  │
│  │   - value_usd                                                 │  │
│  │   - source (EDB)                                              │  │
│  │                                                               │  │
│  │  Derived Columns (calculated):                               │  │
│  │   - unit_price_usd                                            │  │
│  │   - value_growth_pct                                          │  │
│  │   - qty_growth_pct                                            │  │
│  │   - price_growth_pct                                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Scoring Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Product Data                           │
│              (25 years × 1 product = 25 rows)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 1: Calculate Base Metrics                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ CAGR = ((Latest / First)^(1/years) - 1) × 100                │  │
│  │   Example: Coco Peat 10.2% CAGR                               │  │
│  │                                                               │  │
│  │ Recent Growth = Avg(last 3 years YoY growth)                 │  │
│  │   Example: Coco Peat +20.4%                                   │  │
│  │                                                               │  │
│  │ Volatility = StdDev(all YoY growth %)                        │  │
│  │   Example: Coco Peat 5.3 (low = stable)                      │  │
│  │                                                               │  │
│  │ Trend Strength (R²) = LinearRegression.score()               │  │
│  │   Example: Coco Peat 0.776 (strong upward trend)             │  │
│  │                                                               │  │
│  │ Unit Price Trend = Avg(price growth %)                       │  │
│  │   Example: Coco Peat +3.2% (improving margins)               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 2: Score Each Component (0-20 pts)                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Growth Score (20 pts max)                                     │  │
│  │   = CAGR/2 + Recent_Growth/2                                 │  │
│  │   Coco Peat: (10.2/2) + (20.4/2) = 15.3 pts                  │  │
│  │                                                               │  │
│  │ Stability Score (20 pts max)                                  │  │
│  │   = (20 - Volatility/5) + (R² × 10)                          │  │
│  │   Coco Peat: (20 - 5.3/5) + (0.776×10) = 16.9 pts            │  │
│  │                                                               │  │
│  │ Market Size Score (20 pts max)                                │  │
│  │   = (Latest_Value / Max_Value) × 20                          │  │
│  │   Coco Peat: (26.5M / 50M) × 20 = 10.6 pts                   │  │
│  │                                                               │  │
│  │ Price Power Score (20 pts max)                                │  │
│  │   = Price_Trend / 2                                          │  │
│  │   Coco Peat: 3.2 / 2 = 1.6 pts                               │  │
│  │                                                               │  │
│  │ Trend Strength Score (20 pts max)                             │  │
│  │   = R² × 20                                                  │  │
│  │   Coco Peat: 0.776 × 20 = 15.5 pts                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 3: Total Score (0-100)                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Investment Score = Sum of 5 components                        │  │
│  │                                                               │  │
│  │ Coco Peat Example:                                            │  │
│  │   Growth:       15.3 pts                                      │  │
│  │   Stability:    16.9 pts                                      │  │
│  │   Market:       10.6 pts                                      │  │
│  │   Price Power:   1.6 pts                                      │  │
│  │   Trend:        15.5 pts                                      │  │
│  │   ──────────────────                                          │  │
│  │   TOTAL:        60.9 pts  ← HIGH GROWTH STAR!                │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 4: Categorization Logic                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ IF score > 60 AND CAGR > 10%:                                 │  │
│  │    → HIGH GROWTH STAR ⭐                                      │  │
│  │                                                               │  │
│  │ ELSE IF recent_growth > 15% AND CAGR < 10%:                  │  │
│  │    → EMERGING OPPORTUNITY 🚀                                  │  │
│  │                                                               │  │
│  │ ELSE IF CAGR < -5% OR recent_growth < -10%:                  │  │
│  │    → DECLINING WATCHLIST ⚠️                                   │  │
│  │                                                               │  │
│  │ ELSE IF volatility > 50:                                      │  │
│  │    → VOLATILE/HIGH RISK ⚡                                    │  │
│  │                                                               │  │
│  │ ELSE:                                                         │  │
│  │    → STABLE PERFORMER ✅                                      │  │
│  │                                                               │  │
│  │ Coco Peat: 60.9 score + 10.2% CAGR → HIGH GROWTH STAR        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 5: Generate Recommendation                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Category: HIGH GROWTH STAR                                    │  │
│  │ Priority: HIGH                                                │  │
│  │ Product: Coco Peat Fiber Pith & Moulded products              │  │
│  │                                                               │  │
│  │ Action:                                                       │  │
│  │ "Increase production capacity and marketing investment.       │  │
│  │  This product shows strong, consistent growth."               │  │
│  │                                                               │  │
│  │ Expected Outcome:                                             │  │
│  │ "Maximize revenue growth in proven high-demand segment"       │  │
│  │                                                               │  │
│  │ Supporting Data:                                              │  │
│  │  - Export value grew from $16.3M (2000) → $26.5M (2005)      │  │
│  │  - Consistent YoY growth: 18-21% for 3 consecutive years     │  │
│  │  - Strong upward trend (R²=0.776)                            │  │
│  │  - Low volatility (5.3) = predictable revenue                │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OUTPUT: JSON Response                            │
│  {                                                                   │
│    "category": "Scale Up - High Growth Stars",                      │
│    "priority": "HIGH",                                               │
│    "products": ["Coco Peat Fiber Pith & Moulded products"],         │
│    "action": "Increase production capacity...",                      │
│    "expected_outcome": "Maximize revenue growth..."                  │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Example: Product Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Activated Carbon (High Growth Star)                │
├──────────────────────────────────────────────────────────────────────┤
│ Score: 61.7                                                          │
│                                                                      │
│ ████████████████████████████████████████████████████████████ 61.7   │
│                                                                      │
│ Breakdown:                                                           │
│  Growth:      [████████████████    ] 16.0 pts  (CAGR 10.1%)        │
│  Stability:   [██████████████████  ] 18.0 pts  (Low volatility)     │
│  Market:      [██████████████      ] 14.0 pts  ($28.5M value)       │
│  Price Power: [████                ]  4.0 pts  (+2.5% price trend)  │
│  Trend:       [██████████████████  ] 17.6 pts  (R²=0.88 strong)    │
│                                                                      │
│ Category: HIGH GROWTH STAR ⭐                                       │
│ Action: SCALE UP - Increase investment                              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                   Desiccated Coconut (Declining)                     │
├──────────────────────────────────────────────────────────────────────┤
│ Score: 38.2                                                          │
│                                                                      │
│ ████████████████████████████████                      38.2           │
│                                                                      │
│ Breakdown:                                                           │
│  Growth:      [          ] -2.0 pts  (CAGR -5.3%)  ⚠️              │
│  Stability:   [████      ]  4.0 pts  (High volatility)              │
│  Market:      [████████████████████] 20.0 pts  ($50M largest)       │
│  Price Power: [          ] -3.0 pts  (Declining prices)             │
│  Trend:       [██        ]  2.2 pts  (R²=0.11 weak)                 │
│                                                                      │
│ Category: DECLINING WATCHLIST ⚠️                                    │
│ Action: REVIEW - Investigate root causes or exit                     │
└──────────────────────────────────────────────────────────────────────┘

Why the difference?
┌────────────────────────────────────────────────────────────────────┐
│ Activated Carbon:                                                  │
│  ✅ Consistent growth (10-15% every year)                          │
│  ✅ Strong trend (R²=0.88 = very predictable)                      │
│  ✅ Low volatility (stable revenue)                                │
│  ✅ Improving prices (better margins)                              │
│  → Result: Clear investment opportunity                            │
│                                                                    │
│ Desiccated Coconut:                                                │
│  ❌ Declining for 2+ years (CAGR -5.3%)                            │
│  ❌ Weak trend (R²=0.11 = unpredictable)                           │
│  ❌ High volatility (risky)                                        │
│  ❌ Falling prices (margin pressure)                               │
│  → Result: Review or exit recommended                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## Forecasting Logic

```
┌─────────────────────────────────────────────────────────────────────┐
│              Simple Linear Forecast (Phase 1)                        │
│                                                                      │
│  Historical Data:                                                   │
│   2000: $16.3M  ●                                                   │
│   2001: $13.3M  ●                                                   │
│   2002: $15.2M   ●                                                  │
│   2003: $18.5M     ●                                                │
│   2004: $22.0M       ●                                              │
│   2005: $26.5M         ●                                            │
│                                                                      │
│  Linear Regression Fit:                                             │
│   Y = 11.2M + (2.1M × Year)                                         │
│   R² = 0.776 (strong fit)                                           │
│                                                                      │
│  Forecast 2006:                                                     │
│   = 11.2M + (2.1M × 6)                                              │
│   = $26.7M                                                          │
│                                                                      │
│  Confidence: High (R² > 0.7)                                        │
│                                                                      │
│  ────────────────────────────────────────────────────────────────   │
│                                                                      │
│  Future Enhancement (Phase 3):                                      │
│   - ARIMA: Capture seasonal patterns                                │
│   - LSTM: Non-linear growth curves                                  │
│   - Prophet: External events (COVID, policy changes)                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API Request/Response Examples

### Example 1: Get Top Products

**Request:**
```http
GET /top-products?limit=3
```

**Response:**
```json
{
  "status": "success",
  "count": 3,
  "products": [
    {
      "product_code": "S.030301",
      "product_name": "Activated Carbon",
      "investment_score": 61.7,
      "cagr_pct": 10.1,
      "recent_3yr_growth_pct": 15.5,
      "latest_value_usd": 28500000,
      "volatility": 12.3
    },
    {
      "product_code": "S.030205",
      "product_name": "Coco Peat Fiber Pith & Moulded products",
      "investment_score": 60.9,
      "cagr_pct": 10.2,
      "recent_3yr_growth_pct": 20.4,
      "latest_value_usd": 26500000,
      "volatility": 5.3
    },
    {
      "product_code": "S.030208",
      "product_name": "Coconut Husk Chips",
      "investment_score": 59.9,
      "cagr_pct": 23.3,
      "recent_3yr_growth_pct": 22.3,
      "latest_value_usd": 640000,
      "volatility": 18.7
    }
  ]
}
```

---

### Example 2: Ask Natural Language Question

**Request:**
```http
POST /ask
Content-Type: application/json

{
  "question": "Which products are declining?"
}
```

**Response:**
```json
{
  "status": "success",
  "query": "Which products are declining?",
  "response": {
    "answer": "These products show declining export performance:",
    "products": [
      {
        "code": "S.030102",
        "name": "Desiccated Coconut",
        "score": 38.2,
        "cagr": -5.3,
        "recent_growth": -10.2,
        "value_usd": 50000000
      }
    ],
    "recommendation": "Review operations or consider pivoting resources to growth areas."
  }
}
```

---

This visual guide helps explain the system flow from data input → ML processing → business recommendations! 📊
