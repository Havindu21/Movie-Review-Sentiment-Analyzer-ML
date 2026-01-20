# Coconut Export Business Advisor 🥥

AI-driven business growth assistant for SME coconut exporters using export trend analysis and machine learning.

## 🎯 Purpose

Help small coconut export businesses identify growth opportunities, avoid declining products, and make data-driven investment decisions based on 25+ years of Sri Lankan export statistics (2000-2025).

## 📊 What It Does

The system analyzes export data and provides:

1. **Product Scoring** - Ranks products by investment potential (0-100 score)
2. **Growth Metrics** - CAGR, recent trends, volatility, market size
3. **Strategic Categories**:
   - 🌟 High Growth Stars (scale up)
   - 🚀 Emerging Opportunities (early investment)
   - ✅ Stable Performers (maintain)
   - ⚠️ Declining Watchlist (review/exit)
   - ⚡ Volatile/High Risk (caution)
4. **Actionable Recommendations** - Prioritized action items with expected outcomes
5. **Product Forecasts** - Simple linear projections for next year

## 🏗️ Architecture

**Medium-Advanced Features:**
- Multi-factor scoring system (growth, stability, market size, pricing, trend strength)
- Linear regression for trend detection and forecasting
- Volatility analysis (risk assessment)
- CAGR calculations
- R² scores for trend reliability
- Normalized scoring across different metrics

## 📁 Project Structure

```
coconut_advisor/
├── data/
│   └── exports.csv          # Clean export data (2000-2025)
├── analyzer.py              # Core analysis engine
├── app.py                   # FastAPI REST API
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd coconut_advisor
pip install -r requirements.txt
```

### 2. Test the Analyzer (Command Line)

```bash
python analyzer.py
```

Expected output:
```
=== COCONUT EXPORT BUSINESS ADVISOR ===

Total Products Analyzed: 28
Latest Year Total Export Value: $XXX,XXX,XXX

Average Industry CAGR: X.X%

=== TOP RECOMMENDATIONS ===

[HIGH] Scale Up - High Growth Stars
Products: Coco Peat Fiber Pith & Moulded products, Activated Carbon
Action: Increase production capacity and marketing investment...
```

### 3. Start the API Server

```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --port 8000
```

Server runs at: **http://localhost:8000**

### 4. Access Interactive API Docs

Open in browser: **http://localhost:8000/docs**

## 🔌 API Endpoints

### GET `/analysis`
Complete business analysis with all recommendations

```json
{
  "status": "success",
  "data": {
    "summary": {...},
    "categories": {...},
    "recommendations": [...],
    "detailed_scores": [...]
  }
}
```

### GET `/recommendations`
Strategic recommendations only

### GET `/top-products?limit=10`
Top-ranked investment opportunities

### GET `/product/{product_code}`
Detailed analysis for specific product
Example: `/product/S.030205`

### GET `/categories`
Products grouped by opportunity type

### POST `/ask`
Natural language queries (basic keyword matching)

```json
{
  "question": "What products should I invest in?"
}
```

### GET `/market-overview`
High-level market statistics and yearly trends

## 📋 CSV Format

**Required columns:**
```csv
year,product_code,product_name,quantity,quantity_unit,value_usd,source
2000,S.030205,Coco Peat Fiber Pith & Moulded products,75631947,Kg,16300003,EDB
```

- `year`: 2000-2025
- `product_code`: Unique identifier (e.g., S.030205)
- `product_name`: Human-readable name
- `quantity`: Export volume
- `quantity_unit`: Kg, No, M2, L
- `value_usd`: Export value in USD
- `source`: Data source (e.g., EDB)

**Important:** Use "long/tidy" format (one row = one product-year)

## 🧮 Scoring Algorithm

Each product gets a 0-100 score based on:

1. **Growth Component (20 pts)**: CAGR + Recent 3-year growth
2. **Stability Component (20 pts)**: Low volatility + high trend consistency
3. **Market Size Component (20 pts)**: Latest export value (relative)
4. **Price Power Component (20 pts)**: Unit price improvement trend
5. **Trend Strength Component (20 pts)**: R² from linear regression

Higher score = better investment opportunity

## 💡 Example Use Cases

### For SME Owner:
*"I produce coconut oil. Should I expand or diversify?"*

1. Check `/product/S.030101` (Coconut Oil)
2. View its score and trend
3. Compare with `/top-products` to see alternatives
4. Review `/recommendations` for strategic guidance

### For New Entrant:
*"Which coconut product should I start with?"*

1. GET `/analysis` → view `high_growth_stars`
2. Filter by market size (large = established demand)
3. Check volatility (low = predictable)

### For Portfolio Manager:
*"Which products should I phase out?"*

1. GET `/categories` → view `declining_watchlist`
2. Check negative CAGR products
3. Review recommendations for exit/pivot strategies

## 🔧 How to Add Your Own Data

1. **Format your data** as CSV (see format above)
2. **Replace** `data/exports.csv` with your file
3. **Restart** the analyzer/API
4. Data from 2000-2025 works best (minimum 3-4 years per product)

## 🚀 Next Steps (Future Enhancements)

- [ ] Add LLM for natural language chat (Phase 3)
- [ ] Time series forecasting (ARIMA/LSTM)
- [ ] Price vs. quantity trade-off analysis
- [ ] Market diversification recommendations
- [ ] Export destination analysis
- [ ] Seasonal pattern detection
- [ ] Web dashboard with charts
- [ ] Add Kithul and Palmyra data
- [ ] Local market data integration

## 📝 Notes

- **Current version**: Export data only (Phase 1)
- **"Training"**: No ML model training needed; runs instantly
- **SME-friendly**: Clear categories, simple actions, no jargon
- **Scalable**: Easy to add new products/years/domains

## 🤝 For Your Group Project

**You've completed:**
- ✅ Data structure definition
- ✅ Medium-advanced analytics engine
- ✅ REST API for integration
- ✅ Business recommendations

**Next phases:**
- Phase 2: Add more data sources (local markets, prices, production)
- Phase 3: Fine-tune LLM for conversational advisor
- Phase 4: Web UI with dashboards
- Phase 5: Automated marketing content generation

---

**Built for**: CSCI 23072 Group Project - AI-Driven Business Growth Assistant for SMEs
**Team**: Group 7 | University of Kelaniya
