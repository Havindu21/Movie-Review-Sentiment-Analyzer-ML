"""
FastAPI Application for Coconut Export Business Advisor
Provides REST API endpoints for SME business insights
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
from analyzer import ExportAnalyzer

app = FastAPI(
    title="Coconut Export Business Advisor API",
    description="AI-driven insights for SME coconut export businesses",
    version="1.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "exports.csv")
analyzer = ExportAnalyzer(CSV_PATH)


class BusinessQuery(BaseModel):
    """Request model for business queries"""
    question: str
    context: Optional[str] = None


class ProductQuery(BaseModel):
    """Request model for specific product analysis"""
    product_code: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Coconut Export Business Advisor",
        "version": "1.0.0",
        "endpoints": {
            "/analysis": "Get full market analysis and recommendations",
            "/product/{code}": "Get detailed analysis for specific product",
            "/categories": "Get products grouped by opportunity categories",
            "/recommendations": "Get strategic recommendations only",
            "/top-products": "Get top-ranked investment opportunities"
        }
    }


@app.get("/analysis")
async def get_full_analysis():
    """
    Get complete business analysis with recommendations
    
    Returns:
    - Market summary statistics
    - Product categories (growth stars, emerging, stable, declining, volatile)
    - Strategic recommendations with priorities
    - Top-scored products
    """
    try:
        results = analyzer.run_full_analysis()
        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/categories")
async def get_categories():
    """
    Get products grouped by strategic categories
    
    Categories:
    - high_growth_stars: Strong consistent growth, high scores
    - emerging_opportunities: Recent acceleration
    - stable_performers: Reliable revenue
    - declining_watchlist: Negative trends
    - volatile_high_risk: Unpredictable performance
    """
    try:
        metrics_df = analyzer.calculate_product_metrics()
        scored_df = analyzer.score_products(metrics_df)
        categories = analyzer.categorize_products(scored_df)
        
        return {
            "status": "success",
            "categories": categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")


@app.get("/recommendations")
async def get_recommendations():
    """
    Get strategic business recommendations only
    
    Returns prioritized action items:
    - Scale Up opportunities
    - Investment opportunities
    - Maintain strategies
    - Review/exit recommendations
    """
    try:
        results = analyzer.run_full_analysis()
        
        return {
            "status": "success",
            "recommendations": results['recommendations'],
            "summary": results['summary']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@app.get("/top-products")
async def get_top_products(limit: int = 10):
    """
    Get top-ranked products by investment score
    
    Parameters:
    - limit: Number of top products to return (default: 10)
    """
    try:
        metrics_df = analyzer.calculate_product_metrics()
        scored_df = analyzer.score_products(metrics_df)
        
        top_products = scored_df.nlargest(limit, 'investment_score')[
            ['product_code', 'product_name', 'investment_score', 'cagr_pct', 
             'recent_3yr_growth_pct', 'latest_value_usd', 'volatility']
        ].to_dict('records')
        
        return {
            "status": "success",
            "count": len(top_products),
            "products": top_products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top products query failed: {str(e)}")


@app.get("/product/{product_code}")
async def get_product_details(product_code: str):
    """
    Get detailed analysis for a specific product
    
    Parameters:
    - product_code: Product code (e.g., S.030205)
    
    Returns:
    - Historical data
    - Growth metrics
    - Forecast
    - Trend analysis
    """
    try:
        details = analyzer.get_product_details(product_code)
        
        if 'error' in details:
            raise HTTPException(status_code=404, detail=details['error'])
        
        return {
            "status": "success",
            "product": details
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product analysis failed: {str(e)}")


@app.post("/ask")
async def ask_advisor(query: BusinessQuery):
    """
    Natural language query endpoint (basic version)
    
    Ask questions like:
    - "What products should I invest in?"
    - "Which products are declining?"
    - "What's the best stable product?"
    """
    try:
        question = query.question.lower()
        
        # Simple keyword matching (can be enhanced with LLM later)
        if any(word in question for word in ['invest', 'grow', 'scale', 'expand', 'best']):
            results = analyzer.run_full_analysis()
            stars = results['categories']['high_growth_stars'][:5]
            
            response = {
                "answer": "Based on export data analysis, here are the top investment opportunities:",
                "products": stars,
                "recommendation": "These products show strong CAGR and recent growth momentum."
            }
            
        elif any(word in question for word in ['declining', 'falling', 'avoid', 'worst']):
            results = analyzer.run_full_analysis()
            declining = results['categories']['declining_watchlist'][:5]
            
            response = {
                "answer": "These products show declining export performance:",
                "products": declining,
                "recommendation": "Review operations or consider pivoting resources to growth areas."
            }
            
        elif any(word in question for word in ['stable', 'safe', 'reliable', 'consistent']):
            results = analyzer.run_full_analysis()
            stable = sorted(
                results['categories']['stable_performers'], 
                key=lambda x: x['value_usd'], 
                reverse=True
            )[:5]
            
            response = {
                "answer": "These products offer stable, reliable revenue:",
                "products": stable,
                "recommendation": "Good for maintaining cash flow while investing in growth areas."
            }
            
        else:
            # Default: full summary
            results = analyzer.run_full_analysis()
            response = {
                "answer": "Here's a comprehensive analysis of the coconut export market:",
                "summary": results['summary'],
                "top_recommendations": results['recommendations'][:3]
            }
        
        return {
            "status": "success",
            "query": query.question,
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/market-overview")
async def get_market_overview():
    """
    Get high-level market statistics and trends
    """
    try:
        # Group by year to show market evolution
        yearly_totals = analyzer.df.groupby('year').agg({
            'value_usd': 'sum',
            'product_code': 'nunique'
        }).reset_index()
        
        yearly_totals.columns = ['year', 'total_export_value_usd', 'num_products']
        
        # Calculate market growth
        first_year_value = yearly_totals.iloc[0]['total_export_value_usd']
        latest_year_value = yearly_totals.iloc[-1]['total_export_value_usd']
        market_cagr = ((latest_year_value / first_year_value) ** (1 / (len(yearly_totals) - 1)) - 1) * 100
        
        return {
            "status": "success",
            "market_summary": {
                "total_years": len(yearly_totals),
                "current_year": int(yearly_totals.iloc[-1]['year']),
                "current_total_exports_usd": int(latest_year_value),
                "market_cagr_pct": round(market_cagr, 2),
                "total_product_lines": int(yearly_totals.iloc[-1]['num_products'])
            },
            "yearly_data": yearly_totals.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
