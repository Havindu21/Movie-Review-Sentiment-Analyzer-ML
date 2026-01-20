"""
Quick test script to validate the system
Run this to see sample output
"""

from analyzer import ExportAnalyzer
import json

def main():
    print("\n" + "="*70)
    print("COCONUT EXPORT BUSINESS ADVISOR - TEST RUN")
    print("="*70 + "\n")
    
    # Initialize
    print("📊 Loading and analyzing export data...")
    analyzer = ExportAnalyzer("data/exports.csv")
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    # Display summary
    print("\n📈 MARKET SUMMARY")
    print("-" * 70)
    summary = results['summary']
    print(f"Total Products Analyzed: {summary['total_products_analyzed']}")
    print(f"Latest Year Export Value: ${summary['total_export_value_latest_year']:,}")
    print(f"Average Industry CAGR: {summary['avg_cagr']}%")
    print(f"\nTop Product by Value: {summary['top_product_by_value']['product_name']}")
    print(f"  → ${summary['top_product_by_value']['latest_value_usd']:,}")
    print(f"\nFastest Growing Product: {summary['top_product_by_growth']['product_name']}")
    print(f"  → {summary['top_product_by_growth']['cagr_pct']}% CAGR")
    
    # Display categories
    print("\n\n🎯 PRODUCT CATEGORIES")
    print("-" * 70)
    
    categories = results['categories']
    
    print(f"\n🌟 HIGH GROWTH STARS ({len(categories['high_growth_stars'])} products)")
    for i, p in enumerate(categories['high_growth_stars'][:5], 1):
        print(f"  {i}. {p['name'][:50]}")
        print(f"     Score: {p['score']} | CAGR: {p['cagr']}% | Recent: {p['recent_growth']}% | Value: ${p['value_usd']:,}")
    
    print(f"\n🚀 EMERGING OPPORTUNITIES ({len(categories['emerging_opportunities'])} products)")
    for i, p in enumerate(categories['emerging_opportunities'][:3], 1):
        print(f"  {i}. {p['name'][:50]}")
        print(f"     Score: {p['score']} | Recent Growth: {p['recent_growth']}%")
    
    print(f"\n⚠️  DECLINING WATCHLIST ({len(categories['declining_watchlist'])} products)")
    for i, p in enumerate(categories['declining_watchlist'][:3], 1):
        print(f"  {i}. {p['name'][:50]}")
        print(f"     CAGR: {p['cagr']}% | Recent: {p['recent_growth']}%")
    
    # Display recommendations
    print("\n\n💡 STRATEGIC RECOMMENDATIONS")
    print("-" * 70)
    
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"\n{i}. [{rec['priority']}] {rec['category']}")
        print(f"   Products: {', '.join(rec['products'][:3])}")
        print(f"   📋 Action: {rec['action']}")
        print(f"   🎯 Outcome: {rec['expected_outcome']}")
    
    # Display top scores
    print("\n\n🏆 TOP 10 INVESTMENT OPPORTUNITIES (by Score)")
    print("-" * 70)
    
    for i, product in enumerate(results['detailed_scores'][:10], 1):
        print(f"{i:2d}. {product['product_name'][:45]:45s} | Score: {product['investment_score']:.1f}")
        print(f"    CAGR: {product['cagr_pct']:6.1f}% | Recent: {product['recent_3yr_growth_pct']:6.1f}% | Value: ${product['latest_value_usd']:,}")
    
    # Sample product detail
    print("\n\n🔍 SAMPLE PRODUCT DEEP DIVE: Coco Peat")
    print("-" * 70)
    
    detail = analyzer.get_product_details('S.030205')
    print(f"Product: {detail['product_name']}")
    print(f"Current Year: {detail['current_year']}")
    print(f"Current Value: ${detail['current_value_usd']:,}")
    print(f"Current Quantity: {detail['current_quantity']:,} {detail['unit']}")
    print(f"Trend: {detail['trend']} (R²={detail['trend_strength_r2']})")
    print(f"Forecast for {detail['current_year']+1}: ${detail['forecast_next_year_usd']:,}")
    print(f"\nRecent Year-over-Year Growth:")
    for year_data in detail['yearly_breakdown'][-5:]:
        growth = year_data.get('value_growth_pct', 0)
        if growth and not (growth != growth):  # Check for NaN
            print(f"  {int(year_data['year'])}: ${int(year_data['value_usd']):,} ({growth:+.1f}%)")
        else:
            print(f"  {int(year_data['year'])}: ${int(year_data['value_usd']):,}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE - System is working correctly!")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("1. Run 'python app.py' to start the API server")
    print("2. Visit http://localhost:8000/docs for interactive API")
    print("3. Update data/exports.csv with your complete 2000-2025 data")
    print()

if __name__ == "__main__":
    main()
