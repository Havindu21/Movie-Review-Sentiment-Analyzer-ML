"""
Coconut Export Business Advisor - Analysis Engine
Medium-Advanced version with trend analysis, scoring, and ML-based insights
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExportAnalyzer:
    """Analyzes coconut export data and provides business recommendations"""
    
    def __init__(self, csv_path: str):
        """Load and prepare export data"""
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['value_usd'] > 0]  # Remove zero-value entries
        self.prepare_data()
        
    def prepare_data(self):
        """Calculate derived metrics for each product-year"""
        # Sort by product and year
        self.df = self.df.sort_values(['product_code', 'year'])
        
        # Calculate unit price (only for Kg products for fair comparison)
        self.df['unit_price_usd'] = np.where(
            self.df['quantity_unit'] == 'Kg',
            self.df['value_usd'] / self.df['quantity'],
            np.nan
        )
        
        # Calculate year-over-year growth rates
        self.df['value_growth_pct'] = self.df.groupby('product_code')['value_usd'].pct_change() * 100
        self.df['qty_growth_pct'] = self.df.groupby('product_code')['quantity'].pct_change() * 100
        self.df['price_growth_pct'] = self.df.groupby('product_code')['unit_price_usd'].pct_change() * 100
        
    def calculate_product_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive metrics for each product"""
        
        metrics = []
        
        for product_code in self.df['product_code'].unique():
            product_data = self.df[self.df['product_code'] == product_code].copy()
            
            if len(product_data) < 2:
                continue
                
            product_name = product_data['product_name'].iloc[0]
            unit = product_data['quantity_unit'].iloc[0]
            
            # Basic stats
            total_years = len(product_data)
            latest_value = product_data['value_usd'].iloc[-1]
            first_value = product_data['value_usd'].iloc[0]
            avg_value = product_data['value_usd'].mean()
            
            # CAGR (Compound Annual Growth Rate)
            n_years = total_years - 1
            cagr = ((latest_value / first_value) ** (1 / n_years) - 1) * 100 if first_value > 0 and n_years > 0 else 0
            
            # Recent trend (last 3 years average growth)
            recent_data = product_data.tail(3)
            recent_growth = recent_data['value_growth_pct'].mean()
            
            # Volatility (std dev of growth rates)
            volatility = product_data['value_growth_pct'].std()
            
            # ML-based trend strength using linear regression
            X = product_data['year'].values.reshape(-1, 1)
            y = product_data['value_usd'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            trend_slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # Unit price metrics (only for Kg products)
            avg_unit_price = product_data['unit_price_usd'].mean() if unit == 'Kg' else np.nan
            price_trend = product_data['price_growth_pct'].mean() if unit == 'Kg' else np.nan
            
            # Consistency score (based on r-squared and volatility)
            consistency_score = r_squared * (1 - min(volatility / 100, 1)) if not np.isnan(volatility) else 0
            
            # Market size
            market_share_estimate = latest_value  # Can be refined with total market data
            
            metrics.append({
                'product_code': product_code,
                'product_name': product_name,
                'unit': unit,
                'total_years': total_years,
                'latest_value_usd': latest_value,
                'avg_value_usd': avg_value,
                'cagr_pct': cagr,
                'recent_3yr_growth_pct': recent_growth,
                'volatility': volatility,
                'trend_slope': trend_slope,
                'trend_strength_r2': r_squared,
                'consistency_score': consistency_score,
                'avg_unit_price_usd': avg_unit_price,
                'price_trend_pct': price_trend,
                'market_value': market_share_estimate
            })
            
        return pd.DataFrame(metrics)
    
    def score_products(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score products based on multiple criteria
        Higher score = better investment opportunity
        """
        
        df = metrics_df.copy()
        
        # Normalize metrics for scoring (0-100 scale)
        scaler = StandardScaler()
        
        # Score components (each 0-20 points)
        
        # 1. Growth Score (CAGR + Recent Growth)
        df['growth_component'] = (
            np.clip(df['cagr_pct'] / 2, -10, 10) +  # CAGR contribution
            np.clip(df['recent_3yr_growth_pct'] / 2, -10, 10)  # Recent growth contribution
        )
        
        # 2. Stability Score (low volatility + high consistency)
        df['stability_component'] = (
            np.clip(20 - (df['volatility'].fillna(50) / 5), 0, 10) +  # Low volatility bonus
            df['consistency_score'] * 10  # Trend consistency
        )
        
        # 3. Market Size Score (larger markets = more opportunity)
        df['market_component'] = np.clip(
            (df['latest_value_usd'] / df['latest_value_usd'].max()) * 20, 0, 20
        )
        
        # 4. Price Power Score (unit price improvement)
        df['price_component'] = np.clip(
            df['price_trend_pct'].fillna(0) / 2, -10, 10
        )
        
        # 5. Trend Strength Score (R-squared from linear model)
        df['trend_component'] = df['trend_strength_r2'] * 20
        
        # Total Score (0-100)
        df['investment_score'] = (
            df['growth_component'] +
            df['stability_component'] +
            df['market_component'] +
            df['price_component'] +
            df['trend_component']
        )
        
        # Normalize to 0-100
        df['investment_score'] = np.clip(df['investment_score'], 0, 100)
        
        return df
    
    def categorize_products(self, scored_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Categorize products into strategic buckets"""
        
        df = scored_df.copy()
        
        categories = {
            'high_growth_stars': [],      # High growth + high score
            'stable_performers': [],       # Moderate growth + stable + large market
            'emerging_opportunities': [],  # Recent growth spike + improving
            'declining_watchlist': [],     # Negative growth
            'volatile_high_risk': [],      # High volatility
        }
        
        for _, row in df.iterrows():
            product_info = {
                'code': row['product_code'],
                'name': row['product_name'],
                'score': round(row['investment_score'], 1),
                'cagr': round(row['cagr_pct'], 1),
                'recent_growth': round(row['recent_3yr_growth_pct'], 1),
                'value_usd': int(row['latest_value_usd']),
                'volatility': round(row['volatility'], 1) if not np.isnan(row['volatility']) else 'N/A'
            }
            
            # Classification logic
            if row['cagr_pct'] > 10 and row['investment_score'] > 60:
                categories['high_growth_stars'].append(product_info)
            elif row['recent_3yr_growth_pct'] > 15 and row['cagr_pct'] < 10:
                categories['emerging_opportunities'].append(product_info)
            elif row['cagr_pct'] < -5 or row['recent_3yr_growth_pct'] < -10:
                categories['declining_watchlist'].append(product_info)
            elif row['volatility'] > 50:
                categories['volatile_high_risk'].append(product_info)
            else:
                categories['stable_performers'].append(product_info)
        
        # Sort each category by score
        for key in categories:
            categories[key] = sorted(categories[key], key=lambda x: x['score'], reverse=True)
        
        return categories
    
    def generate_recommendations(self, categories: Dict) -> List[Dict]:
        """Generate actionable business recommendations"""
        
        recommendations = []
        
        # Top growth opportunities
        if categories['high_growth_stars']:
            top_stars = categories['high_growth_stars'][:3]
            recommendations.append({
                'category': 'Scale Up - High Growth Stars',
                'priority': 'HIGH',
                'products': [p['name'] for p in top_stars],
                'action': 'Increase production capacity and marketing investment. These products show strong, consistent growth.',
                'expected_outcome': 'Maximize revenue growth in proven high-demand segments'
            })
        
        # Emerging opportunities
        if categories['emerging_opportunities']:
            emerging = categories['emerging_opportunities'][:3]
            recommendations.append({
                'category': 'Invest - Emerging Opportunities',
                'priority': 'MEDIUM-HIGH',
                'products': [p['name'] for p in emerging],
                'action': 'Pilot expansion with these recently accelerating products. Monitor market conditions closely.',
                'expected_outcome': 'Capture early market growth before competition intensifies'
            })
        
        # Stable performers
        if categories['stable_performers']:
            stable = sorted(categories['stable_performers'], key=lambda x: x['value_usd'], reverse=True)[:3]
            recommendations.append({
                'category': 'Maintain - Stable Revenue Base',
                'priority': 'MEDIUM',
                'products': [p['name'] for p in stable],
                'action': 'Maintain current operations. Optimize costs and explore efficiency improvements.',
                'expected_outcome': 'Reliable revenue stream with improved margins'
            })
        
        # Declining products
        if categories['declining_watchlist']:
            declining = categories['declining_watchlist'][:3]
            recommendations.append({
                'category': 'Review - Declining Products',
                'priority': 'URGENT',
                'products': [p['name'] for p in declining],
                'action': 'Investigate root causes: pricing issues, quality concerns, or market saturation. Consider pivoting or discontinuing.',
                'expected_outcome': 'Prevent further losses; reallocate resources to growth areas'
            })
        
        # High volatility products
        if categories['volatile_high_risk']:
            volatile = categories['volatile_high_risk'][:3]
            recommendations.append({
                'category': 'Caution - High Volatility',
                'priority': 'LOW',
                'products': [p['name'] for p in volatile],
                'action': 'Diversify risk. Don\'t over-invest. Use for portfolio balance only.',
                'expected_outcome': 'Minimize exposure to unpredictable markets'
            })
        
        return recommendations
    
    def get_product_details(self, product_code: str) -> Dict:
        """Get detailed analysis for a specific product"""
        
        product_data = self.df[self.df['product_code'] == product_code].copy()
        
        if product_data.empty:
            return {'error': 'Product not found'}
        
        product_name = product_data['product_name'].iloc[0]
        
        # Year-by-year breakdown
        yearly_data = product_data[['year', 'value_usd', 'quantity', 'value_growth_pct']].to_dict('records')
        
        # Latest metrics
        latest = product_data.iloc[-1]
        
        # Growth trajectory
        X = product_data['year'].values.reshape(-1, 1)
        y = product_data['value_usd'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast next year (simple linear extrapolation)
        next_year = product_data['year'].max() + 1
        forecast_value = model.predict([[next_year]])[0]
        
        return {
            'product_code': product_code,
            'product_name': product_name,
            'current_year': int(latest['year']),
            'current_value_usd': int(latest['value_usd']),
            'current_quantity': int(latest['quantity']),
            'unit': latest['quantity_unit'],
            'recent_growth_pct': round(latest['value_growth_pct'], 2),
            'forecast_next_year_usd': int(forecast_value),
            'yearly_breakdown': yearly_data,
            'trend': 'Upward' if model.coef_[0] > 0 else 'Downward',
            'trend_strength_r2': round(model.score(X, y), 3)
        }
    
    def run_full_analysis(self) -> Dict:
        """Execute complete analysis and return results"""
        
        # Calculate metrics
        metrics_df = self.calculate_product_metrics()
        
        # Score products
        scored_df = self.score_products(metrics_df)
        
        # Categorize
        categories = self.categorize_products(scored_df)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(categories)
        
        # Summary statistics
        summary = {
            'total_products_analyzed': len(scored_df),
            'total_export_value_latest_year': int(self.df.groupby('year')['value_usd'].sum().iloc[-1]),
            'avg_cagr': round(scored_df['cagr_pct'].mean(), 2),
            'top_product_by_value': scored_df.nlargest(1, 'latest_value_usd')[['product_name', 'latest_value_usd']].to_dict('records')[0],
            'top_product_by_growth': scored_df.nlargest(1, 'cagr_pct')[['product_name', 'cagr_pct']].to_dict('records')[0]
        }
        
        return {
            'summary': summary,
            'categories': categories,
            'recommendations': recommendations,
            'detailed_scores': scored_df.nlargest(10, 'investment_score')[
                ['product_name', 'investment_score', 'cagr_pct', 'recent_3yr_growth_pct', 'latest_value_usd']
            ].to_dict('records')
        }


# Quick test function
if __name__ == "__main__":
    analyzer = ExportAnalyzer("data/exports.csv")
    results = analyzer.run_full_analysis()
    
    print("\n=== COCONUT EXPORT BUSINESS ADVISOR ===\n")
    print(f"Total Products Analyzed: {results['summary']['total_products_analyzed']}")
    print(f"Latest Year Total Export Value: ${results['summary']['total_export_value_latest_year']:,}")
    print(f"\nAverage Industry CAGR: {results['summary']['avg_cagr']}%")
    
    print("\n=== TOP RECOMMENDATIONS ===")
    for rec in results['recommendations']:
        print(f"\n[{rec['priority']}] {rec['category']}")
        print(f"Products: {', '.join(rec['products'][:2])}")
        print(f"Action: {rec['action']}")
