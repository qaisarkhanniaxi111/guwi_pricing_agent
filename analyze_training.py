"""
Training Data Analyzer
Helps identify issues with training data quality
"""

import pandas as pd
import numpy as np
import sys

def analyze_training_data(csv_path, price_column):
    """Analyze training data for potential issues"""
    
    print("=" * 80)
    print("TRAINING DATA ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {csv_path}")
    print(f"Price Column: {price_column}")
    
    # Load data
    print("\n1. Loading data...")
    try:
        df = None
        for encoding in ['utf-8', 'windows-1252', 'latin-1']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"   ✅ Loaded {len(df)} records (encoding: {encoding})")
                break
            except:
                continue
        
        if df is None:
            print("   ❌ Could not load file")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Check price column
    print(f"\n2. Checking price column '{price_column}'...")
    if price_column not in df.columns:
        print(f"   ❌ Column not found!")
        print(f"   Available columns: {df.columns.tolist()}")
        return
    
    prices = df[price_column]
    
    print(f"   ✅ Column found")
    print(f"\n   📊 Price Statistics:")
    print(f"      Total values: {len(prices)}")
    print(f"      Missing:      {prices.isna().sum()} ({prices.isna().sum()/len(prices)*100:.1f}%)")
    print(f"      Valid:        {prices.notna().sum()}")
    
    valid_prices = prices.dropna()
    
    if len(valid_prices) == 0:
        print("   ❌ No valid prices!")
        return
    
    print(f"\n   💰 Price Range:")
    print(f"      Min:    ${valid_prices.min():.2f}")
    print(f"      Max:    ${valid_prices.max():.2f}")
    print(f"      Mean:   ${valid_prices.mean():.2f}")
    print(f"      Median: ${valid_prices.median():.2f}")
    print(f"      Std:    ${valid_prices.std():.2f}")
    
    # Check for outliers
    q1 = valid_prices.quantile(0.25)
    q3 = valid_prices.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = valid_prices[(valid_prices < lower_bound) | (valid_prices > upper_bound)]
    
    if len(outliers) > 0:
        print(f"\n   ⚠️  Outliers detected: {len(outliers)} ({len(outliers)/len(valid_prices)*100:.1f}%)")
        print(f"      Values: {sorted(outliers.values)}")
    else:
        print(f"\n   ✅ No extreme outliers")
    
    # Price distribution
    print(f"\n   📊 Price Distribution:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = valid_prices.quantile(p/100)
        print(f"      {p}th percentile: ${val:.2f}")
    
    # Check feature columns
    print(f"\n3. Checking Feature Columns...")
    
    important_features = [
        'Home Square Footage',
        'Home Value', 
        'Average Home Value in Zip code',
        'Average Home value in Zip code',
        'Number of Stories',
        'Roof Type',
        'Roof Type/ Material',
        'Roof Details >> Roof: >> Steepness'
    ]
    
    available_features = []
    missing_features = []
    
    for feat in important_features:
        if feat in df.columns:
            available_features.append(feat)
            missing_count = df[feat].isna().sum()
            missing_pct = missing_count / len(df) * 100
            
            if missing_pct > 50:
                print(f"   ❌ {feat}: {missing_pct:.1f}% missing - VERY HIGH!")
            elif missing_pct > 20:
                print(f"   ⚠️  {feat}: {missing_pct:.1f}% missing")
            elif missing_pct > 0:
                print(f"   ⚠️  {feat}: {missing_pct:.1f}% missing")
            else:
                print(f"   ✅ {feat}: All present")
        else:
            missing_features.append(feat)
    
    if missing_features:
        print(f"\n   ❌ Missing feature columns:")
        for feat in missing_features:
            print(f"      • {feat}")
    
    # Check correlations
    print(f"\n4. Feature Correlations with Price...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = []
    
    for col in numeric_cols:
        if col != price_column and df[col].notna().sum() > 0:
            corr = df[[col, price_column]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n   Top correlations:")
    for col, corr in correlations[:10]:
        emoji = "✅" if abs(corr) > 0.3 else "⚠️" if abs(corr) > 0.1 else "❌"
        print(f"   {emoji} {col:<50} {corr:>7.3f}")
    
    if correlations and abs(correlations[0][1]) < 0.3:
        print(f"\n   ❌ WARNING: All correlations are weak (< 0.3)")
        print(f"      This means features don't predict price well!")
    
    # Data quality score
    print(f"\n" + "=" * 80)
    print("DATA QUALITY SCORE")
    print("=" * 80)
    
    score = 0
    max_score = 0
    issues = []
    
    # Check 1: Enough data
    max_score += 20
    if len(valid_prices) >= 1000:
        score += 20
        print("✅ Sample size: GOOD (1000+)")
    elif len(valid_prices) >= 500:
        score += 15
        print("⚠️  Sample size: OK (500+)")
    elif len(valid_prices) >= 200:
        score += 10
        print("⚠️  Sample size: LOW (200+)")
    else:
        print("❌ Sample size: TOO SMALL (<200)")
        issues.append("Need more training data (aim for 1000+)")
    
    # Check 2: Missing data
    max_score += 20
    missing_pct = prices.isna().sum() / len(prices) * 100
    if missing_pct < 5:
        score += 20
        print("✅ Missing prices: GOOD (<5%)")
    elif missing_pct < 20:
        score += 10
        print("⚠️  Missing prices: OK (<20%)")
    else:
        print(f"❌ Missing prices: HIGH ({missing_pct:.1f}%)")
        issues.append("Too many missing prices")
    
    # Check 3: Feature availability
    max_score += 20
    feature_score = len(available_features) / len(important_features) * 20
    score += feature_score
    if feature_score >= 15:
        print("✅ Features: GOOD")
    elif feature_score >= 10:
        print("⚠️  Features: OK (some missing)")
    else:
        print("❌ Features: POOR (many missing)")
        issues.append("Missing important features")
    
    # Check 4: Correlations
    max_score += 20
    if correlations and abs(correlations[0][1]) > 0.4:
        score += 20
        print("✅ Correlations: STRONG")
    elif correlations and abs(correlations[0][1]) > 0.2:
        score += 10
        print("⚠️  Correlations: MODERATE")
    else:
        print("❌ Correlations: WEAK")
        issues.append("Features don't predict price well - add better features!")
    
    # Check 5: Data consistency
    max_score += 20
    if len(outliers) / len(valid_prices) < 0.05:
        score += 20
        print("✅ Outliers: MINIMAL")
    elif len(outliers) / len(valid_prices) < 0.10:
        score += 10
        print("⚠️  Outliers: SOME")
    else:
        print("❌ Outliers: MANY")
        issues.append("Many outliers - clean data")
    
    print(f"\n📊 Overall Score: {score}/{max_score} ({score/max_score*100:.0f}%)")
    
    if score >= 80:
        print("🎉 EXCELLENT data quality!")
    elif score >= 60:
        print("✅ GOOD data quality")
    elif score >= 40:
        print("⚠️  FAIR data quality - improvements needed")
    else:
        print("❌ POOR data quality - significant improvements needed")
    
    # Recommendations
    if issues:
        print(f"\n" + "=" * 80)
        print("RECOMMENDED ACTIONS")
        print("=" * 80)
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_training.py <csv_file> <price_column>")
        print("\nExamples:")
        print("  python analyze_training.py train2.csv 'Gutter Clearing'")
        print("  python analyze_training.py exteriorwindowtrain.csv 'Exterior Window Cleaning'")
        print("  python analyze_training.py chemicaltrain.csv 'Chemical Spray'")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    price_col = sys.argv[2]
    
    analyze_training_data(csv_file, price_col)