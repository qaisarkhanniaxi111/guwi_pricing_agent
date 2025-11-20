"""
Diagnostic Script - Check CSV Files and Columns
This helps identify why training failed for some services
"""

import pandas as pd
import os

# Expected files and columns
EXPECTED_FILES = {
    'train2_fix.csv': 'Gutter Clearing',
    'chemicaltrain.csv': 'Chemical Spray',
    'zinctrain.csv': 'Zinc Treatment',
    'exteriorwindowtrain.csv': 'Exterior Window Cleaning',
    'interiorwindowtrain.csv': 'Interior Window Cleaning'
}

print("=" * 80)
print("CSV FILES DIAGNOSTIC")
print("=" * 80)

print("\nChecking current directory:", os.getcwd())
print("\nLooking for CSV files...")
print("-" * 80)

all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"\nFound {len(all_files)} CSV files in folder:")
for f in all_files:
    print(f"  • {f}")

print("\n" + "=" * 80)
print("DETAILED FILE ANALYSIS")
print("=" * 80)

for csv_file, expected_column in EXPECTED_FILES.items():
    print(f"\n📄 File: {csv_file}")
    print("-" * 80)
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"  ❌ FILE NOT FOUND")
        print(f"  Action: Create or rename file to '{csv_file}'")
        continue
    
    try:
        # Try to load the file with different encodings
        df = None
        encoding_used = None
        
        for encoding in ['utf-8', 'windows-1252', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"  ❌ Could not read file with any common encoding")
            continue
            
        print(f"  ✅ File exists and readable (encoding: {encoding_used})")
        print(f"  📊 Records: {len(df)}")
        print(f"  📋 Columns: {len(df.columns)}")
        
        # Check for expected price column
        if expected_column in df.columns:
            print(f"  ✅ Price column '{expected_column}' found")
            
            # Check for valid prices
            prices = df[expected_column].dropna()
            if len(prices) > 0:
                print(f"  ✅ Valid prices: {len(prices)}")
                print(f"     Range: ${prices.min():.2f} - ${prices.max():.2f}")
                print(f"     Mean: ${prices.mean():.2f}")
            else:
                print(f"  ⚠️  No valid prices found in '{expected_column}'")
        else:
            print(f"  ❌ Price column '{expected_column}' NOT FOUND")
            print(f"  Available columns:")
            
            # Show all columns
            for i, col in enumerate(df.columns, 1):
                # Highlight columns that might be price columns
                if any(keyword in col.lower() for keyword in ['price', 'cost', 'clearing', 'spray', 'zinc', 'window', 'cleaning', 'treatment']):
                    print(f"     {i:2}. {col} ⭐ (might be price column)")
                else:
                    print(f"     {i:2}. {col}")
            
            # Suggest possible matches
            possible_matches = [col for col in df.columns 
                              if any(word in col.lower() for word in expected_column.lower().split())]
            if possible_matches:
                print(f"\n  💡 Possible matches:")
                for match in possible_matches:
                    print(f"     • '{match}'")
        
    except Exception as e:
        print(f"  ❌ ERROR reading file: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

missing_files = [f for f in EXPECTED_FILES.keys() if not os.path.exists(f)]
wrong_columns = []

for csv_file, expected_column in EXPECTED_FILES.items():
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if expected_column not in df.columns:
                wrong_columns.append((csv_file, expected_column))
        except:
            pass

if missing_files:
    print("\n❌ Missing Files:")
    for f in missing_files:
        print(f"   • {f}")
    print("\n   Action: Add these CSV files to your folder")
    print("   Or rename existing files to match these names")

if wrong_columns:
    print("\n❌ Wrong Column Names:")
    for csv_file, expected_col in wrong_columns:
        print(f"   • {csv_file} needs column '{expected_col}'")
    print("\n   Options:")
    print("   1. Rename columns in your CSV files")
    print("   2. Or update train_all_services.py with correct column names")

if not missing_files and not wrong_columns:
    print("\n✅ All files and columns look good!")
    print("   Run: python train_all_services.py")
else:
    print("\n" + "=" * 80)
    print("QUICK FIX OPTIONS")
    print("=" * 80)
    
    if missing_files:
        print("\nOption 1: Add Missing Files")
        print("  Place these CSV files in your folder:")
        for f in missing_files:
            print(f"    • {f}")
    
    if wrong_columns:
        print("\nOption 2: Fix Column Names")
        print("  Open each CSV and rename the price column to:")
        for csv_file, expected_col in wrong_columns:
            print(f"    • {csv_file} → '{expected_col}'")
        
        print("\nOption 3: Update Configuration")
        print("  Edit train_all_services.py and change 'price_column' to match your CSV")
        print("  Example:")
        print("    'price_column': 'Your Actual Column Name'")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)