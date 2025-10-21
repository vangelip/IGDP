import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# Test with just one wind farm
TEST_BMU = 'HOWAO-1'
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def test_api_endpoint(bmu_unit, days_back=3):
    """Test the API and show what data structure we get back"""

    # Get data from 3 days ago to today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print("=" * 70)
    print(f"Testing BMRS API for BMU: {bmu_unit}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 70)

    endpoint = f"{BASE_URL}/balancing/physical"
    params = {
        'bmUnit': f"T_{bmu_unit}",
        'from': start_date.strftime('%Y-%m-%dT00:00Z'),
        'to': end_date.strftime('%Y-%m-%dT23:59Z'),
        'format': 'json'
    }

    print(f"\n1. REQUEST URL:")
    print(f"   {endpoint}")
    print(f"\n2. REQUEST PARAMETERS:")
    for key, value in params.items():
        print(f"   {key}: {value}")

    try:
        response = requests.get(endpoint, params=params, timeout=30)

        print(f"\n3. RESPONSE STATUS:")
        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"\n4. RESPONSE STRUCTURE:")
            print(f"   Top-level keys: {list(data.keys())}")

            if 'data' in data and data['data']:
                print(f"   Number of records: {len(data['data'])}")

                # Show first record structure
                print(f"\n5. FIRST RECORD (full structure):")
                first_record = data['data'][0]
                print(json.dumps(first_record, indent=2, default=str))

                # Convert to DataFrame
                df = pd.DataFrame(data['data'])

                print(f"\n6. DATAFRAME INFO:")
                print(f"   Shape: {df.shape}")
                print(f"\n   All Columns:")
                for i, col in enumerate(df.columns, 1):
                    non_null = df[col].notna().sum()
                    dtype = df[col].dtype
                    print(f"   {i:2d}. {col:25s} - {non_null:4d} non-null values, dtype: {dtype}")

                # Look for power-related columns
                print(f"\n7. POTENTIAL POWER COLUMNS:")
                power_keywords = ['level', 'pn', 'qpn', 'power', 'mw', 'generation',
                                  'output', 'quantity', 'mils', 'mels']
                found_power_cols = []

                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in power_keywords):
                        found_power_cols.append(col)
                        sample_values = df[col].dropna().head(5).tolist()
                        print(f"   • {col}")
                        print(f"     Sample values: {sample_values}")
                        print(f"     Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")

                if not found_power_cols:
                    print("   ⚠️  No obvious power-related columns found!")

                # Show sample of the data
                print(f"\n8. SAMPLE DATA (first 10 rows):")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                print(df.head(10).to_string())

                return df

            else:
                print("   ⚠️  No data returned in response")
                print(f"   Full response: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response text: {response.text[:500]}")

    except Exception as e:
        print(f"\n❌ ERROR occurred:")
        print(f"   {type(e).__name__}: {str(e)}")

    return None


def test_alternative_endpoint(bmu_unit, days_back=3):
    """Test the generation/outturn endpoint as fallback"""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print("\n" + "=" * 70)
    print("TESTING ALTERNATIVE ENDPOINT: generation/outturn")
    print("=" * 70)

    endpoint = f"{BASE_URL}/generation/outturn"
    params = {
        'settlementDateFrom': start_date.strftime('%Y-%m-%d'),
        'settlementDateTo': end_date.strftime('%Y-%m-%d'),
        'bmUnit': f"T_{bmu_unit}",
        'format': 'json'
    }

    print(f"\nREQUEST URL: {endpoint}")
    print(f"PARAMETERS: {params}")

    try:
        response = requests.get(endpoint, params=params, timeout=30)
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                print(f"Records returned: {len(df)}")
                print(f"\nColumns: {df.columns.tolist()}")
                print(f"\nSample data:")
                print(df.head(10).to_string())
                return df
            else:
                print("No data in response")
        else:
            print(f"Error response: {response.text[:500]}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

    return None


if __name__ == "__main__":
    print("\n" + "🔍 BMRS API DATA STRUCTURE TEST" + "\n")

    # Test main endpoint
    df1 = test_api_endpoint(TEST_BMU, days_back=3)

    # Test alternative endpoint
    df2 = test_alternative_endpoint(TEST_BMU, days_back=3)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    if df1 is not None:
        print("Primary endpoint (balancing/physical) returned data")
    else:
        print("Primary endpoint did not return data")

    if df2 is not None:
        print("Alternative endpoint (generation/outturn) returned data")
    else:
        print("Alternative endpoint did not return data")

    print("\nNext steps:")
    print("1. Check which columns contain the MW output data")
    print("2. Verify the data values look reasonable")
    print("3. Update the main script to use the correct column name")