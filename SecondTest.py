import requests
import pandas as pd
from datetime import datetime, timedelta

# Quick test with just one day of one wind farm
TEST_BMU = 'HOWAO-1'
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Get yesterday's data
end_date = datetime.now()
start_date = end_date - timedelta(days=1)

endpoint = f"{BASE_URL}/balancing/physical"
params = {
    'bmUnit': f"T_{TEST_BMU}",
    'from': start_date.strftime('%Y-%m-%dT00:00Z'),
    'to': end_date.strftime('%Y-%m-%dT23:59Z'),
    'format': 'json'
}

print(f"Fetching data for {TEST_BMU} from {start_date.date()} to {end_date.date()}")
print("=" * 70)

response = requests.get(endpoint, params=params, timeout=30)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data['data'])

    print(f"\n1. RAW DATA:")
    print(f"   Total records: {len(df)}")
    print(f"   Dataset types: {df['dataset'].value_counts().to_dict()}")

    print(f"\n2. ALL DATASET TYPES (sample from each):")
    for dataset_type in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_type]
        print(f"\n   {dataset_type} ({len(subset)} records):")
        print(f"   levelFrom range: {subset['levelFrom'].min()} - {subset['levelFrom'].max()} MW")
        print(f"   Sample: {subset[['timeFrom', 'levelFrom', 'levelTo']].head(3).to_string(index=False)}")

    # Filter for PN only
    pn_data = df[df['dataset'] == 'PN'].copy()
    pn_data['timeFrom'] = pd.to_datetime(pn_data['timeFrom'])
    pn_data = pn_data.sort_values('timeFrom')

    print(f"\n3. FILTERED PN DATA (Physical Notification - Actual Output):")
    print(f"   Records: {len(pn_data)}")
    print(f"   Output range: {pn_data['levelFrom'].min()} - {pn_data['levelFrom'].max()} MW")
    print(f"   Average output: {pn_data['levelFrom'].mean():.1f} MW")
    print(f"   Capacity: 400 MW")
    print(f"   Capacity factor: {(pn_data['levelFrom'].mean() / 400 * 100):.1f}%")

    print(f"\n4. TIMELINE OF OUTPUT (every 4 hours):")
    sample = pn_data[::8]  # Every 8th record (4 hours in half-hour periods)
    print(sample[['timeFrom', 'levelFrom', 'levelTo']].to_string(index=False))

    print("\n" + "=" * 70)
    print("SUCCESS: PN data extracted correctly!")
    print(
        f"This shows the actual MW output varying from {pn_data['levelFrom'].min()} to {pn_data['levelFrom'].max()} MW")

else:
    print(f"Error: {response.status_code}")
    print(response.text)