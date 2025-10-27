import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import calendar

WIND_FARMS = {
    'HOWAO-1': 400, 'HOWAO-2': 400, 'HOWAO-3': 400,
    'HOWBO-1': 440, 'HOWBO-2': 440, 'HOWBO-3': 440,
    'DBAWO-1': 302, 'DBAWO-2': 304, 'DBAWO-3': 303,
    'DBAWO-4': 153, 'DBAWO-5': 138
}

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def fetch_bmu_data(bmu_unit, start_date, end_date):
    """Fetch data from BMRS API for a specific BMU"""
    all_data = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=6), end_date)

        endpoint = f"{BASE_URL}/balancing/physical"
        params = {
            'bmUnit': f"T_{bmu_unit}",
            'from': current.strftime('%Y-%m-%dT00:00Z'),
            'to': chunk_end.strftime('%Y-%m-%dT23:59Z'),
            'format': 'json'
        }

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'])
                    df['bmUnit'] = bmu_unit
                    all_data.append(df)
        except Exception as e:
            print(f"Error fetching {bmu_unit} for {current}: {e}")

        current = chunk_end + timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    return pd.DataFrame()


def get_month_data(year, month):
    """Get data for all wind farms for a specific month"""
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(seconds=1)

    all_data = []
    for bmu, capacity in WIND_FARMS.items():
        print(f"Fetching {bmu}...")
        df = fetch_bmu_data(bmu, start, end)
        if not df.empty:
            df['capacity_mw'] = capacity
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Convert time columns
        for col in ['timeFrom', 'timeTo', 'settlementDate', 'startTime']:
            if col in combined.columns:
                combined[col] = pd.to_datetime(combined[col], errors='coerce')
        return combined
    return pd.DataFrame()


def extract_power_data(df):
    """Extract MW output from the dataframe - filter for PN (Physical Notification) only"""
    if df.empty:
        return df

    # The API returns multiple dataset types: PN, QPN, MELS, MILS
    # PN (Physical Notification) contains the actual MW output
    if 'dataset' in df.columns:
        print(f"  Filtering for PN data... (Original records: {len(df)})")
        df = df[df['dataset'] == 'PN'].copy()
        print(f"  After filtering: {len(df)} PN records")

    # Use levelFrom as the MW output (start of settlement period)
    # You could also use levelTo or average them
    if 'levelFrom' in df.columns:
        df['mw_output'] = pd.to_numeric(df['levelFrom'], errors='coerce')
        # Ensure non-negative values (wind farms should be positive generation)
        df['mw_output'] = df['mw_output'].clip(lower=0)
    else:
        print(f"Warning: No levelFrom column found. Available columns: {df.columns.tolist()}")
        df['mw_output'] = 0

    return df


def plot_monthly_mw_output(year, months):
    """Create plots showing MW output over time"""
    monthly_data = {}

    for month in months:
        df = get_month_data(year, month)
        if not df.empty:
            df = extract_power_data(df)
            monthly_data[month] = df

    if not monthly_data:
        print("No data found")
        return

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Time series of total MW output
    ax1 = fig.add_subplot(gs[0, :])
    for month in sorted(monthly_data.keys()):
        df = monthly_data[month]
        if 'timeFrom' in df.columns and 'mw_output' in df.columns:
            # Aggregate by timestamp
            timeseries = df.groupby('timeFrom')['mw_output'].sum().sort_index()
            ax1.plot(timeseries.index, timeseries.values,
                     label=calendar.month_name[month], alpha=0.7, linewidth=1)

    ax1.set_ylabel('Total Output (MW)', fontsize=11)
    ax1.set_title(f'Wind Farm Fleet Output - {year}', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average hourly output by month
    ax2 = fig.add_subplot(gs[1, 0])
    month_names = []
    avg_outputs = []

    for month in sorted(monthly_data.keys()):
        df = monthly_data[month]
        month_names.append(calendar.month_name[month][:3])
        avg_outputs.append(df['mw_output'].mean())

    ax2.bar(month_names, avg_outputs, color='steelblue', alpha=0.7)
    ax2.set_ylabel('Average Output (MW)', fontsize=10)
    ax2.set_title('Average Output by Month', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Output distribution (box plot)
    ax3 = fig.add_subplot(gs[1, 1])
    output_by_month = [monthly_data[m]['mw_output'].values for m in sorted(monthly_data.keys())]
    ax3.boxplot(output_by_month, labels=month_names)
    ax3.set_ylabel('Output (MW)', fontsize=10)
    ax3.set_title('Output Distribution by Month', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Individual farm comparison
    ax4 = fig.add_subplot(gs[2, 0])
    farm_groups = {'HOWAO': [], 'HOWBO': [], 'DBAWO': []}

    for month in sorted(monthly_data.keys()):
        df = monthly_data[month]
        month_avg = {}
        for group in farm_groups.keys():
            group_farms = [f for f in WIND_FARMS.keys() if group in f]
            group_data = df[df['bmUnit'].isin(group_farms)]
            month_avg[group] = group_data['mw_output'].mean()

        for group in farm_groups.keys():
            farm_groups[group].append(month_avg[group])

    x = np.arange(len(month_names))
    width = 0.25
    for i, (group, values) in enumerate(farm_groups.items()):
        ax4.bar(x + (i - 1) * width, values, width, label=group, alpha=0.8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(month_names)
    ax4.set_ylabel('Average Output (MW)', fontsize=10)
    ax4.set_title('Average Output by Farm Group', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Summary statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    summary_text = f"Summary Statistics for {year}\n" + "=" * 40 + "\n\n"
    summary_text += f"Total farms tracked: {len(WIND_FARMS)}\n"
    summary_text += f"Combined capacity: {sum(WIND_FARMS.values())} MW\n\n"

    for month in sorted(monthly_data.keys()):
        df = monthly_data[month]
        total_intervals = len(df)
        avg_output = df['mw_output'].mean()
        max_output = df['mw_output'].max()

        summary_text += f"{calendar.month_name[month]}:\n"
        summary_text += f"  Avg output: {avg_output:.1f} MW\n"
        summary_text += f"  Max output: {max_output:.1f} MW\n"
        summary_text += f"  Data points: {total_intervals:,}\n\n"

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Wind Farm Production Analysis - {year}',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.show()

    # Print column info from first dataset for debugging
    if monthly_data:
        first_df = monthly_data[list(monthly_data.keys())[0]]
        print("\nDataFrame columns found:", first_df.columns.tolist())
        print("\nFirst few rows of power data:")
        print(first_df[['bmUnit', 'mw_output']].head(10))


if __name__ == "__main__":
    year = 2025
    months = [3,4,5]
    plot_monthly_mw_output(year, months)