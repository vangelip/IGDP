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
        except:
            pass

        current = chunk_end + timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    # Fallback to alternative endpoint
    endpoint = f"{BASE_URL}/generation/outturn"
    params = {
        'settlementDateFrom': start_date.strftime('%Y-%m-%d'),
        'settlementDateTo': end_date.strftime('%Y-%m-%d'),
        'bmUnit': bmu_unit,
        'format': 'json'
    }

    try:
        response = requests.get(endpoint, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                df['bmUnit'] = bmu_unit
                return df
    except:
        pass

    return pd.DataFrame()


def get_month_data(year, month):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(seconds=1)

    all_data = []
    for bmu, capacity in WIND_FARMS.items():
        df = fetch_bmu_data(bmu, start, end)
        if not df.empty:
            df['capacity_mw'] = capacity
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        for col in ['timeFrom', 'timeTo', 'settlementDate', 'startTime']:
            if col in combined.columns:
                combined[col] = pd.to_datetime(combined[col], errors='coerce')
        return combined
    return pd.DataFrame()


def calculate_metrics(df):
    if df.empty:
        return {}

    metrics = {}
    power_col = None

    for col in ['levelFrom', 'quantity', 'power', 'generation', 'output', 'demand']:
        if col in df.columns and df[col].notna().any():
            power_col = col
            break

    if not power_col:
        return metrics

    for bmu in df['bmUnit'].unique():
        bmu_data = df[df['bmUnit'] == bmu].copy()
        bmu_data = bmu_data[bmu_data[power_col].notna()]

        if len(bmu_data) == 0:
            continue

        power_values = bmu_data[power_col].values
        if power_col == 'demand':
            power_values = -power_values
            power_values = np.maximum(power_values, 0)

        production_mwh = power_values.sum() * 0.5
        capacity = bmu_data['capacity_mw'].iloc[0]
        total_hours = len(bmu_data) * 0.5
        max_possible_mwh = capacity * total_hours

        metrics[bmu] = {
            'production_mwh': production_mwh,
            'capacity_mw': capacity,
            'capacity_factor': (production_mwh / max_possible_mwh * 100) if max_possible_mwh > 0 else 0,
            'average_output_mw': power_values.mean(),
            'max_output_mw': power_values.max(),
            'data_points': len(bmu_data)
        }

    return metrics


def plot_monthly_summary(year, months):
    monthly_results = {}

    for month in months:
        df = get_month_data(year, month)
        if not df.empty:
            metrics = calculate_metrics(df)
            if metrics:
                monthly_results[month] = metrics

    if not monthly_results:
        print("No data found")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Wind Farm Production - {year}')

    months_with_data = sorted(monthly_results.keys())
    month_labels = [calendar.month_name[m][:3] for m in months_with_data]

    # Total production
    total_monthly = [sum(m['production_mwh'] for m in monthly_results[month].values())
                     for month in months_with_data]

    ax1.plot(month_labels, total_monthly, 'o-', linewidth=2, markersize=8)
    ax1.set_ylabel('Total Production (MWh)')
    ax1.set_title('Monthly Production')
    ax1.grid(True, alpha=0.3)

    # Capacity factors by farm
    for bmu in WIND_FARMS.keys():
        cfs = []
        for month in months_with_data:
            if bmu in monthly_results[month]:
                cfs.append(monthly_results[month][bmu]['capacity_factor'])
            else:
                cfs.append(0)
        if any(cf > 0 for cf in cfs):
            ax2.plot(month_labels, cfs, 'o-', label=bmu, alpha=0.7)

    ax2.set_ylabel('Capacity Factor (%)')
    ax2.set_title('Capacity Factors')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Production by group
    groups = {'HOWAO': [], 'HOWBO': [], 'DBAWO': []}
    for month in months_with_data:
        for group in groups.keys():
            total = sum(m['production_mwh'] for bmu, m in monthly_results[month].items()
                        if group in bmu)
            groups[group].append(total)

    x = np.arange(len(months_with_data))
    width = 0.25
    for i, (group, values) in enumerate(groups.items()):
        ax3.bar(x + (i - 1) * width, values, width, label=group)

    ax3.set_xticks(x)
    ax3.set_xticklabels(month_labels)
    ax3.set_ylabel('Production (MWh)')
    ax3.set_title('Production by Farm Group')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary stats
    ax4.axis('off')
    summary_text = f"Summary for {year}\n\n"
    summary_text += f"Total farms: {len(WIND_FARMS)}\n"
    summary_text += f"Total capacity: {sum(WIND_FARMS.values())} MW\n\n"

    for month in months_with_data:
        total = sum(m['production_mwh'] for m in monthly_results[month].values())
        farms_reporting = sum(1 for m in monthly_results[month].values() if m['production_mwh'] > 0)
        summary_text += f"{calendar.month_name[month]}:\n"
        summary_text += f"  Production: {total:,.0f} MWh\n"
        summary_text += f"  Farms reporting: {farms_reporting}/{len(WIND_FARMS)}\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    year = 2025
    months = [7,8,9]
    plot_monthly_summary(year, months)
