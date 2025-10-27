import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import calendar

# Only Hornsea wind farms - removed Dogger Bank
WIND_FARMS = {
    'HOWAO-1': 400, 'HOWAO-2': 400, 'HOWAO-3': 400,  # Hornsea 1
    'HOWBO-1': 440, 'HOWBO-2': 440, 'HOWBO-3': 440,  # Hornsea 2
}

# Group farms by project
HORNSEA_1 = ['HOWAO-1', 'HOWAO-2', 'HOWAO-3']
HORNSEA_2 = ['HOWBO-1', 'HOWBO-2', 'HOWBO-3']

HORNSEA_1_CAPACITY = sum([WIND_FARMS[f] for f in HORNSEA_1])  # 1200 MW
HORNSEA_2_CAPACITY = sum([WIND_FARMS[f] for f in HORNSEA_2])  # 1320 MW

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
    """Get data for Hornsea wind farms for a specific month"""
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

    # Filter for PN data only
    if 'dataset' in df.columns:
        df = df[df['dataset'] == 'PN'].copy()

    # Use levelFrom as the MW output
    if 'levelFrom' in df.columns:
        df['mw_output'] = pd.to_numeric(df['levelFrom'], errors='coerce')
        df['mw_output'] = df['mw_output'].clip(lower=0)
    else:
        df['mw_output'] = 0

    return df


def filter_maintenance_periods(df):
    """
    Remove data points where one farm is at 0 while all others are producing.
    This indicates maintenance rather than genuine low wind conditions.
    """
    if df.empty or 'timeFrom' not in df.columns:
        return df

    # Group by timestamp
    time_groups = df.groupby('timeFrom')

    valid_timestamps = []

    for timestamp, group in time_groups:
        # Get output for each farm
        farm_outputs = {}
        for farm in WIND_FARMS.keys():
            farm_data = group[group['bmUnit'] == farm]
            if not farm_data.empty:
                farm_outputs[farm] = farm_data['mw_output'].values[0]
            else:
                farm_outputs[farm] = 0

        # Check Hornsea 1 farms
        h1_outputs = [farm_outputs.get(f, 0) for f in HORNSEA_1]
        h1_zeros = sum([1 for x in h1_outputs if x == 0])
        h1_producing = sum([1 for x in h1_outputs if x > 0])

        # Check Hornsea 2 farms
        h2_outputs = [farm_outputs.get(f, 0) for f in HORNSEA_2]
        h2_zeros = sum([1 for x in h2_outputs if x == 0])
        h2_producing = sum([1 for x in h2_outputs if x > 0])

        # Keep data if:
        # - All farms are producing something, OR
        # - All farms are at zero (genuine no-wind situation), OR
        # - Multiple farms are at zero (not just one)
        # Remove data if exactly one farm is at zero while others produce (maintenance)
        h1_valid = not (h1_zeros == 1 and h1_producing > 0)
        h2_valid = not (h2_zeros == 1 and h2_producing > 0)

        if h1_valid and h2_valid:
            valid_timestamps.append(timestamp)

    # Filter dataframe to keep only valid timestamps
    filtered_df = df[df['timeFrom'].isin(valid_timestamps)].copy()

    print(f"  Filtered out {len(df) - len(filtered_df)} records (potential maintenance periods)")

    return filtered_df


def calculate_capacity_factor(df, farm_group, total_capacity):
    """Calculate capacity factor as % of nominal capacity for a farm group"""
    if df.empty:
        return pd.DataFrame()

    # Filter for farms in this group
    group_df = df[df['bmUnit'].isin(farm_group)].copy()

    if group_df.empty:
        return pd.DataFrame()

    # Aggregate by timestamp - sum all farms in group
    timeseries = group_df.groupby('timeFrom')['mw_output'].sum().sort_index()

    # Calculate capacity factor as percentage
    cf_series = (timeseries / total_capacity) * 100

    result_df = pd.DataFrame({
        'timestamp': cf_series.index,
        'output_mw': timeseries.values,
        'capacity_factor_pct': cf_series.values
    })

    # Filter out erroneous data points where CF > 100%
    initial_count = len(result_df)
    result_df = result_df[result_df['capacity_factor_pct'] <= 100].copy()
    filtered_count = initial_count - len(result_df)
    if filtered_count > 0:
        print(f"  Removed {filtered_count} erroneous data points with CF > 100%")

    return result_df


def find_below_average_periods(df, avg_cf, total_capacity):
    """
    Find largest continuous periods where output is below average capacity factor.
    Calculate MWh deficit for storage needs.
    """
    if df.empty:
        return []

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['below_avg'] = df['capacity_factor_pct'] < avg_cf

    # Find continuous periods below average
    periods = []
    current_period = None

    for idx, row in df.iterrows():
        if row['below_avg']:
            if current_period is None:
                current_period = {
                    'start_idx': idx,
                    'start_time': row['timestamp'],
                    'end_idx': idx,
                    'end_time': row['timestamp'],
                    'mwh_produced': row['output_mw'] * 0.5,  # Assuming 30-min settlements
                    'count': 1
                }
            else:
                current_period['end_idx'] = idx
                current_period['end_time'] = row['timestamp']
                current_period['mwh_produced'] += row['output_mw'] * 0.5
                current_period['count'] += 1
        else:
            if current_period is not None:
                periods.append(current_period)
                current_period = None

    # Don't forget the last period if it ends below average
    if current_period is not None:
        periods.append(current_period)

    # Calculate what would have been produced at average CF
    for period in periods:
        duration_hours = period['count'] * 0.5
        period['duration_hours'] = duration_hours
        # Calculate deficit using the correct capacity
        avg_mw = (avg_cf / 100) * total_capacity
        period['avg_mwh'] = avg_mw * duration_hours
        period['deficit_mwh'] = period['avg_mwh'] - period['mwh_produced']

    # Sort by duration (longest first)
    periods.sort(key=lambda x: x['duration_hours'], reverse=True)

    return periods


def plot_monthly_analysis(year, month, h1_data, h2_data):
    """Create detailed plots for a single month"""

    month_name = calendar.month_name[month]

    # Calculate averages
    h1_avg_cf = h1_data['capacity_factor_pct'].mean()
    h2_avg_cf = h2_data['capacity_factor_pct'].mean()

    # Find below-average periods
    h1_periods = find_below_average_periods(h1_data, h1_avg_cf, HORNSEA_1_CAPACITY)
    h2_periods = find_below_average_periods(h2_data, h2_avg_cf, HORNSEA_2_CAPACITY)

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Hornsea 1 capacity factor over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(h1_data['timestamp'], h1_data['capacity_factor_pct'],
             linewidth=0.8, alpha=0.7, color='steelblue')
    ax1.axhline(y=h1_avg_cf, color='red', linestyle='--', linewidth=2,
                label=f'Monthly Avg: {h1_avg_cf:.2f}%')

    # Highlight largest below-average period
    if h1_periods:
        largest = h1_periods[0]
        mask = (h1_data['timestamp'] >= largest['start_time']) & \
               (h1_data['timestamp'] <= largest['end_time'])
        ax1.fill_between(h1_data[mask]['timestamp'],
                         0, h1_data[mask]['capacity_factor_pct'],
                         alpha=0.3, color='orange',
                         label=f'Largest lull: {largest["duration_hours"]:.1f}h')

    ax1.set_ylabel('Capacity Factor (%)', fontsize=11)
    ax1.set_title(f'Hornsea 1 Output - {month_name} {year}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Hornsea 2 capacity factor over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(h2_data['timestamp'], h2_data['capacity_factor_pct'],
             linewidth=0.8, alpha=0.7, color='darkgreen')
    ax2.axhline(y=h2_avg_cf, color='red', linestyle='--', linewidth=2,
                label=f'Monthly Avg: {h2_avg_cf:.2f}%')

    # Highlight largest below-average period
    if h2_periods:
        largest = h2_periods[0]
        mask = (h2_data['timestamp'] >= largest['start_time']) & \
               (h2_data['timestamp'] <= largest['end_time'])
        ax2.fill_between(h2_data[mask]['timestamp'],
                         0, h2_data[mask]['capacity_factor_pct'],
                         alpha=0.3, color='orange',
                         label=f'Largest lull: {largest["duration_hours"]:.1f}h')

    ax2.set_ylabel('Capacity Factor (%)', fontsize=11)
    ax2.set_title(f'Hornsea 2 Output - {month_name} {year}', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Distribution comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(h1_data['capacity_factor_pct'], bins=50, alpha=0.6,
             label='Hornsea 1', color='steelblue', edgecolor='black')
    ax3.axvline(x=h1_avg_cf, color='blue', linestyle='--', linewidth=2)
    ax3.hist(h2_data['capacity_factor_pct'], bins=50, alpha=0.6,
             label='Hornsea 2', color='darkgreen', edgecolor='black')
    ax3.axvline(x=h2_avg_cf, color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('Capacity Factor (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Capacity Factor Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Storage needs analysis
    ax4 = fig.add_subplot(gs[1, 1])
    if h1_periods and h2_periods:
        # Show top 5 largest lulls for each
        h1_top = h1_periods[:5]
        h2_top = h2_periods[:5]

        x_pos = np.arange(len(h1_top))
        width = 0.35

        h1_deficits = [p['deficit_mwh'] for p in h1_top]
        h2_deficits = [p['deficit_mwh'] for p in h2_top[:len(h1_top)]]

        ax4.bar(x_pos - width / 2, h1_deficits, width, label='Hornsea 1',
                color='steelblue', alpha=0.8)
        ax4.bar(x_pos + width / 2, h2_deficits, width, label='Hornsea 2',
                color='darkgreen', alpha=0.8)

        ax4.set_xlabel('Lull Start Date', fontsize=10)
        ax4.set_ylabel('Energy Deficit (MWh)', fontsize=10)
        ax4.set_title('Storage Requirements - Top 5 Lulls', fontsize=11, fontweight='bold')
        ax4.set_xticks(x_pos)
        # Format dates as MM/DD for individual monthly plots
        date_labels = [p['start_time'].strftime('%m/%d') for p in h1_top]
        ax4.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    summary_text = f"Storage Analysis Summary - {month_name} {year}\n"
    summary_text += "=" * 80 + "\n\n"

    # Hornsea 1 summary
    summary_text += f"HORNSEA 1 (Capacity: {HORNSEA_1_CAPACITY} MW):\n"
    summary_text += f"  Average Capacity Factor: {h1_avg_cf:.2f}%\n"
    summary_text += f"  Peak CF: {h1_data['capacity_factor_pct'].max():.2f}%\n"
    summary_text += f"  Minimum CF: {h1_data['capacity_factor_pct'].min():.2f}%\n"
    if h1_periods:
        summary_text += f"  Number of below-average periods: {len(h1_periods)}\n"
        summary_text += f"  Longest lull: {h1_periods[0]['duration_hours']:.1f} hours\n"
        summary_text += f"  Max deficit in single lull: {h1_periods[0]['deficit_mwh']:.1f} MWh\n"
        total_deficit = sum(p['deficit_mwh'] for p in h1_periods)
        summary_text += f"  Total monthly deficit: {total_deficit:.1f} MWh\n"
    summary_text += "\n"

    # Hornsea 2 summary
    summary_text += f"HORNSEA 2 (Capacity: {HORNSEA_2_CAPACITY} MW):\n"
    summary_text += f"  Average Capacity Factor: {h2_avg_cf:.2f}%\n"
    summary_text += f"  Peak CF: {h2_data['capacity_factor_pct'].max():.2f}%\n"
    summary_text += f"  Minimum CF: {h2_data['capacity_factor_pct'].min():.2f}%\n"
    if h2_periods:
        summary_text += f"  Number of below-average periods: {len(h2_periods)}\n"
        summary_text += f"  Longest lull: {h2_periods[0]['duration_hours']:.1f} hours\n"
        summary_text += f"  Max deficit in single lull: {h2_periods[0]['deficit_mwh']:.1f} MWh\n"
        total_deficit = sum(p['deficit_mwh'] for p in h2_periods)
        summary_text += f"  Total monthly deficit: {total_deficit:.1f} MWh\n"

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    filename = f'hornsea_analysis_{year}_{month:02d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {filename}")
    plt.close()

    return h1_avg_cf, h2_avg_cf, h1_periods, h2_periods


def create_trend_plot(results_df):
    """Create trend plot showing CF over time"""

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Create x-axis labels
    x_labels = [f"{row['month_name'][:3]} {row['year']}"
                for _, row in results_df.iterrows()]
    x_pos = np.arange(len(x_labels))

    # Plot 1: Average capacity factors
    ax1 = axes[0]
    ax1.plot(x_pos, results_df['h1_avg_cf'], marker='o', linewidth=2,
             markersize=8, label='Hornsea 1', color='steelblue')
    ax1.plot(x_pos, results_df['h2_avg_cf'], marker='s', linewidth=2,
             markersize=8, label='Hornsea 2', color='darkgreen')

    # Add reference lines
    h1_overall_avg = results_df['h1_avg_cf'].mean()
    h2_overall_avg = results_df['h2_avg_cf'].mean()
    ax1.axhline(y=h1_overall_avg, color='steelblue', linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'H1 Overall: {h1_overall_avg:.2f}%')
    ax1.axhline(y=h2_overall_avg, color='darkgreen', linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'H2 Overall: {h2_overall_avg:.2f}%')

    ax1.set_ylabel('Average Capacity Factor (%)', fontsize=11)
    ax1.set_title('Monthly Capacity Factor Trends', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Maximum lull duration
    ax2 = axes[1]
    width = 0.35
    ax2.bar(x_pos - width / 2, results_df['h1_longest_lull_hours'], width,
            label='Hornsea 1', color='steelblue', alpha=0.8)
    ax2.bar(x_pos + width / 2, results_df['h2_longest_lull_hours'], width,
            label='Hornsea 2', color='darkgreen', alpha=0.8)

    ax2.set_ylabel('Duration (hours)', fontsize=11)
    ax2.set_title('Longest Below-Average Period Each Month', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Storage requirements (max single lull deficit)
    ax3 = axes[2]
    ax3.bar(x_pos - width / 2, results_df['h1_max_deficit_mwh'], width,
            label='Hornsea 1', color='steelblue', alpha=0.8)
    ax3.bar(x_pos + width / 2, results_df['h2_max_deficit_mwh'], width,
            label='Hornsea 2', color='darkgreen', alpha=0.8)

    ax3.set_ylabel('Energy Deficit (MWh)', fontsize=11)
    ax3.set_title('Maximum Single-Lull Storage Requirement', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = 'hornsea_trends.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Trend plot saved: {filename}")
    plt.close()


if __name__ == "__main__":
    # Analyze only specific reliable months
    # Format: (year, month)
    months_to_analyze = [
        (2024, 8),  # August 2024
        (2024, 10),  # October 2024
        (2024, 11),  # November 2024
        (2024, 12),  # December 2024
        (2025, 1),  # January 2025
        (2025, 2),  # February 2025
        (2025, 3),  # March 2025
        (2025, 8),  # August 2025
        (2025, 9),  # September 2025
    ]

    print("\n" + "=" * 60)
    print("HORNSEA WIND FARM CAPACITY FACTOR & STORAGE ANALYSIS")
    print("=" * 60 + "\n")

    results = []

    for year, month in months_to_analyze:
        print(f"\n{'=' * 60}")
        print(f"Processing {calendar.month_name[month]} {year}")
        print(f"{'=' * 60}")

        # Get data for the month
        df = get_month_data(year, month)

        if df.empty:
            print(f"  No data available for {calendar.month_name[month]} {year}")
            continue

        # Extract power data
        df = extract_power_data(df)
        print(f"  Total records after PN filtering: {len(df)}")

        # Filter out maintenance periods
        df = filter_maintenance_periods(df)
        print(f"  Records after maintenance filtering: {len(df)}")

        # Calculate capacity factors for each farm group
        h1_data = calculate_capacity_factor(df, HORNSEA_1, HORNSEA_1_CAPACITY)
        h2_data = calculate_capacity_factor(df, HORNSEA_2, HORNSEA_2_CAPACITY)

        if h1_data.empty or h2_data.empty:
            print(f"  Insufficient data for analysis")
            continue

        # Create plots and get statistics
        h1_avg, h2_avg, h1_periods, h2_periods = plot_monthly_analysis(
            year, month, h1_data, h2_data)

        # Store results
        result = {
            'year': year,
            'month': month,
            'month_name': calendar.month_name[month],
            'h1_avg_cf': h1_avg,
            'h2_avg_cf': h2_avg,
            'h1_max_cf': h1_data['capacity_factor_pct'].max(),
            'h2_max_cf': h2_data['capacity_factor_pct'].max(),
            'h1_min_cf': h1_data['capacity_factor_pct'].min(),
            'h2_min_cf': h2_data['capacity_factor_pct'].min(),
            'h1_longest_lull_hours': h1_periods[0]['duration_hours'] if h1_periods else 0,
            'h2_longest_lull_hours': h2_periods[0]['duration_hours'] if h2_periods else 0,
            'h1_max_deficit_mwh': h1_periods[0]['deficit_mwh'] if h1_periods else 0,
            'h2_max_deficit_mwh': h2_periods[0]['deficit_mwh'] if h2_periods else 0,
            'h1_total_deficit_mwh': sum(p['deficit_mwh'] for p in h1_periods) if h1_periods else 0,
            'h2_total_deficit_mwh': sum(p['deficit_mwh'] for p in h2_periods) if h2_periods else 0,
        }
        results.append(result)

        print(f"  ✓ Analysis complete")

    # Create summary dataframe
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_filename = 'hornsea_monthly_summary.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\n{'=' * 60}")
    print(f"Summary saved to: {csv_filename}")
    print(f"{'=' * 60}\n")

    # Create trend plot
    create_trend_plot(results_df)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"\nHornsea 1 - Overall Average CF: {results_df['h1_avg_cf'].mean():.2f}%")
    print(f"Hornsea 2 - Overall Average CF: {results_df['h2_avg_cf'].mean():.2f}%")
    print(
        f"\nCapacity Factor Improvement (H2 vs H1): {((results_df['h2_avg_cf'].mean() / results_df['h1_avg_cf'].mean()) - 1) * 100:.2f}%")
    print(f"\nMax single-lull storage need:")
    print(f"  Hornsea 1: {results_df['h1_max_deficit_mwh'].max():.1f} MWh")
    print(f"  Hornsea 2: {results_df['h2_max_deficit_mwh'].max():.1f} MWh")
    print(f"\nLongest lull observed:")
    print(f"  Hornsea 1: {results_df['h1_longest_lull_hours'].max():.1f} hours")
    print(f"  Hornsea 2: {results_df['h2_longest_lull_hours'].max():.1f} hours")
    print("\n" + "=" * 60 + "\n")