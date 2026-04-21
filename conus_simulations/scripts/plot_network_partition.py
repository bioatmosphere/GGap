"""
Visualize CONUS network graph and METIS partitioning.

Creates publication-ready plots showing:
- Geographic partition map
- Network connectivity
- Partition quality metrics
- Dispersal distance distributions

Usage:
    cd conus_simulations/scripts

    # Basic plots
    python plot_network_partition.py

    # Interactive map
    python plot_network_partition.py --interactive

    # Custom output
    python plot_network_partition.py --output ../results/plots --format pdf --dpi 300
"""

import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_data(results_dir):
    """Load partition, edges, and dispersal data."""
    partition_file = os.path.join(results_dir, "partition.csv")
    edges_file = os.path.join(results_dir, "edges_directed.csv")
    dispersal_file = os.path.join(results_dir, "site_dispersal.csv")

    # Load partition data
    partition_data = []
    with open(partition_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            partition_data.append({
                'site_id': int(row['site_id']),
                'partition': int(row['partition']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'num_out_edges': int(row['num_out_edges']),
                'num_in_edges': int(row['num_in_edges']),
                'max_dispersal_km': float(row['max_dispersal_km']),
            })

    # Load edges data
    edges_data = []
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges_data.append({
                'reader_site': int(row['reader_site']),
                'source_site': int(row['source_site']),
                'distance_km': float(row['distance_km']),
                'cross_partition': int(row['cross_partition']),
            })

    # Load dispersal data
    dispersal_data = []
    with open(dispersal_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dispersal_data.append({
                'site_id': int(row['site_id']),
                'num_species': int(row['num_species']),
                'max_dispersal_km': float(row['max_dispersal_km']),
                'out_degree': int(row['out_degree']),
                'in_degree': int(row['in_degree']),
            })

    return partition_data, edges_data, dispersal_data


def plot_geographic_partition(partition_data, edges_data, output_file, max_edges=500, max_distance=None):
    """Plot sites on geographic map showing spatial clustering by partition."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Get unique partitions
    partitions = sorted(set(p['partition'] for p in partition_data))
    num_partitions = len(partitions)

    # Group sites by partition to visualize spatial clustering
    partition_sites = {}
    for p in partition_data:
        part_id = p['partition']
        if part_id not in partition_sites:
            partition_sites[part_id] = []
        partition_sites[part_id].append(p)

    # Use random colors with fixed seed for reproducibility
    np.random.seed(42)

    # Draw bounding boxes around each partition
    for part_id in sorted(partition_sites.keys()):
        sites = partition_sites[part_id]

        lons = [s['longitude'] for s in sites]
        lats = [s['latitude'] for s in sites]

        # Random color for each partition
        color = np.random.rand(3,)

        # Calculate bounding box with small padding
        min_lon, max_lon = min(lons) - 0.1, max(lons) + 0.1
        min_lat, max_lat = min(lats) - 0.1, max(lats) + 0.1

        # Draw rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                         linewidth=1.2, edgecolor=color, facecolor=color,
                         alpha=0.15, zorder=1)
        ax.add_patch(rect)

        # Also draw sites in partition with matching color
        ax.scatter(lons, lats, c=[color], s=30, alpha=0.8,
                  edgecolors='black', linewidths=0.5, zorder=2)

    # Labels and formatting
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'CONUS Forest Sites - METIS Partitioning ({num_partitions} partitions)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add info text
    avg_sites_per_partition = len(partition_data) / num_partitions
    info_text = f'Total sites: {len(partition_data)}\n'
    info_text += f'Partitions: {num_partitions}\n'
    info_text += f'Avg sites per partition: {avg_sites_per_partition:.1f}\n'
    info_text += f'\nNearby sites are grouped\ninto the same partition\n(shown by color clustering)'

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Created: {output_file}")
    plt.close()


def plot_degree_distribution(dispersal_data, output_file):
    """Plot in-degree and out-degree distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    out_degrees = [d['out_degree'] for d in dispersal_data]
    in_degrees = [d['in_degree'] for d in dispersal_data]

    # Out-degree histogram
    ax1.hist(out_degrees, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(out_degrees), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(out_degrees):.1f}')
    ax1.set_xlabel('Out-Degree (sites this site can disperse to)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Out-Degree Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # In-degree histogram
    ax2.hist(in_degrees, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(in_degrees), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(in_degrees):.1f}')
    ax2.set_xlabel('In-Degree (sites that can disperse to this site)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('In-Degree Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Created: {output_file}")
    plt.close()


def plot_partition_sizes(partition_data, output_file):
    """Plot bar chart of sites per partition."""
    partitions = sorted(set(p['partition'] for p in partition_data))
    partition_counts = defaultdict(int)
    for p in partition_data:
        partition_counts[p['partition']] += 1

    fig, ax = plt.subplots(figsize=(12, 6))

    counts = [partition_counts[p] for p in partitions]

    # Use consistent color scheme with geographic plot
    if len(partitions) <= 20:
        cmap = plt.cm.tab20
        colors = cmap(np.linspace(0, 1, len(partitions)))
    else:
        # Match the geographic plot color scheme
        np.random.seed(42)
        indices = np.random.permutation(len(partitions))
        cmap = plt.cm.hsv
        all_colors = cmap(np.linspace(0, 0.95, len(partitions)))
        colors = all_colors[indices]

    bars = ax.bar(partitions, counts, color=colors, edgecolor='black', alpha=0.7)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Statistics
    mean_count = np.mean(counts)
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')

    ax.set_xlabel('Partition ID', fontsize=14)
    ax.set_ylabel('Number of Sites', fontsize=14)
    ax.set_title('Sites Per Partition - Load Balance', fontsize=16, fontweight='bold')
    ax.set_xticks(partitions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add imbalance info
    imbalance = (max(counts) - min(counts)) / max(counts) * 100
    info_text = f'Min: {min(counts)}, Max: {max(counts)}\nImbalance: {imbalance:.1f}%'
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Created: {output_file}")
    plt.close()


def plot_dispersal_distances(dispersal_data, edges_data, output_file):
    """Plot dispersal distance distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Max dispersal per site
    max_dispersals = [d['max_dispersal_km'] for d in dispersal_data]
    ax1.hist(max_dispersals, bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(max_dispersals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_dispersals):.1f} km')
    ax1.set_xlabel('Max Dispersal Distance (km)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Per-Site Maximum Dispersal Distance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Edge distances
    edge_distances = [e['distance_km'] for e in edges_data]
    ax2.hist(edge_distances, bins=50, color='purple', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(edge_distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(edge_distances):.1f} km')
    ax2.set_xlabel('Edge Distance (km)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Edge Distances', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Created: {output_file}")
    plt.close()


def plot_edge_analysis(edges_data, partition_data, output_file):
    """Plot edge cut analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Create site lookup
    site_lookup = {p['site_id']: p for p in partition_data}

    # Cross-partition vs within-partition
    cross_edges = [e for e in edges_data if e['cross_partition']]
    within_edges = [e for e in edges_data if not e['cross_partition']]

    # Pie chart
    sizes = [len(cross_edges), len(within_edges)]
    labels = [f'Cross-partition\n({len(cross_edges):,} edges)',
              f'Within-partition\n({len(within_edges):,} edges)']
    colors = ['#ff6b6b', '#95e1d3']
    explode = (0.05, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Edge Distribution', fontsize=14, fontweight='bold')

    # Edge cuts by partition pair
    partition_pairs = defaultdict(int)
    for edge in cross_edges:
        reader = site_lookup[edge['reader_site']]
        source = site_lookup[edge['source_site']]
        pair = tuple(sorted([reader['partition'], source['partition']]))
        partition_pairs[pair] += 1

    # Sort by count
    sorted_pairs = sorted(partition_pairs.items(), key=lambda x: x[1], reverse=True)[:15]

    if sorted_pairs:
        pair_labels = [f"{p[0]}-{p[1]}" for p, _ in sorted_pairs]
        pair_counts = [count for _, count in sorted_pairs]

        ax2.barh(range(len(pair_labels)), pair_counts, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_yticks(range(len(pair_labels)))
        ax2.set_yticklabels(pair_labels)
        ax2.set_xlabel('Number of Cross-Partition Edges', fontsize=12)
        ax2.set_ylabel('Partition Pair', fontsize=12)
        ax2.set_title('Top Cross-Partition Edge Cuts', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Created: {output_file}")
    plt.close()


def plot_interactive_map(partition_data, edges_data, output_file, max_edges=1000):
    """Create interactive plotly map."""
    if not HAS_PLOTLY:
        print("  Skipping interactive map (plotly not installed)")
        return

    # Prepare data
    partitions = sorted(set(p['partition'] for p in partition_data))

    # Create traces for each partition
    fig = go.Figure()

    # Add edges (sample)
    edges_sample = edges_data[:max_edges]
    site_lookup = {p['site_id']: p for p in partition_data}

    edge_lons = []
    edge_lats = []
    for edge in edges_sample:
        reader = site_lookup[edge['reader_site']]
        source = site_lookup[edge['source_site']]

        edge_lons.extend([source['longitude'], reader['longitude'], None])
        edge_lats.extend([source['latitude'], reader['latitude'], None])

    fig.add_trace(go.Scattergeo(
        lon=edge_lons,
        lat=edge_lats,
        mode='lines',
        line=dict(width=0.5, color='gray'),
        opacity=0.3,
        name='Edges',
        hoverinfo='skip'
    ))

    # Add sites by partition
    colors = px.colors.qualitative.Set3[:len(partitions)]
    for i, part in enumerate(partitions):
        part_sites = [p for p in partition_data if p['partition'] == part]

        lons = [p['longitude'] for p in part_sites]
        lats = [p['latitude'] for p in part_sites]
        sizes = [5 + p['num_out_edges'] * 0.5 for p in part_sites]
        hover_text = [
            f"Site {p['site_id']}<br>" +
            f"Partition: {p['partition']}<br>" +
            f"Out-degree: {p['num_out_edges']}<br>" +
            f"In-degree: {p['num_in_edges']}<br>" +
            f"Max dispersal: {p['max_dispersal_km']:.1f} km"
            for p in part_sites
        ]

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='markers',
            marker=dict(size=sizes, color=colors[i], line=dict(width=0.5, color='black')),
            name=f'Partition {part}',
            text=hover_text,
            hoverinfo='text'
        ))

    # Update layout
    fig.update_layout(
        title=f'CONUS Forest Sites - Interactive Map ({len(partition_data)} sites, {len(partitions)} partitions)',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
        ),
        height=800,
        showlegend=True
    )

    fig.write_html(output_file)
    print(f"  Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CONUS network graph and METIS partitioning"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory with partition CSV files (default: ../results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/plots",
        help="Output directory for plots (default: ../results/plots)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot format (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Plot resolution (default: 300)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive plotly map (requires plotly)"
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=500,
        help="Max edges to show on geographic map (default: 500)"
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=None,
        help="Only show edges below this distance (km)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CONUS Network Visualization")
    print("=" * 80)
    print(f"\nLoading data from {args.results_dir}...")

    # Load data
    partition_data, edges_data, dispersal_data = load_data(args.results_dir)

    print(f"  Sites: {len(partition_data)}")
    print(f"  Edges: {len(edges_data)}")
    print(f"  Partitions: {len(set(p['partition'] for p in partition_data))}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nGenerating plots...")

    # 1. Geographic partition map
    output_file = os.path.join(args.output_dir, f"01_geographic_partition.{args.format}")
    plot_geographic_partition(partition_data, edges_data, output_file, args.max_edges, args.max_distance)

    # 2. Degree distribution
    output_file = os.path.join(args.output_dir, f"02_degree_distribution.{args.format}")
    plot_degree_distribution(dispersal_data, output_file)

    # 3. Partition sizes
    output_file = os.path.join(args.output_dir, f"03_partition_sizes.{args.format}")
    plot_partition_sizes(partition_data, output_file)

    # 4. Dispersal distances
    output_file = os.path.join(args.output_dir, f"04_dispersal_distances.{args.format}")
    plot_dispersal_distances(dispersal_data, edges_data, output_file)

    # 5. Edge analysis
    output_file = os.path.join(args.output_dir, f"05_edge_analysis.{args.format}")
    plot_edge_analysis(edges_data, partition_data, output_file)

    # 6. Interactive map
    if args.interactive:
        output_file = os.path.join(args.output_dir, "interactive_map.html")
        plot_interactive_map(partition_data, edges_data, output_file, args.max_edges)

    print(f"\n{'=' * 80}")
    print(f"Plots saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
