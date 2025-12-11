"""
Visualization script for benchmark results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


class BenchmarkVisualizer:
    """Visualizer for benchmark results"""

    def __init__(self, results_dir: str = "./results", output_dir: str = "./results/plots"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []

    def load_results(self):
        """Load all benchmark results"""
        for file in self.results_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                self.results.append(json.load(f))

        print(f"Loaded {len(self.results)} results for visualization")

    def plot_search_time_comparison(self):
        """Plot search time comparison across configurations"""
        data = []

        for result in self.results:
            if 'search' in result['results']:
                data.append({
                    'Model': result['embedding_model']['model_name'][:20],
                    'Store': result['vector_store']['store_name'],
                    'Search Time (ms)': result['results']['search']['avg_search_time'] * 1000
                })

        if not data:
            print("No search time data available")
            return

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))
        df_pivot = df.pivot(index='Model', columns='Store', values='Search Time (ms)')
        df_pivot.plot(kind='bar', ax=ax)

        ax.set_ylabel('Search Time (ms)')
        ax.set_title('Search Time Comparison Across Vector Stores')
        ax.legend(title='Vector Store')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.output_dir / 'search_time_comparison.png'
        plt.savefig(output_file, dpi=300)
        print(f"Saved: {output_file}")
        plt.close()

    def plot_indexing_time_comparison(self):
        """Plot indexing time comparison"""
        data = []

        for result in self.results:
            if 'indexing' in result['results']:
                data.append({
                    'Model': result['embedding_model']['model_name'][:20],
                    'Store': result['vector_store']['store_name'],
                    'Indexing Time (s)': result['results']['indexing']['total_time']
                })

        if not data:
            print("No indexing time data available")
            return

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))
        df_pivot = df.pivot(index='Model', columns='Store', values='Indexing Time (s)')
        df_pivot.plot(kind='bar', ax=ax)

        ax.set_ylabel('Indexing Time (s)')
        ax.set_title('Indexing Time Comparison Across Vector Stores')
        ax.legend(title='Vector Store')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.output_dir / 'indexing_time_comparison.png'
        plt.savefig(output_file, dpi=300)
        print(f"Saved: {output_file}")
        plt.close()

    def plot_quality_metrics(self):
        """Plot retrieval quality metrics"""
        data = []

        for result in self.results:
            if 'quality' in result['results']:
                quality = result['results']['quality']
                data.append({
                    'Model': result['embedding_model']['model_name'][:20],
                    'Store': result['vector_store']['store_name'],
                    'Precision': quality['avg_precision'],
                    'Recall': quality['avg_recall'],
                    'F1': quality['avg_f1'],
                    'MRR': quality['avg_mrr']
                })

        if not data:
            print("No quality metrics data available")
            return

        df = pd.DataFrame(data)
        df['Config'] = df['Model'] + '\n' + df['Store']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['Precision', 'Recall', 'F1', 'MRR']
        for ax, metric in zip(axes.flat, metrics):
            df_sorted = df.sort_values(by=metric, ascending=False)
            ax.barh(df_sorted['Config'], df_sorted[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.invert_yaxis()

        plt.tight_layout()

        output_file = self.output_dir / 'quality_metrics.png'
        plt.savefig(output_file, dpi=300)
        print(f"Saved: {output_file}")
        plt.close()

    def plot_dimension_vs_performance(self):
        """Plot dimension vs performance scatter plot"""
        data = []

        for result in self.results:
            if 'search' in result['results']:
                data.append({
                    'Dimension': result['embedding_model']['dimension'],
                    'Search Time (ms)': result['results']['search']['avg_search_time'] * 1000,
                    'Model': result['embedding_model']['model_name'][:15]
                })

        if not data:
            print("No dimension performance data available")
            return

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            ax.scatter(model_data['Dimension'], model_data['Search Time (ms)'],
                      label=model, s=100, alpha=0.7)

        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Search Time (ms)')
        ax.set_title('Embedding Dimension vs Search Performance')
        ax.legend()
        plt.tight_layout()

        output_file = self.output_dir / 'dimension_vs_performance.png'
        plt.savefig(output_file, dpi=300)
        print(f"Saved: {output_file}")
        plt.close()

    def plot_heatmap(self):
        """Plot heatmap of search times"""
        data = []

        for result in self.results:
            if 'search' in result['results']:
                data.append({
                    'Embedding': result['embedding_model']['model_name'][:20],
                    'Vector Store': result['vector_store']['store_name'],
                    'Search Time': result['results']['search']['avg_search_time'] * 1000
                })

        if not data:
            print("No data for heatmap")
            return

        df = pd.DataFrame(data)
        pivot = df.pivot_table(values='Search Time',
                              index='Embedding',
                              columns='Vector Store',
                              aggfunc='mean')

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Search Time Heatmap (ms)')
        plt.tight_layout()

        output_file = self.output_dir / 'search_time_heatmap.png'
        plt.savefig(output_file, dpi=300)
        print(f"Saved: {output_file}")
        plt.close()

    def generate_all_plots(self):
        """Generate all visualization plots"""
        self.load_results()

        print("Generating plots...")
        self.plot_search_time_comparison()
        self.plot_indexing_time_comparison()
        self.plot_quality_metrics()
        self.plot_dimension_vs_performance()
        self.plot_heatmap()

        print(f"\nAll plots saved to {self.output_dir}")


def main():
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
