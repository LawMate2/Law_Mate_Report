"""
Analysis script for benchmark results
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


class BenchmarkAnalyzer:
    """Analyzer for benchmark results"""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results = []

    def load_results(self):
        """Load all benchmark results from the results directory"""
        for file in self.results_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                self.results.append(json.load(f))

        print(f"Loaded {len(self.results)} benchmark results")

    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all results"""
        data = []

        for result in self.results:
            embedding_info = result['embedding_model']
            store_info = result['vector_store']
            perf = result['results']

            row = {
                'Embedding Model': embedding_info['model_name'],
                'Dimension': embedding_info['dimension'],
                'Vector Store': store_info['store_name'],
                'Num Vectors': store_info['num_vectors'],
            }

            if 'indexing' in perf:
                row['Indexing Time (s)'] = perf['indexing']['total_time']
                row['Time per Doc (ms)'] = perf['indexing']['embedding_time_per_doc'] * 1000

            if 'search' in perf:
                row['Avg Search Time (ms)'] = perf['search']['avg_search_time'] * 1000

            if 'quality' in perf:
                row['Precision'] = perf['quality']['avg_precision']
                row['Recall'] = perf['quality']['avg_recall']
                row['F1 Score'] = perf['quality']['avg_f1']
                row['MRR'] = perf['quality']['avg_mrr']

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def get_best_performers(self) -> Dict[str, any]:
        """Identify best performing configurations"""
        df = self.create_comparison_table()

        best = {}

        if 'Avg Search Time (ms)' in df.columns:
            fastest_search = df.loc[df['Avg Search Time (ms)'].idxmin()]
            best['fastest_search'] = {
                'embedding': fastest_search['Embedding Model'],
                'vector_store': fastest_search['Vector Store'],
                'time_ms': fastest_search['Avg Search Time (ms)']
            }

        if 'Indexing Time (s)' in df.columns:
            fastest_indexing = df.loc[df['Indexing Time (s)'].idxmin()]
            best['fastest_indexing'] = {
                'embedding': fastest_indexing['Embedding Model'],
                'vector_store': fastest_indexing['Vector Store'],
                'time_s': fastest_indexing['Indexing Time (s)']
            }

        if 'F1 Score' in df.columns:
            best_quality = df.loc[df['F1 Score'].idxmax()]
            best['best_quality'] = {
                'embedding': best_quality['Embedding Model'],
                'vector_store': best_quality['Vector Store'],
                'f1': best_quality['F1 Score']
            }

        return best

    def generate_summary_report(self, output_file: str = "summary_report.md"):
        """Generate a markdown summary report"""
        df = self.create_comparison_table()
        best = self.get_best_performers()

        report = []
        report.append("# RAG 성능 비교 리포트\n")
        report.append(f"총 실험 수: {len(self.results)}\n\n")

        report.append("## 성능 비교 표\n")
        report.append(df.to_markdown(index=False))
        report.append("\n\n")

        report.append("## 최고 성능 구성\n\n")

        if 'fastest_search' in best:
            report.append("### 가장 빠른 검색\n")
            report.append(f"- Embedding: {best['fastest_search']['embedding']}\n")
            report.append(f"- Vector Store: {best['fastest_search']['vector_store']}\n")
            report.append(f"- Search Time: {best['fastest_search']['time_ms']:.2f} ms\n\n")

        if 'fastest_indexing' in best:
            report.append("### 가장 빠른 인덱싱\n")
            report.append(f"- Embedding: {best['fastest_indexing']['embedding']}\n")
            report.append(f"- Vector Store: {best['fastest_indexing']['vector_store']}\n")
            report.append(f"- Indexing Time: {best['fastest_indexing']['time_s']:.2f} s\n\n")

        if 'best_quality' in best:
            report.append("### 최고 검색 품질\n")
            report.append(f"- Embedding: {best['best_quality']['embedding']}\n")
            report.append(f"- Vector Store: {best['best_quality']['vector_store']}\n")
            report.append(f"- F1 Score: {best['best_quality']['f1']:.4f}\n\n")

        report_content = "".join(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"Summary report saved to {output_file}")
        return report_content


def main():
    analyzer = BenchmarkAnalyzer()
    analyzer.load_results()

    df = analyzer.create_comparison_table()
    print("\nComparison Table:")
    print(df.to_string())

    print("\n" + "="*80 + "\n")

    best = analyzer.get_best_performers()
    print("Best Performers:")
    for category, info in best.items():
        print(f"\n{category}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    analyzer.generate_summary_report()


if __name__ == "__main__":
    main()
