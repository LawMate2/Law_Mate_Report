"""
data 디렉토리의 모든 PDF 파일에 대해 벤치마크 실행
"""

import os
import subprocess
from pathlib import Path
import json

def main():
    # data 디렉토리의 모든 PDF 파일 찾기
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("data 디렉토리에 PDF 파일이 없습니다.")
        return

    print(f"총 {len(pdf_files)}개의 PDF 파일 발견:")
    for pdf in pdf_files:
        print(f"  - {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")

    # 각 PDF에 대해 벤치마크 실행
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name} 처리 중...")
        print(f"{'='*80}")

        # 출력 디렉토리 이름 생성 (PDF 파일명 기반)
        pdf_name = pdf_file.stem  # 확장자 제외한 파일명
        output_dir = f"results/pdf_benchmark/{pdf_name}"

        # pdf_benchmark.py 실행
        cmd = [
            "python", "pdf_benchmark.py",
            "--pdf", str(pdf_file),
            "--num-runs", "3",
            "--chunk-size", "500",
            "--overlap", "50",
            "--output-dir", output_dir
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            print(f"\n✓ {pdf_file.name} 벤치마크 완료!")
            print(f"  결과 저장 위치: {output_dir}")

        except subprocess.CalledProcessError as e:
            print(f"\n✗ {pdf_file.name} 처리 실패: {e}")
            continue

    print(f"\n{'='*80}")
    print("모든 PDF 벤치마크 완료!")
    print(f"{'='*80}")

    # 모든 결과 요약
    print("\n결과 요약:")
    results_base = Path("results/pdf_benchmark")
    for pdf in pdf_files:
        pdf_name = pdf.stem
        result_file = results_base / pdf_name / "all_results.json"

        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"\n{pdf.name}:")
            for exp in data:
                print(f"  - {exp['experiment_name']}:")
                print(f"    임베딩: {exp['indexing']['avg_embedding_time']:.3f}±{exp['indexing']['std_embedding_time']:.3f}s")
                print(f"    검색: {exp['search']['avg_search_time']:.4f}±{exp['search']['std_search_time']:.4f}s")


if __name__ == "__main__":
    main()