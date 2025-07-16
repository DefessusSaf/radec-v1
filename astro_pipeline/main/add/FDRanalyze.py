import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_report(report_path):
    """
    Парсинг текстового отчёта.
    Формат:
      File: filename.fits
      timestamp
      [*]RA Dec x y ...  — корректная детекция
      RA Dec x y ...   — ложная детекция
      -                — пропущенный объект (FN)
    Возвращает:
      df: DataFrame с колонками ['starred'] (True для *, False для coords),
      missed_count: int — число строк '-' (FN)
    """
    starred_count = 0
    false_count = 0
    missed_count = 0
    with open(report_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines[2:]:  
        if line == '-':
            missed_count += 1
        elif line.startswith('*'):
            starred_count += 1
        else:
            false_count += 1
    return starred_count, false_count, missed_count


def plot_results(res_df):
    df = res_df.sort_values('FDR')

    plt.figure()
    plt.bar(df['file'], df['FDR'])
    plt.xticks(rotation=90)
    plt.xlabel('file')
    plt.ylabel('FDR')
    plt.title('False Detection Rate per File')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(res_df['FDR'].dropna(), bins=20)
    plt.xlabel('FDR')
    plt.ylabel('Number of files')
    plt.title('Distribution FDR')
    plt.tight_layout()
    plt.show()


def main(report_dir, output_csv=None):
    results = []
    pattern = os.path.join(report_dir, '**', '*')
    for report_path in sorted(glob.glob(pattern, recursive=True)):
        if not os.path.isfile(report_path):
            continue
        rel_path = os.path.relpath(report_path, report_dir)
        tp, fp, fn = parse_report(report_path)
        # here fp = the number of false detection, FN = missed_count
        fdr = fp / (tp + fp) if (tp + fp) > 0 else float('nan')
        results.append({'file': rel_path, 'TP': tp, 'FP': fp, 'FN': fn, 'FDR': fdr})

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    total_tp = res_df['TP'].sum()
    total_fp = res_df['FP'].sum()
    total_fn = res_df['FN'].sum()
    overall_fdr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    print(f"\nSummary: TP={total_tp}, FP={total_fp}, FN={total_fn}, Overall FDR={overall_fdr:.3f}")

    if output_csv:
        res_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    plot_results(res_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis FDR according to text reports taking into account markers')
    parser.add_argument('-r', '--reports', required=True, help='Root Directory with reports')
    parser.add_argument('-o', '--output', help='CSV To preserve the results')
    args = parser.parse_args()
    main(args.reports, args.output)