#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run(cmd, cwd=None, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def choose_lengths(max_length: int):
    base = [4096, 8192, 16384, 32768, 65536, 131072]
    return [x for x in base if x <= max_length]


def maybe_trim_lengths(lengths, scores, floor=0.01, consecutive=2):
    if not lengths or not scores or len(lengths) != len(scores):
        return lengths
    below = 0
    keep_upto = len(lengths)
    for i, s in enumerate(scores):
        if s is not None and s <= floor:
            below += 1
        else:
            below = 0
        if below >= consecutive:
            keep_upto = i + 1
            break
    return lengths[:keep_upto]


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a HF model or local checkpoint on RULER and generate figures/reports.')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--model', type=str, help='Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B-Instruct')
    src.add_argument('--checkpoint', type=str, help='Local checkpoint path')
    p.add_argument('--backend', choices=['hf', 'vllm'], default='hf', help='Inference backend for lm-eval')
    p.add_argument('--device', default='cuda', help='cuda, cpu, cuda:0, etc.')
    p.add_argument('--dtype', default='auto', help='Model dtype for HF backend')
    p.add_argument('--batch-size', default='1', help='Batch size for lm-eval')
    p.add_argument('--max-length', type=int, default=32768, help='Maximum context length to test')
    p.add_argument('--lengths', type=int, nargs='*', default=None, help='Explicit context lengths to test')
    p.add_argument('--tasks', nargs='*', default=['niah_single_1','niah_single_2','niah_multikey_1','niah_multiquery','ruler_vt','ruler_cwe','ruler_qa_squad'], help='Subset of RULER tasks')
    p.add_argument('--output-dir', default='output/ruler_results', help='Output directory')
    p.add_argument('--limit', type=int, default=None, help='Optional sample limit per task for smoke tests')
    p.add_argument('--fewshot-as-multiturn', action='store_true')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--tokenizer', type=str, default=None, help='Optional tokenizer override')
    return p.parse_args()


def build_model_args(args, model_ref):
    if args.backend == 'hf':
        parts = [
            f'pretrained={model_ref}',
            f'dtype={args.dtype}',
            f'max_length={args.max_length}'
        ]
        if args.trust_remote_code:
            parts.append('trust_remote_code=True')
        if args.tokenizer:
            parts.append(f'tokenizer={args.tokenizer}')
        return 'hf', ','.join(parts)
    parts = [f'pretrained={model_ref}', f'max_model_len={args.max_length}']
    if args.tokenizer:
        parts.append(f'tokenizer={args.tokenizer}')
    return 'vllm', ','.join(parts)


def extract_rows(obj, task_names, lengths):
    rows = []
    results = obj.get('results', {})
    groups = obj.get('group_subtasks', {})
    all_names = set(results.keys())
    for maybe_group, members in groups.items():
        if 'ruler' in maybe_group:
            all_names.update(members)
    for task in sorted(all_names):
        task_res = results.get(task, {})
        metric_keys = [k for k in task_res.keys() if 'acc' in k.lower() or 'exact' in k.lower() or 'score' in k.lower()]
        preferred = None
        for k in metric_keys:
            if ',none' in k:
                preferred = k
                break
        if preferred is None and metric_keys:
            preferred = metric_keys[0]
        if preferred is None:
            continue
        value = task_res.get(preferred)
        m = re.search(r'(\d+)$', task)
        seq_len = int(m.group(1)) if m else None
        base_task = task
        for L in sorted(lengths, reverse=True):
            suffix = f'_{L}'
            if task.endswith(suffix):
                base_task = task[:-len(suffix)]
                seq_len = L
                break
        if base_task not in task_names:
            continue
        stderr = None
        stderr_key = preferred + '_stderr,none'
        if stderr_key in task_res:
            stderr = task_res.get(stderr_key)
        rows.append({
            'task_full': task,
            'task': base_task,
            'context_length': seq_len,
            'metric': preferred,
            'score': value,
            'stderr': stderr,
            'n_samples': obj.get('n-samples', {}).get(task)
        })
    return rows


def plot_results(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    df = df.dropna(subset=['context_length', 'score']).copy()
    df['context_length'] = df['context_length'].astype(int)
    df['score_pct'] = df['score'] * 100
    summary = df.groupby('context_length', as_index=False)['score'].mean().rename(columns={'score': 'mean_score'})
    summary['mean_score_pct'] = summary['mean_score'] * 100

    sns.set_theme(style='whitegrid')
    created = []

    plt.figure(figsize=(10, 6))
    for task, g in df.sort_values('context_length').groupby('task'):
        plt.plot(g['context_length'], g['score_pct'], marker='o', linewidth=2, label=task)
    plt.xscale('log', base=2)
    plt.xticks(sorted(df['context_length'].unique()), [f'{int(x/1024)}K' if x >= 1024 else str(x) for x in sorted(df['context_length'].unique())])
    plt.ylabel('Score (%)')
    plt.xlabel('Context length')
    plt.title('RULER per-task performance vs context length')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    fp1 = out_dir / 'ruler_per_task_by_length.png'
    plt.savefig(fp1, dpi=220, bbox_inches='tight')
    plt.close()
    created.append(fp1)

    pivot = df.pivot_table(index='task', columns='context_length', values='score_pct', aggfunc='mean')
    plt.figure(figsize=(10, max(4, 0.65 * len(pivot))))
    sns.heatmap(
        pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index],
        annot=True,
        fmt='.1f',
        cmap='viridis',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Score (%)'}
    )
    plt.title('RULER heatmap')
    plt.xlabel('Context length')
    plt.ylabel('Task')
    plt.tight_layout()
    fp2 = out_dir / 'ruler_heatmap.png'
    plt.savefig(fp2, dpi=220, bbox_inches='tight')
    plt.close()
    created.append(fp2)

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(summary['context_length'], summary['mean_score_pct'], marker='o', linewidth=3, color='#146C94')
    plt.xscale('log', base=2)
    plt.xticks(summary['context_length'], [f'{int(x/1024)}K' if x >= 1024 else str(x) for x in summary['context_length']])
    plt.ylabel('Mean score (%)')
    plt.xlabel('Context length')
    plt.title('Mean RULER score by context length')
    plt.tight_layout()
    fp3 = out_dir / 'ruler_mean_score.png'
    plt.savefig(fp3, dpi=220, bbox_inches='tight')
    plt.close()
    created.append(fp3)

    return created


def write_markdown(df_csv: Path, report_md: Path, model_ref: str, backend: str, lengths, tasks):
    df = pd.read_csv(df_csv)
    if df.empty:
        report_md.write_text('# RULER report\n\nNo results found.\n')
        return

    dfx = df.dropna(subset=['context_length', 'score']).copy()
    dfx['context_length'] = dfx['context_length'].astype(int)
    dfx['score_pct'] = dfx['score'] * 100
    mean_by_len = dfx.groupby('context_length', as_index=False)['score_pct'].mean().sort_values('context_length')
    overall = dfx['score_pct'].mean()
    best = dfx.loc[dfx['score_pct'].idxmax()]
    worst = dfx.loc[dfx['score_pct'].idxmin()]

    useful = []
    for _, row in mean_by_len.iterrows():
        useful.append((int(row['context_length']), float(row['score_pct'])))
        if row['score_pct'] <= 1.0 and len(useful) >= 2:
            break

    lines = []
    lines.append(f'# RULER report for `{model_ref}`')
    lines.append('')
    lines.append('## Setup')
    lines.append(f'- Backend: `{backend}`')
    lines.append(f'- Tasks: `{", ".join(tasks)}`')
    lines.append(f'- Requested lengths: `{", ".join(str(x) for x in lengths)}`')
    lines.append('')
    lines.append('## Main findings')
    lines.append(f'- Mean score across all task/length pairs: **{overall:.2f}%**.')
    lines.append(f'- Best observed cell: `{best.task}` at `{int(best.context_length)}` tokens = **{best.score_pct:.2f}%**.')
    lines.append(f'- Worst observed cell: `{worst.task}` at `{int(worst.context_length)}` tokens = **{worst.score_pct:.2f}%**.')
    if useful:
        kept = ', '.join([f'{int(l/1024)}K={s:.1f}%' if l >= 1024 else f'{l}={s:.1f}%' for l, s in useful])
        lines.append(f'- Useful context range to inspect first: {kept}.')
    lines.append('')
    lines.append('## How to read')
    lines.append('- `ruler_mean_score.png`: global degradation with length; use it to detect where long-context collapse starts.')
    lines.append('- `ruler_per_task_by_length.png`: separates retrieval-heavy tasks from tracing/aggregation/QA behavior.')
    lines.append('- `ruler_heatmap.png`: fastest way to spot which tasks fail abruptly at a given context length.')
    lines.append('- `ruler_results.csv`: exhaustive long-format table, one row per task-length metric.')
    lines.append('')
    lines.append('## Interpretation hints')
    lines.append('- A smooth decay suggests context-length stress; a cliff suggests tokenizer/prompting/backend truncation or hard retrieval failure.')
    lines.append('- If only NIAH-style tasks survive while VT/CWE/QA collapse, the model retrieves isolated facts but struggles with composition over long context.')
    lines.append('- If scores are already near zero at 8K, testing 32K+ is usually not decision-relevant for this model size.')

    report_md.write_text('\n'.join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_ref = args.model or os.path.abspath(args.checkpoint)
    lengths = sorted(set(args.lengths or choose_lengths(args.max_length)))
    task_names = args.tasks
    model_type, model_args = build_model_args(args, model_ref)
    results_json = out_dir / 'lm_eval_results.json'

    cmd = [
        sys.executable, '-m', 'lm_eval',
        '--model', model_type,
        '--model_args', model_args,
        '--tasks', ','.join(task_names),
        '--batch_size', str(args.batch_size),
        '--output_path', str(results_json),
        '--metadata', json.dumps({'max_seq_lengths': lengths})
    ]
    if args.limit is not None:
        cmd += ['--limit', str(args.limit)]
    if args.fewshot_as_multiturn:
        cmd += ['--apply_chat_template', '--fewshot_as_multiturn']
    else:
        cmd += ['--apply_chat_template']

    run(cmd)

    with open(results_json, 'r') as f:
        obj = json.load(f)
    rows = extract_rows(obj, set(task_names), lengths)

    df = pd.DataFrame(rows).sort_values(['task', 'context_length'])
    csv_path = out_dir / 'ruler_results.csv'
    df.to_csv(csv_path, index=False)

    if not df.empty:
        mean_by_len = df.dropna(subset=['context_length', 'score']).groupby('context_length', as_index=False)['score'].mean().sort_values('context_length')
        trimmed_lengths = maybe_trim_lengths(mean_by_len['context_length'].tolist(), mean_by_len['score'].tolist())
        trim_json = out_dir / 'recommended_lengths.json'
        trim_json.write_text(json.dumps({'requested_lengths': lengths, 'recommended_lengths': trimmed_lengths}, indent=2))
    else:
        (out_dir / 'recommended_lengths.json').write_text(json.dumps({'requested_lengths': lengths, 'recommended_lengths': lengths}, indent=2))

    plot_results(csv_path, out_dir)
    write_markdown(csv_path, out_dir / 'REPORT.md', model_ref, args.backend, lengths, task_names)

    print(str(out_dir.resolve()))


if __name__ == '__main__':
    main()