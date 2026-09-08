import glob, pandas as pd, re
rows = []
for f in glob.glob('experiment_results/*_per_client.csv'):
    df = pd.read_csv(f)
    m = re.search(r'results_(\w+)_([\w-]+)_n(\d+)_alpha([\d.]+)_seed(\d+)_eps([\d.]+)', f)
    if m:
        df['dataset'], df['config'], df['num_clients'] = m.group(1), m.group(2), int(m.group(3))
        df['alpha'], df['seed'], df['epsilon'] = float(m.group(4)), int(m.group(5)), float(m.group(6))
    rows.append(df)
if rows:
    pd.concat(rows).to_csv('test_artifacts/per_client_metrics.csv', index=False)
    print('wrote test_artifacts/per_client_metrics.csv')
else:
    print('No per-client CSVs found.')
