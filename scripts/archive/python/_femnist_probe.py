"""Validate the FEMNIST writer-partitioned data path before running the experiment grid.
Checks: writers load, local 80/20 split sizes, image tensor shape (1x28x28), label range."""
import sys, time
sys.path.insert(0, ".")
from qpriviot_fl.task import load_data
import torch

t0 = time.time()
print("Loading FEMNIST writer partitions (first download may take several minutes)...", flush=True)
sizes = []
for pid in [0, 1, 2, 49, 99, 199]:
    tr, va = load_data(partition_id=pid, num_partitions=200, batch_size=32,
                       dataset_name="femnist", alpha=0.3, seed=42)
    ntr, nva = len(tr.dataset), len(va.dataset)
    sizes.append(ntr + nva)
    batch = next(iter(tr))
    x, y = batch["img"], batch["label"]
    print(f"  writer pid={pid:>3}: train={ntr:>4} val={nva:>4} total={ntr+nva:>4} "
          f"| x.shape={tuple(x.shape)} dtype={x.dtype} "
          f"| y range=[{int(min(y))},{int(max(y))}] n_uniq_labels={len(set(int(v) for v in y))}",
          flush=True)
print(f"\nper-writer total samples: min={min(sizes)} max={max(sizes)} "
      f"mean={sum(sizes)/len(sizes):.0f}")
print(f"OK in {time.time()-t0:.0f}s")