"""One persistent stratified split, bound to torchvision's sample ordering.

Decoded uint8 tensors are cached to avoid repeated JPEG decoding on Windows.
Only training batches receive independent horizontal flips. All sets use the
same fixed ImageNet normalization, which estimates nothing from held-out data.
"""
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import EuroSAT
from sklearn.model_selection import train_test_split
from config import ROOT, SEED, BATCH_SIZE, MEAN, STD
from src.utils import write_json, read_json, sha256, log

def prepare_data():
    ds = EuroSAT(str(ROOT / "data"), download=True)
    labels = np.asarray(ds.targets)
    manifest = [{"path": str(__import__('pathlib').Path(p).relative_to(ds.root)),
                 "label": int(y)} for p, y in ds.samples]
    splitdir = ROOT / "data/splits"
    splitdir.mkdir(parents=True, exist_ok=True)
    manifest_path = splitdir / "sample_manifest.json"
    if manifest_path.exists():
        assert read_json(manifest_path) == manifest, "Dataset ordering changed"
    else:
        write_json(manifest_path, manifest)
    paths = [splitdir / f"{s}_indices.json" for s in ("train", "val", "test")]
    if not any(p.exists() for p in paths):
        train, holdout = train_test_split(np.arange(len(ds)), test_size=0.30,
                                         random_state=SEED, stratify=labels)
        val, test = train_test_split(holdout, test_size=0.5,
                                   random_state=SEED, stratify=labels[holdout])
        for path, indices in zip(paths, (train, val, test)):
            write_json(path, sorted(indices.tolist()))
    assert all(p.exists() for p in paths), "Incomplete saved split; investigate before replacing"
    groups = [read_json(p) for p in paths]
    assert [len(g) for g in groups] == [18900, 4050, 4050]
    assert len(set(sum(groups, []))) == len(ds) == 27000
    assert sorted(sum(groups, [])) == list(range(len(ds)))
    counts = {name: {"total": int((labels == i).sum()),
               **{s: int((labels[idx] == i).sum()) for s, idx in zip(("train", "val", "test"), groups)}}
              for i, name in enumerate(ds.classes)}
    for c in counts.values():
        assert c["train"] == c["total"] * .70
        assert c["val"] == c["test"] == c["total"] * .15
    info = {"total_images": len(ds), "num_classes": len(ds.classes), "classes": ds.classes,
            "class_counts": counts, "split_sizes": dict(zip(("train", "val", "test"), map(len, groups))),
            "seed": SEED, "manifest_sha256": sha256(manifest_path),
            "split_sha256": {p.stem: sha256(p) for p in paths}}
    cache = ROOT / "data/rgb_uint8.pt"
    if not cache.exists():
        images = torch.empty((len(ds), 3, 64, 64), dtype=torch.uint8)
        content_hashes = []
        for i in range(len(ds)):
            im, _ = ds[i]
            arr = np.asarray(im)
            assert arr.shape == (64, 64, 3)
            images[i] = torch.from_numpy(arr.copy()).permute(2, 0, 1)
            content_hashes.append(hashlib.sha256(arr.tobytes()).hexdigest())
            if i % 5000 == 0:
                print(f"Decoded {i}/{len(ds)}", flush=True)
        # Stronger than index disjointness: detect pixel-identical images crossing sets.
        sets = [set(content_hashes[i] for i in g) for g in groups]
        overlaps = sum(len(sets[a] & sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        assert overlaps == 0, "Identical image content crosses split boundaries"
        write_json(splitdir / "content_audit.json", {"unique_images": len(set(content_hashes)),
                   "cross_split_identical_images": overlaps})
        torch.save({"images": images, "labels": torch.tensor(labels),
                    "manifest_sha256": info["manifest_sha256"]}, cache)
    write_json(splitdir / "dataset_info.json", info)
    print(info, flush=True)
    log("EuroSAT prepared: 27,000 RGB images; stratified 18,900/4,050/4,050 split; saved index and manifest hashes.")
    return info

def loaders(batch_size=BATCH_SIZE, seed=SEED, splits=("train", "val", "test")):
    info = read_json(ROOT / "data/splits/dataset_info.json")
    cache = torch.load(ROOT / "data/rgb_uint8.pt", weights_only=True)
    assert cache["manifest_sha256"] == info["manifest_sha256"]
    result = {}
    for split in splits:
        path = ROOT / f"data/splits/{split}_indices.json"
        assert sha256(path) == info["split_sha256"][f"{split}_indices"]
        idx = torch.tensor(read_json(path))
        ds = TensorDataset(cache["images"][idx], cache["labels"][idx])
        result[split] = DataLoader(ds, batch_size=batch_size, shuffle=split == "train",
                                  num_workers=0, pin_memory=torch.cuda.is_available(),
                                  generator=torch.Generator().manual_seed(seed))
    return result

def preprocess(x, device, training=False):
    x = x.to(device, non_blocking=True).float().div_(255)
    if training:
        flip = torch.rand((len(x), 1, 1, 1), device=device) < .5
        x = torch.where(flip, x.flip(-1), x)
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x - mean) / std
