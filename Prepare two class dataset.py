"""
Roboflow warehouse dataset(s) -> clean 2-class YOLO dataset (forklift, pallet) for Ultralytics YOLO11.

Run it in one Kaggle cell (paste the file) or as `python prepare_two_class_dataset.py`.
One or more source datasets are read (each folder has a data.yaml). The sources are never modified.
Everything is written to OUT_DIR.

Decisions (details in the printed report):
  * Class names are matched exactly after normalising case/spaces, so `pallet_truck` is never mistaken for `pallet`.
  * Images with a target object are kept; boxes of every other class are removed (they become background).
  * Images that contain ONLY other classes are hard negatives. They are kept as empty-label images, but only a
    capped, class-balanced sample (NEGATIVE_POLICY="cap"), and the sample is written to an audit sheet for a visual check.
  * Images with an empty label file, a missing label file or an unreadable image are dropped: we cannot tell whether
    the annotator saw the scene, so their "background" label is not trusted.
  * Boxes are validated, clipped to the image, and degenerate or duplicate boxes are removed. Polygons become boxes.
  * With several sources, each source's own test split is merged into train (a test split from one source would not
    contain both classes). A fresh test split is carved afterwards from the pooled data.
  * Sources that barely annotate a target class are flagged: objects of that class are unlabeled in its images.
  * The finished dataset is re-read from disk and checked independently of the code that wrote it.
"""
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# ------------------------------------------------------------------ config
SRC_DIRS = None                                  # None = auto-detect every dataset under /kaggle/input with a forklift or pallet class.
#                                                  Or list them: ["/kaggle/input/datasets/ranakagglecom/petrochoice", "/kaggle/input/datasets/ranakagglecom/forklift"]
OUT_DIR = Path("/kaggle/working/warehouse_2cls")
TARGETS = {"forklift": 0, "pallet": 1}           # normalised source name -> new class id
NEGATIVE_POLICY = "cap"                          # "drop" | "cap" | "keep"  (images with only non-target classes)
NEG_MAX_FRACTION = 0.10                          # cap: max negatives as a fraction of positive images, per split
SEED = 42
MIN_BOX = 1e-3                                   # normalised width/height below this = degenerate box
OVERLAP_IOU = 0.5                                # removed-class box ~ same object as a kept box
INSIDE_FRAC = 0.8                                # kept box mostly inside a removed-class box
MIN_CLASS_BOXES = 20                             # a source with fewer boxes of a target class is flagged
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}
PALETTE = {0: "#f2b705", 1: "#1f77b4"}
TARGET_NAMES = list(TARGETS)


def norm(s):
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


# ------------------------------------------------------------------ inspect
YAML_NAMES = ("data.yaml", "data.yml")


def yaml_file(root):
    for n in YAML_NAMES:
        if (Path(root) / n).exists():
            return Path(root) / n
    raise FileNotFoundError(f"no data.yaml or data.yml in {root}")


def read_names(root):
    names = yaml.safe_load(open(yaml_file(root)))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def find_roots():
    if SRC_DIRS:
        roots = []
        for s in SRC_DIRS:
            p = Path(s)
            roots += [p] if any((p / n).exists() for n in YAML_NAMES) else sorted({q.parent for n in YAML_NAMES for q in p.rglob(n)})
        if not roots:
            raise FileNotFoundError(f"No data.yaml under {SRC_DIRS}")
        return roots
    base = Path("/kaggle/input")
    roots = []
    for y in (sorted({q for n in YAML_NAMES for q in base.rglob(n)}) if base.exists() else []):
        try:
            if y.parent not in roots and {norm(n) for n in read_names(y.parent)} & set(TARGETS):
                roots.append(y.parent)
        except Exception:
            pass
    if not roots:
        raise FileNotFoundError("No dataset (data.yaml or data.yml) with a forklift or pallet class under /kaggle/input. Set SRC_DIRS.")
    return roots


def find_splits(root):
    splits = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        key = SPLIT_ALIASES.get(d.name.lower())
        if key and (d / "images").is_dir():
            n = sum(1 for f in (d / "images").iterdir() if f.suffix.lower() in IMG_EXT)
            if n:
                splits[key] = d
    return splits


# ------------------------------------------------------------------ labels
def parse_labels(path, n_cls, issues):
    """Return valid boxes as (source_class, xc, yc, w, h), normalised and clipped. Counts every problem in `issues`."""
    boxes, seen = [], set()
    for line in path.read_text(errors="ignore").splitlines():
        p = line.split()
        if not p:
            continue
        try:
            v = [float(t) for t in p]
        except ValueError:
            issues["bad_format"] += 1
            continue
        if not np.all(np.isfinite(v)):
            issues["bad_format"] += 1
            continue
        if v[0] != int(v[0]) or not (0 <= int(v[0]) < n_cls):
            issues["bad_class_id"] += 1
            continue
        c = int(v[0])
        if len(v) == 5:
            x, y, w, h = v[1:]
        elif len(v) >= 7 and len(v) % 2 == 1:           # polygon -> enclosing box
            xs, ys = v[1::2], v[2::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            issues["polygon_to_box"] += 1
        else:
            issues["bad_format"] += 1
            continue
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        if min(x0, y0) < -0.02 or max(x1, y1) > 1.02:
            issues["out_of_bounds_over_2pct"] += 1
        if min(x0, y0) < 0 or max(x1, y1) > 1:
            issues["clipped_to_image"] += 1
        x0, y0, x1, y1 = (min(max(t, 0.0), 1.0) for t in (x0, y0, x1, y1))
        w, h = x1 - x0, y1 - y0
        if w < MIN_BOX or h < MIN_BOX:
            issues["degenerate_box_removed"] += 1
            continue
        key = (c, round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
        if key in seen:
            issues["duplicate_box_removed"] += 1
            continue
        seen.add(key)
        boxes.append((c, (x0 + x1) / 2, (y0 + y1) / 2, w, h))
    return boxes


def _xyxy(b):
    return b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2


def overlap_flags(kept, excluded):
    """Removed-class boxes that may make a kept label ambiguous. Loads sitting on a pallet are normal and not flagged.
    `kept` boxes carry the new class id, `excluded` boxes carry the source class name."""
    out = []
    for e in excluded:
        ea = _xyxy(e)
        for k in kept:
            ka = _xyxy(k)
            iw = min(ea[2], ka[2]) - max(ea[0], ka[0])
            ih = min(ea[3], ka[3]) - max(ea[1], ka[1])
            if iw <= 0 or ih <= 0:
                continue
            inter, area_e, area_k = iw * ih, e[3] * e[4], k[3] * k[4]
            iou = inter / (area_e + area_k - inter)
            if iou >= OVERLAP_IOU:
                out.append(("same_object", e[0], TARGET_NAMES[k[0]], round(iou, 2)))
            elif inter / area_k >= INSIDE_FRAC:
                out.append(("target_inside_removed_class", e[0], TARGET_NAMES[k[0]], round(inter / area_k, 2)))
    return out


# ------------------------------------------------------------------ per split
def load_split(src, src_split, d, issues):
    """Read one split of one source. Returns records; counts problems in `issues`."""
    norm_names = src["norm"]
    src_to_tgt = {i: TARGETS[n] for i, n in enumerate(norm_names) if n in TARGETS}
    recs = []
    imgs = sorted(f for f in (d / "images").iterdir() if f.suffix.lower() in IMG_EXT)
    labs = {f.stem: f for f in (d / "labels").glob("*.txt")} if (d / "labels").is_dir() else {}
    issues["label_file_without_image"] += len(set(labs) - {f.stem for f in imgs})
    for f in imgs:
        out_name = f"{src['tag']}__{f.name}"
        rec = dict(source=src["tag"], src_split=src_split, img=f, out_name=out_name, stem=Path(out_name).stem,
                   kept=[], excl=[], src=Counter(), flags=[], status="")
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                rec["size"] = im.size
        except Exception:
            issues["corrupt_image"] += 1
            rec["status"] = "dropped:corrupt_image"
            recs.append(rec)
            continue
        lab = labs.get(f.stem)
        if lab is None:
            issues["image_without_label_file"] += 1
            rec["status"] = "dropped:missing_label_file"
            recs.append(rec)
            continue
        boxes = parse_labels(lab, len(norm_names), issues)
        rec["src"] = Counter(norm_names[b[0]] for b in boxes)
        rec["kept"] = [(src_to_tgt[b[0]],) + b[1:] for b in boxes if b[0] in src_to_tgt]
        rec["excl"] = [(norm_names[b[0]],) + b[1:] for b in boxes if b[0] not in src_to_tgt]
        if rec["kept"]:
            rec["status"] = "positive"
            rec["flags"] = overlap_flags(rec["kept"], rec["excl"]) if rec["excl"] else []
        elif rec["excl"]:
            rec["status"] = "negative_candidate"
        else:
            issues["empty_label_file"] += 1
            rec["status"] = "dropped:empty_label_file"
        recs.append(rec)
    return recs


def select_negatives(recs):
    """Apply NEGATIVE_POLICY to images that contain only non-target classes."""
    n_pos = sum(r["status"] == "positive" for r in recs)
    cands = [r for r in recs if r["status"] == "negative_candidate"]
    n_keep = {"drop": 0, "keep": len(cands), "cap": int(n_pos * NEG_MAX_FRACTION)}[NEGATIVE_POLICY]
    rng = random.Random(SEED)
    groups = {}
    for r in cands:                                     # balance by the most frequent removed class in the image
        dom = Counter(b[0] for b in r["excl"]).most_common(1)[0][0]
        groups.setdefault((r["source"], dom), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    chosen = []
    while len(chosen) < n_keep and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(chosen) < n_keep:
                chosen.append(groups[k].pop())
    chosen_ids = {id(r) for r in chosen}
    for r in cands:
        r["status"] = "negative_kept" if id(r) in chosen_ids else "dropped:excluded_only"


def write_split(split, recs):
    oi, ol = OUT_DIR / split / "images", OUT_DIR / split / "labels"
    oi.mkdir(parents=True, exist_ok=True)
    ol.mkdir(parents=True, exist_ok=True)
    for r in recs:
        if r["status"] not in ("positive", "negative_kept"):
            continue
        shutil.copy2(r["img"], oi / r["out_name"])
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["kept"]]
        (ol / f"{r['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


# ------------------------------------------------------------------ verify (reads the output from disk)
def verify_output(written, expected):
    rows, problems = [], []
    for split in written:
        imgs = {f.stem for f in (OUT_DIR / split / "images").iterdir() if f.suffix.lower() in IMG_EXT}
        labs = {f.stem: f for f in (OUT_DIR / split / "labels").glob("*.txt")}
        if imgs != set(labs):
            problems.append(f"{split}: image/label mismatch (images only: {len(imgs - set(labs))}, labels only: {len(set(labs) - imgs)})")
        counts, bad = Counter(), 0
        for f in labs.values():
            for line in f.read_text().splitlines():
                try:
                    p = line.split()
                    c, (x, y, w, h) = int(p[0]), map(float, p[1:])
                    ok = len(p) == 5 and c in (0, 1) and w > 0 and h > 0 and min(x - w / 2, y - h / 2) >= -1e-6 \
                        and max(x + w / 2, y + h / 2) <= 1 + 1e-6
                except Exception:
                    ok = False
                if ok:
                    counts[c] += 1
                else:
                    bad += 1
        if bad:
            problems.append(f"{split}: {bad} invalid label lines")
        if counts != expected[split]:
            problems.append(f"{split}: annotation counts on disk {dict(counts)} differ from expected {dict(expected[split])}")
        rows.append(dict(split=split, images=len(imgs), label_files=len(labs), forklift=counts[0], pallet=counts[1], invalid_lines=bad))
    cfg = yaml.safe_load(open(OUT_DIR / "data.yaml"))
    if cfg["names"] != {0: "forklift", 1: "pallet"}:
        problems.append(f"data.yaml names are {cfg['names']}")
    for key in ("train", "val"):
        if not (OUT_DIR / cfg[key]).is_dir():
            problems.append(f"data.yaml path for {key} does not exist: {cfg[key]}")
    try:                                                # optional: Ultralytics' own dataset check
        from ultralytics.data.utils import check_det_dataset
        check_det_dataset(str(OUT_DIR / "data.yaml"))
        rows.append(dict(split="ultralytics check_det_dataset", images="ok"))
    except ImportError:
        pass
    except Exception as e:
        problems.append(f"ultralytics check_det_dataset failed: {e}")
    return pd.DataFrame(rows), problems


# ------------------------------------------------------------------ audit sheet
def audit_sheet(recs, path, title, max_n=12):
    import matplotlib.pyplot as plt
    recs = recs[:max_n]
    if not recs:
        return None
    cols = 4
    rows = -(-len(recs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, recs):
        im = Image.open(r["img"]).convert("RGB")
        W, H = im.size
        ax.imshow(im)
        for b in r["kept"]:
            x0, y0, x1, y1 = (v * s for v, s in zip(_xyxy(b), (W, H, W, H)))
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, ec=PALETTE[b[0]]))
            ax.text(x0, y0, TARGET_NAMES[b[0]], fontsize=7, bbox=dict(fc=PALETTE[b[0]], lw=0, pad=1))
        for b in r["excl"]:
            x0, y0, x1, y1 = (v * s for v, s in zip(_xyxy(b), (W, H, W, H)))
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, ls="--", ec="#d62728"))
            ax.text(x0, y0, b[0], fontsize=7, color="white", bbox=dict(fc="#d62728", lw=0, pad=1))
        ax.set_title(f"{r['source']} / {r['img'].name[:26]}", fontsize=7)
    fig.suptitle(title + "   (solid = kept labels, dashed red = removed classes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main
def main():
    roots = find_roots()
    sources = []
    for i, root in enumerate(roots):
        names = read_names(root)
        nn = [norm(n) for n in names]
        if not set(nn) & set(TARGETS):
            print(f"[skip] {root}: none of {TARGET_NAMES} in classes {names}")
            continue
        if OUT_DIR.resolve() == root.resolve() or root.resolve() in OUT_DIR.resolve().parents:
            raise ValueError("OUT_DIR must be outside the source dataset")
        tag = f"s{i}_" + re.sub(r"[^A-Za-z0-9]+", "-", root.name).strip("-")[:20]
        sources.append(dict(root=root, tag=tag, names=names, norm=nn, splits=find_splits(root)))
    if not sources:
        raise ValueError("No usable source dataset")

    for s in sources:
        print(f"Source {s['tag']}: {s['root']}")
        print("  classes :", {i: n for i, n in enumerate(s["names"])})
        print("  action  :", {n: (f"keep -> {TARGETS[norm(n)]}" if norm(n) in TARGETS else "remove") for n in s["names"]})
        print("  splits  :", {k: sum(1 for f in (d / 'images').iterdir() if f.suffix.lower() in IMG_EXT) for k, d in s["splits"].items()})
    covered = {t for s in sources for t in TARGETS if t in s["norm"]}
    if covered != set(TARGETS):
        raise ValueError(f"Target classes {sorted(set(TARGETS) - covered)} not found in any source")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    pooled, issues = {"train": [], "valid": []}, {"train": Counter(), "valid": Counter()}
    merged_test = 0
    for s in sources:
        for split, d in s["splits"].items():
            tgt = "valid" if split == "valid" else "train"      # source test splits are merged into train
            merged_test += split == "test"
            pooled[tgt] += load_split(s, split, d, issues[tgt])
    if not pooled["valid"]:
        raise ValueError("No validation images in any source")
    if merged_test:
        print(f"NOTE: {merged_test} source test split(s) merged into train. A fresh test split is carved from the pooled data later.")
    else:
        print("NOTE: no test split in the sources. A test split is carved from train later.")

    split_rows, expected = [], {}
    for split, recs in pooled.items():
        select_negatives(recs)
        write_split(split, recs)
        st = Counter(r["status"] for r in recs)
        split_rows.append(dict(
            split=split, source_images=len(recs), positive=st["positive"], negatives_kept=st["negative_kept"],
            output_images=st["positive"] + st["negative_kept"],
            dropped_excluded_only=st["dropped:excluded_only"], dropped_empty_label=st["dropped:empty_label_file"],
            dropped_missing_label=st["dropped:missing_label_file"], dropped_corrupt=st["dropped:corrupt_image"]))
        expected[split] = Counter(b[0] for r in recs if r["status"] in ("positive", "negative_kept") for b in r["kept"])

    cfg = {"path": str(OUT_DIR), "train": "train/images", "val": "valid/images",
           "nc": 2, "names": {0: "forklift", 1: "pallet"}}
    yaml.safe_dump(cfg, open(OUT_DIR / "data.yaml", "w"), sort_keys=False)

    # ---- report
    flat = [r for recs in pooled.values() for r in recs]
    all_names = []
    for s in sources:
        all_names += [n for n in s["norm"] if n not in all_names]
    ann_rows, cov = [], {}
    for s in sources:
        for split, recs in pooled.items():
            rs = [r for r in recs if r["source"] == s["tag"]]
            if not rs:
                continue
            src = Counter()
            for r in rs:
                src.update(r["src"])
            kept = Counter(b[0] for r in rs if r["status"] in ("positive", "negative_kept") for b in r["kept"])
            ann_rows.append({"source": s["tag"], "split": split, "images": len(rs), **{f"src_{n}": src[n] for n in all_names},
                             "out_forklift": kept[0], "out_pallet": kept[1]})
        cov[s["tag"]] = Counter(b[0] for r in flat if r["source"] == s["tag"] and r["status"] in ("positive", "negative_kept")
                                for b in r["kept"])
    split_df, ann_df = pd.DataFrame(split_rows), pd.DataFrame(ann_rows)
    issues_df = pd.DataFrame(issues).fillna(0).astype(int)
    print("\n== Images per split (all sources pooled) ==");           print(split_df.to_string(index=False))
    print("\n== Annotations per source and split: source classes -> output ==");  print(ann_df.to_string(index=False))
    if len(issues_df):
        print("\n== Data issues found in the sources (rows = issue, columns = split) ==");  print(issues_df.to_string())

    pos_with_excl = [r for r in flat if r["status"] == "positive" and r["excl"]]
    unlabeled = Counter(b[0] for r in pos_with_excl for b in r["excl"])
    flagged = [r for r in flat if r["status"] == "positive" and r["flags"]]
    neg_kept = [r for r in flat if r["status"] == "negative_kept"]
    dropped_excl = [r for r in flat if r["status"] == "dropped:excluded_only"]
    dropped_other = [r for r in flat if r["status"].startswith("dropped:") and r["status"] != "dropped:excluded_only"]

    print("\n== Flags ==")
    n = 0

    def flag(msg):
        nonlocal n
        n += 1
        print(f"{n}. {msg}")

    tot = Counter()
    for c in cov.values():
        tot.update(c)
    for t, tid in TARGETS.items():
        if tot[tid] < MIN_CLASS_BOXES * 5:
            flag(f"CRITICAL: class '{t}' has only {tot[tid]} boxes in the whole output. A detector cannot learn it reliably. "
                 f"Add a dataset that annotates it.")
    for s in sources:
        for t, tid in TARGETS.items():
            if cov[s["tag"]][tid] < MIN_CLASS_BOXES:
                n_img = sum(1 for r in flat if r["source"] == s["tag"])
                flag(f"Source {s['tag']} has {cov[s['tag']][tid]} '{t}' boxes. Any '{t}' visible in its {n_img} images is unlabeled, "
                     f"so it counts as background when training.")
    flag(f"{len(pos_with_excl)} kept images still contain removed-class objects; {sum(unlabeled.values())} boxes are now "
         f"background: {dict(unlabeled)}. The model will learn that they are not forklift/pallet.")
    flag(f"{len(flagged)} kept images have a removed-class box that overlaps a kept box (possible label ambiguity). "
         f"Review audit_overlaps.png and overlap_flags.csv.")
    flag(f"Images with only removed classes: {len(neg_kept)} kept as negatives, {len(dropped_excl)} dropped "
         f"(policy='{NEGATIVE_POLICY}'). Check audit_negatives.png: a target visible there would be a false negative label.")
    flag(f"{len(dropped_other)} images dropped for untrusted labels or unreadable files (see dropped_images.csv).")
    vals = [tot[0], tot[1]]
    ratio = max(vals) / max(1, min(vals))
    flag(f"Class balance after filtering: forklift={tot[0]}, pallet={tot[1]} (ratio {ratio:.1f}:1)."
         + (" Evaluate per class." if ratio > 3 else ""))
    if len(sources) > 1:
        flag("Several sources are merged: the model can pick up source style (camera, site) as a shortcut for the class. "
             "Look at false positives per source.")

    # ---- audit files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(source=r["source"], split=r["src_split"], image=r["img"].name, reason=r["status"].split(":")[1], removed_classes=dict(r["src"]))
                  for r in flat if r["status"].startswith("dropped:")]).to_csv(OUT_DIR / "dropped_images.csv", index=False)
    pd.DataFrame([dict(source=r["source"], image=r["img"].name, kind=f[0], removed_class=f[1], kept_class=f[2], value=f[3])
                  for r in flagged for f in r["flags"]]).to_csv(OUT_DIR / "overlap_flags.csv", index=False)
    ann_df.to_csv(OUT_DIR / "class_distribution.csv", index=False)
    sheets = [audit_sheet(neg_kept, OUT_DIR / "audit_negatives.png", "Negatives kept for training"),
              audit_sheet(flagged, OUT_DIR / "audit_overlaps.png", "Overlap flags")]
    try:
        from IPython.display import Image as IPImage, display
        for sh in sheets:
            if sh:
                display(IPImage(filename=str(sh)))
    except Exception:
        pass

    # ---- verify from disk
    ver_df, problems = verify_output(list(pooled), expected)
    print("\n== Verification (re-read from disk) ==");  print(ver_df.fillna("").to_string(index=False))
    print("PROBLEMS:" if problems else "All checks passed: image/label pairs, class ids 0/1, box ranges, counts, data.yaml.")
    for p in problems:
        print("  -", p)

    json.dump(dict(sources=[s["tag"] for s in sources], policy=NEGATIVE_POLICY, splits=split_rows, annotations=ann_rows,
                   issues={k: dict(v) for k, v in issues.items()}, problems=problems),
              open(OUT_DIR / "report.json", "w"), indent=2, default=str)
    print(f"\nReady: {OUT_DIR / 'data.yaml'}")
    return OUT_DIR / "data.yaml"


if __name__ == "__main__":
    DATA_YAML = main()
