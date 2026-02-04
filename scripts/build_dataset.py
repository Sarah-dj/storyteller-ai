from pathlib import Path
import random

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "questcrafter_pairs.jsonl"
OUT_TRAIN = BASE / "data" / "train" / "train.jsonl"
OUT_VAL = BASE / "data" / "train" / "val.jsonl"

SEED = 42
VAL_RATIO = 0.02  # 2% validation

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing: {RAW}")

    # Read JSONL lines as-is (each line is {"text": ...})
    lines = RAW.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]

    random.seed(SEED)
    random.shuffle(lines)

    n_val = max(1, int(len(lines) * VAL_RATIO))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_VAL.parent.mkdir(parents=True, exist_ok=True)

    OUT_TRAIN.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    OUT_VAL.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"✅ Read {len(lines)} samples from {RAW}")
    print(f"✅ Wrote {len(train_lines)} train → {OUT_TRAIN}")
    print(f"✅ Wrote {len(val_lines)} val   → {OUT_VAL}")

if __name__ == "__main__":
    main()