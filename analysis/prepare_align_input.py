"""Convert Switchboard conversations to ALIGN tab-separated input format.

Output: one .txt file per conversation with columns: participant\tcontent
Usage: python analysis/prepare_align_input.py --n 30 --out /tmp/align_sw_input
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.swda import parse_conversation, iter_conversation_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of conversations")
    parser.add_argument("--out", default="/tmp/align_sw_input")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    count = 0
    for fp in iter_conversation_files():
        if count >= args.n:
            break
        turns = parse_conversation(fp)
        if len(turns) < 4:
            continue
        fname = fp.name.replace(".csv", ".txt")
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write("participant\tcontent\n")
            for speaker, text in turns:
                text_clean = text.strip().replace("\t", " ")
                if text_clean:
                    f.write(f"{speaker}\t{text_clean}\n")
        count += 1

    print(f"Wrote {count} conversations to {args.out}")
    sample = sorted(os.listdir(args.out))[0]
    print(f"Sample ({sample}):")
    with open(os.path.join(args.out, sample)) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
