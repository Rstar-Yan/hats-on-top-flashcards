#!/usr/bin/env python3
"""用 edge-tts 命令行批量生成单词音频（en-US-AnaNeural 可爱女童音）。
产出 audio/<key>.mp3。用 subprocess 调 CLI，避开 python 版本/模块路径问题。
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
EDGE = os.path.expanduser("~/Library/Python/3.9/bin/edge-tts")
VOICE = os.environ.get("HOT_VOICE", "en-US-AnaNeural")
RATE = os.environ.get("HOT_RATE", "-10%")
OUT = os.path.join(BASE, "audio")


def key_of(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", word.lower())


def load_words():
    with open(os.path.join(BASE, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    return sorted(set(re.findall(r'\["([^"]+)","[^"]*","[^"]+"\]', html)))


def gen(word: str):
    path = os.path.join(OUT, key_of(word) + ".mp3")
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return ("skip", word)
    err_msg = ""
    for _ in range(2):
        try:
            r = subprocess.run(
                [EDGE, "--voice", VOICE, "--rate=" + RATE,
                 "--text", word, "--write-media", path],
                check=False, capture_output=True, timeout=30,
            )
            if os.path.exists(path) and os.path.getsize(path) > 200:
                return ("ok", word)
            err_msg = (r.stderr or b"").decode("utf-8", "ignore").strip()[:140]
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)[:140]
    return ("err", word + " | " + err_msg)


def main():
    os.makedirs(OUT, exist_ok=True)
    words = load_words()
    print("words:", len(words), "| voice:", VOICE, "| rate:", RATE, flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(gen, words))
    ok = sum(1 for r in results if r[0] in ("ok", "skip"))
    errs = [r[1] for r in results if r[0] == "err"]
    print("generated: %d | errors: %d" % (ok, len(errs)), flush=True)
    for e in errs[:10]:
        print("  ERR:", e)
    files = [f for f in os.listdir(OUT) if f.endswith(".mp3")]
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in files)
    print("mp3 files: %d | %.0f KB" % (len(files), total / 1024))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
