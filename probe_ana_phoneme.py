#!/usr/bin/env python3
"""测：Ana 对不同写法的发音，哪种能逼出「单个音素」。"""
import asyncio, os, subprocess, wave
import numpy as np
import edge_tts

OUT = "/tmp/ana_probe"
os.makedirs(OUT, exist_ok=True)
VOICE = "en-US-AnaNeural"

CASES = {
    # 目标 æ（cat 里的 a）
    "ae_ipa":  "æ",
    "ae_a":    "a",
    "ae_ah":   "ah",
    "ae_at":   "at",
    # 目标 s
    "s_sss":   "sss",
    "s_s":     "s",
    "s_sun":   "sun",
    # 目标 ʃ
    "sh_sh":   "sh",
    "sh_ssh":  "shh",
    "sh_ship": "ship",
    # 目标 m
    "m_mmm":   "mmm",
    "m_m":     "m",
    "m_man":   "man",
}


async def gen(name, text):
    c = edge_tts.Communicate(text, VOICE, rate="-10%")
    p = os.path.join(OUT, name + ".mp3")
    with open(p, "wb") as f:
        async for chunk in c.stream():
            if chunk.get("type") == "audio":
                f.write(chunk["data"])
    return p


def stats(p):
    w = "/tmp/_p.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-ac", "1", "-ar", "22050", w], check=True)
    f = wave.open(w)
    a = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if not len(a):
        return 0, 0, 0
    # 去掉首尾静音后的有效时长
    st = int(0.01 * 22050)
    fr = [np.sqrt((a[i:i + st] ** 2).mean()) for i in range(0, len(a) - st, st)]
    fr = np.array(fr)
    thr = max(0.01, 0.1 * fr.max())
    idx = np.where(fr > thr)[0]
    voiced = (idx[-1] - idx[0] + 1) * 0.01 if len(idx) else 0
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    fq = np.fft.rfftfreq(len(a), 1 / 22050)
    return len(a) / 22050, voiced, ((sp * fq).sum() / sp.sum(), (sp[fq > 3000] ** 2).sum() / (sp ** 2).sum())


async def main():
    for n, t in CASES.items():
        try:
            p = await gen(n, t)
            dur, voiced, (cen, hi) = stats(p)
            print(f"{n:10s} 文本={t!r:8s} 总{dur:5.2f}s 有效{voiced:4.2f}s 重心{cen:5.0f}Hz 高频{hi:.2f}")
        except Exception as e:
            print(f"{n}: {type(e).__name__} {e}")


asyncio.run(main())
