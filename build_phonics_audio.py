#!/usr/bin/env python3
"""从 Jolly Phonics 官方 42 音录音里切出单个音素短音频。

源包: https://jollylearning.com/hubfs/Jolly_Songs_Audio_AE.zip (美音, 39 个 .aif)
每个 .aif 是「纯音素重复 + 儿歌」，我们只取开头纯音素那一小段。

用法: python3 build_phonics_audio.py /tmp/jolly
输出: phonics/<slug>.mp3   （slug 与 index.html 的 IPA_SLUG 对齐）
"""
import os, sys, wave, subprocess, shutil
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/jolly"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phonics")
SR = 22050
FRAME = 0.02          # 20ms 分析帧
CUT_SEC = 0.85        # 目标「有声」时长（含重复，孩子多听几遍有好处）
MIN_SEC = 0.30        # 太短的音（p/t/k 这类爆破音）也至少留这么长
MAX_SEC = 1.00        # 太长就截断，不然一个词拼读拖 10 秒孩子会走神

# 文件编号 -> 输出 slug 列表（组合文件按顺序给两个 slug）
MAP = {
    "01": ["s"], "02": ["a"], "03": ["t"], "04": ["i"], "05": ["p"], "06": ["n"],
    "07": ["k"], "08": ["e"], "09": ["h"], "10": ["r"], "11": ["m"], "12": ["d"],
    "13": ["g"], "14": ["o"], "15": ["u"], "16": ["l"], "17": ["f"], "18": ["b"],
    "19": ["ai"], "20": ["j"], "21": ["oa"], "22": ["ie"],
    "23": ["ee", "or"],
    "24": ["z"], "25": ["w"], "26": ["ng"], "27": ["v"],
    "28": ["oo", "oo-short"],
    "29": ["y"], "30": ["x"], "31": ["ch"], "32": ["sh"],
    "33": ["th", "th2"],
    "34": ["qu"], "35": ["ou"], "36": ["oi"], "37": ["ue"], "38": ["er"], "39": ["ar"],
}


def load(path):
    """统一用 ffmpeg 解码成 22.05k 单声道 wav（aif 也能读）"""
    tmp = path + ".tmp.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), tmp], check=True)
    w = wave.open(tmp)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    os.remove(tmp)
    return a


def rms_frames(a):
    step = int(SR * FRAME)
    k = len(a) // step
    return np.sqrt((a[: k * step].reshape(k, step) ** 2).mean(axis=1))


def voiced(r):
    thr = max(0.008, 0.10 * r.max())
    return r > thr


def biggest_gap(v, lo, hi):
    """在 [lo,hi) 帧区间里找最长的静音段，返回中点帧号；没有就返回中点。"""
    best, start, cur = None, None, None
    for i in range(lo, hi):
        if not v[i]:
            if cur is None:
                cur = i
        else:
            if cur is not None:
                span = i - cur
                if best is None or span > best[0]:
                    best = (span, cur + span // 2)
                cur = None
    if cur is not None:
        span = hi - cur
        if best is None or span > best[0]:
            best = (span, cur + span // 2)
    if best is None or best[0] < 8:      # 少于 160ms 的缝不算分界
        return (lo + hi) // 2
    return best[1]


def cut_one(a, f0, f1):
    """在一段里取开头纯音素：从第一个有声帧起，攒够 CUT_SEC 秒「有声」为止。
    中间的短缝（音素重复之间的停顿）保留，遇到 0.4s 以上的长静音就收。"""
    r = rms_frames(a[f0:f1])
    v = voiced(r)
    idx = np.where(v)[0]
    if not len(idx):
        return None
    start = idx[0]
    need = int(CUT_SEC / FRAME)          # 需要多少个有声帧
    got, gap, last_voiced, end = 0, 0, start, start
    for i in range(start, len(v)):
        if v[i]:
            got += 1
            last_voiced = i
            gap = 0
            if got >= need:
                end = i + 1
                break
        else:
            gap += 1
            if got and gap >= int(0.55 / FRAME):   # 长静音：一句结束了
                end = last_voiced + 1
                break
    if not end:
        end = last_voiced + 1
    end = min(end + 3, len(v))            # 补 60ms 尾巴，别把尾音切掉
    step = int(SR * FRAME)
    return a[f0 + start * step: f0 + end * step]


def write_mp3(seg, slug):
    wav = os.path.join(OUT, f"_{slug}.wav")
    with wave.open(wav, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((seg * 32767).astype(np.int16).tobytes())
    mp3 = os.path.join(OUT, f"{slug}.mp3")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "afade=t=in:d=0.02,afade=t=out:st=" + f"{max(0,len(seg)/SR-0.05):.3f}" + ":d=0.05",
                    "-ac", "1", "-ar", "22050", "-b:a", "64k", mp3], check=True)
    os.remove(wav)
    return mp3, len(seg) / SR


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(".aif"))
    made = []
    for f in files:
        num = f.split()[0]
        slugs = MAP.get(num)
        if not slugs:
            print("!! 未映射:", f); continue
        a = load(os.path.join(SRC, f))
        r = rms_frames(a)
        v = voiced(r)
        if len(slugs) == 1:
            segs = [cut_one(a, 0, len(a))]
        else:
            mid = biggest_gap(v, len(v) // 4, len(v) * 3 // 4)
            segs = [cut_one(a, 0, mid * int(SR * FRAME)),
                    cut_one(a, mid * int(SR * FRAME), len(a))]
        for slug, seg in zip(slugs, segs):
            if seg is None or len(seg) < SR // 20:
                print(f"!! {slug}: 没切出声，跳过"); continue
            # 太长截断（留 1 秒听 1-2 遍就够）
            if len(seg) > MAX_SEC * SR:
                seg = seg[:int(MAX_SEC * SR)]
            # 太短的（爆破音）循环补到 MIN_SEC，避免一闪而过
            while len(seg) < MIN_SEC * SR:
                seg = np.concatenate([seg, np.zeros(int(0.06 * SR), dtype=np.float32), seg])
            mp3, dur = write_mp3(seg, slug)
            made.append((slug, dur))
    # 复合音 /ts/（pizza 的 zz）：t + s 拼起来
    ts = os.path.join(OUT, "ts.mp3")
    if os.path.exists(os.path.join(OUT, "t.mp3")) and os.path.exists(os.path.join(OUT, "s.mp3")):
        lst = os.path.join(OUT, "_ts.txt")
        with open(lst, "w") as f:
            f.write(f"file '{os.path.join(OUT,'t.mp3')}'\nfile '{os.path.join(OUT,'s.mp3')}'\n")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-t", "0.5", "-ac", "1", "-ar", "22050", "-b:a", "64k", ts], check=True)
        os.remove(lst)
    # 别名：c 就是 k；happy 元音 /i/ 借用 ee；/ts/ 用 t+s 拼
    alias = {"c": "k", "i": "i", "schwa": "er"}
    print(f"\n生成 {len(made)} 个: " + " ".join(f"{s}({d:.2f}s)" for s, d in made))
    if len(made) < 30:
        sys.exit("生成数量异常")


if __name__ == "__main__":
    main()
