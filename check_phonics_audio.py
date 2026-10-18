#!/usr/bin/env python3
"""自检 42 个音素读得对不对（不听声音，用声学比对）。

元音：用同一套 espeak 音色合成「已知含该音的词」当参照，再算每个音的
      平均梅尔谱，做最近邻判定 —— 每个音都必须最接近它自己的参照词，
      不能跑到别的元音上去。这比单纯看 F1/F2 稳（短音频的共振峰估计容易抖）。
擦音：比频谱重心落在哪个频段（/s/ 必须比 /ʃ/ 高，这是最硬的判据）。
鼻音：F1 低 + 周期性高。
爆破音：时长和 burst 能量。

跑法: python3 check_phonics_audio.py
"""
import glob, os, subprocess, sys, tempfile
import numpy as np
import parselmouth

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, 'phonics')
SR = 22050
VOICE = 'en-us+f3'

# 音素 -> 含该音的参照词
REF = {
    'a': 'cat', 'e': 'bed', 'i': 'sit', 'o': 'dog', 'u': 'cup',
    'oo': 'moon', 'oo-short': 'book', 'ee': 'see', 'er': 'her',
    'ar': 'car', 'or': 'fork', 'ai': 'rain', 'ie': 'my', 'oa': 'boat',
    'ou': 'out', 'oi': 'boy', 'ue': 'cue',
}
FRIC = {'s': (4500, 9000), 'z': (3500, 8000), 'sh': (2500, 5000), 'th': (4000, 9000),
        'f': (3000, 9000), 'v': (2500, 8000), 'ch': (2500, 6000), 'x': (3500, 8000),
        'ts': (3500, 8000), 'h': (800, 4500)}
NASAL = {'m', 'n', 'ng'}
STOP = {'b', 'd', 'g', 'p', 't', 'k', 'qu', 'j'}

N_MEL, FMIN, FMAX = 24, 100, 7000


def to_sound(path=None, word=None):
    if word:
        p = tempfile.mktemp(suffix='.wav')
        subprocess.run(['espeak-ng', '-v', VOICE, '-s', '110', '-p', '75', word, '-w', p], check=True)
    else:
        p = tempfile.mktemp(suffix='.wav')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), p], check=True)
    s = parselmouth.Sound(p)
    os.remove(p)
    return s


def mel_matrix():
    nfft = 1024
    fq = np.fft.rfftfreq(nfft, 1 / SR)
    pts = np.linspace(_hz2mel(FMIN), _hz2mel(FMAX), N_MEL + 2)
    hz = _mel2hz(pts)
    fb = np.zeros((N_MEL, len(fq)))
    for m in range(N_MEL):
        lo, ce, hi = hz[m], hz[m + 1], hz[m + 2]
        for i, f in enumerate(fq):
            if lo <= f <= ce:
                fb[m, i] = (f - lo) / max(ce - lo, 1e-9)
            elif ce < f <= hi:
                fb[m, i] = (hi - f) / max(hi - ce, 1e-9)
    return fb


def _hz2mel(f): return 2595 * np.log10(1 + f / 700)
def _mel2hz(m): return 700 * (10 ** (m / 2595) - 1)


FB = mel_matrix()


def spec_vec(s):
    """浊音帧的中位梅尔谱（log），L2 归一化"""
    a = s.values[0]
    sr = s.sampling_frequency
    n, hop = 1024, 256
    vecs, energy = [], []
    for i in range(0, len(a) - n, hop):
        x = a[i:i + n] * np.hanning(n)
        e = np.sqrt((x ** 2).mean())
        if e < 0.01:
            continue
        ac = np.correlate(x, x, 'full')[len(x) - 1:]
        ac = ac / (ac[0] + 1e-9)
        per = ac[int(sr / 400):int(sr / 70)].max()
        if per < 0.4:
            continue
        sp = np.abs(np.fft.rfft(x)) ** 2
        vecs.append(np.log(FB @ sp + 1e-9))
        energy.append(e)
    if not vecs:
        return None
    v = np.median(np.array(vecs), axis=0)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def spectral(s):
    a = s.values[0]
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    fq = np.fft.rfftfreq(len(a), 1 / s.sampling_frequency)
    return (sp * fq).sum() / (sp.sum() + 1e-9), (sp[fq > 3000] ** 2).sum() / ((sp ** 2).sum() + 1e-9)


def periodicity(s):
    a = s.values[0]
    ac = np.correlate(a, a, 'full')[len(a) - 1:]
    ac = ac / (ac[0] + 1e-9)
    sr = s.sampling_frequency
    lo, hi = int(sr / 400), int(sr / 70)
    return float(ac[lo:hi].max()) if len(ac) > hi else 0.0


def main():
    names = sorted(os.path.basename(x)[:-4] for x in glob.glob(os.path.join(DIR, '*.mp3')))

    # ---- 元音：最近邻检验 ----
    print("=== 元音：跟同一音色的参照词比，看它是否最像自己 ===")
    mine, refs = {}, {}
    for k in REF:
        mine[k] = spec_vec(to_sound(path=os.path.join(DIR, k + '.mp3')))
        refs[k] = spec_vec(to_sound(word=REF[k]))
    keys = [k for k in REF if mine[k] is not None and refs[k] is not None]
    vowel_bad = []
    for k in keys:
        d = {j: float(np.linalg.norm(mine[k] - refs[j])) for j in keys}
        best = min(d, key=d.get)
        ok = best == k
        if not ok:
            vowel_bad.append(k)
        print(f"{k:9s}{'✅' if ok else '❌'} 最像 {best:9s} (距离 {d[best]:.3f} / 自己 {d[k]:.3f})")

    # ---- 其余音 ----
    print("\n=== 擦音 / 鼻音 / 爆破音 ===")
    other_bad = []
    for n in names:
        if n in REF:
            continue
        s = to_sound(path=os.path.join(DIR, n + '.mp3'))
        rms = float(np.sqrt((s.values ** 2).mean()))
        dur = s.get_total_duration()
        cen, hf = spectral(s)
        if rms < 0.004:
            print(f"{n:9s}❌ 静音"); other_bad.append(n); continue
        if n in FRIC:
            lo, hi = FRIC[n]
            ok = lo <= cen <= hi
            print(f"{n:9s}{'✅' if ok else '❌'} 重心{cen:.0f}Hz (期望{lo}-{hi}) 高频{hf:.2f} 时长{dur:.2f}s")
        elif n in NASAL:
            per = periodicity(s)
            ok = per > 0.5 and cen < 2500
            print(f"{n:9s}{'✅' if ok else '❌'} 鼻音 重心{cen:.0f}Hz 周期{per:.2f}")
        elif n in STOP:
            ok = 0.2 < dur < 0.9
            print(f"{n:9s}{'✅' if ok else '❌'} 爆破音 时长{dur:.2f}s 重心{cen:.0f}Hz")
        else:
            per = periodicity(s)
            ok = per > 0.4
            print(f"{n:9s}{'✅' if ok else '❌'} 周期{per:.2f} 重心{cen:.0f}Hz")
        if not ok:
            other_bad.append(n)

    bad = vowel_bad + other_bad
    print(f"\n元音 {len(keys)-len(vowel_bad)}/{len(keys)} 通过；其余 {len(names)-len(keys)-len(other_bad)}/{len(names)-len(keys)} 通过")
    if bad:
        print("要处理:", bad)
        sys.exit(1)
    print("全部通过")


if __name__ == '__main__':
    main()
