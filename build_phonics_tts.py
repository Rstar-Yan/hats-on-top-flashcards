#!/usr/bin/env python3
"""用 espeak-ng 合成 42 个音素（IPA 级精确），并逐个做声学校验。

为什么不用 Jolly 那包：实测是 42 首儿歌（全程有伴奏），里面没有孤立音素。
为什么不用 Ana：TTS 念 "b" 只会念字母名 bee，爆破音和短元音念不出来。

跑法: python3 build_phonics_tts.py
输出: phonics/*.mp3   + 屏幕打印每个音的校验结果
"""
import os, subprocess, wave, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'phonics')
TMP = '/tmp/espeak_phon'
SR = 22050
VOICE = 'en-us+f3'      # 女声变体，比默认男声适合儿童
SPEED = 110
PITCH = 75

# slug -> (espeak 音素串, 类别)
# 类别用于校验：fric=必须多高频；voiced=必须有周期性；stop=短促爆破
PH = {
    's':   ('s',    'fric'),
    'a':   ('a',    'voiced'),   # æ
    't':   ('t',    'stop'),
    'i':   ('I',    'voiced'),   # ɪ
    'p':   ('p',    'stop'),
    'n':   ('n',    'voiced'),
    'k':   ('k',    'stop'),
    'e':   ('E',    'voiced'),   # ɛ
    'h':   ('h',    'fricv'),    # 送气
    'r':   ('r@',   'voiced'),
    'm':   ('m',    'voiced'),
    'd':   ('d@',   'stop'),
    'g':   ('g@',   'stop'),
    'o':   ('0',    'voiced'),   # ɑː (on/octopus)
    'u':   ('V',    'voiced'),   # ʌ
    'l':   ('l',    'voiced'),
    'f':   ('f',    'fric'),
    'b':   ('b@',   'stop'),
    'ai':  ('eI',   'voiced'),
    'j':   ('dZ@',  'fricv'),
    'oa':  ('@U',   'voiced'),
    'ie':  ('aI',   'voiced'),
    'ee':  ('i:',   'voiced'),
    'or':  ('o:',   'voiced'),
    'z':   ('z',    'fricv'),
    'w':   ('w',    'voiced'),
    'ng':  ('N',    'voiced'),
    'v':   ('v',    'fricv'),
    'oo':  ('u:',   'voiced'),
    'oo-short': ('U', 'voiced'),
    'y':   ('j',    'voiced'),
    'x':   ('ks',   'fric'),
    'ch':  ('tS',   'fric'),
    'sh':  ('S',    'fric'),
    'th':  ('T',    'fric'),
    'th2': ('D',    'fricv'),
    'qu':  ('kw',   'stop'),
    'ou':  ('aU',   'voiced'),
    'oi':  ('oI',   'voiced'),
    'ue':  ('ju:',  'voiced'),
    'er':  ('3:',   'voiced'),
    'ar':  ('A:',   'voiced'),
    'ts':  ('ts',   'fric'),
}


def synth(phon):
    wav = os.path.join(TMP, phon.replace(':', '_') + '.wav')
    subprocess.run(['espeak-ng', '-v', VOICE, '-s', str(SPEED), '-p', str(PITCH),
                    f'[[{phon}]]', '-w', wav], check=True)
    w = wave.open(wav)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    w.close()
    os.remove(wav)
    if sr != SR:
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', wav, '-ar', str(SR), wav]) if False else None
    return a, sr


def trim(a, sr):
    """掐掉首尾静音，但留 30ms 前后垫"""
    st = int(0.01 * sr)
    fr = np.array([np.sqrt((a[i:i + st] ** 2).mean()) for i in range(0, len(a) - st, st)])
    if not len(fr) or fr.max() < 0.005:
        return a
    thr = max(0.01, 0.12 * fr.max())
    idx = np.where(fr > thr)[0]
    if not len(idx):
        return a
    s = max(0, (idx[0] - 3) * st)
    e = min(len(a), (idx[-1] + 4) * st)
    return a[s:e]


def metrics(a, sr):
    if not len(a):
        return 0, 0, 0, 0
    rms = float(np.sqrt((a ** 2).mean()))
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    fq = np.fft.rfftfreq(len(a), 1 / sr)
    cen = (sp * fq).sum() / sp.sum()
    hf = (sp[fq > 3000] ** 2).sum() / (sp ** 2).sum()
    ac = np.correlate(a, a, 'full')[len(a) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(sr / 400), int(sr / 70)
    per = float(ac[lo:hi].max()) if len(ac) > hi else 0
    return len(a) / sr, cen, hf, per, rms


def main():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    fails = []
    print(f"{'音素':10s} {'时长':>5} {'重心':>6} {'高频':>5} {'周期':>5}  校验")
    for slug, (phon, kind) in PH.items():
        a, sr = synth(phon)
        a = trim(a, sr)
        if len(a) < int(0.05 * sr):
            fails.append((slug, '合成不出声')); continue
        # 太短就重复几遍（爆破音需要多听几次）
        piece = a.copy()
        gap = np.zeros(int(0.09 * sr), dtype=np.float32)
        while len(a) < int(0.45 * sr):
            a = np.concatenate([a, gap, piece])
        cap = 0.35 if slug in ('b', 'd', 'g', 'j', 'r') else 0.9
        if len(a) > int(cap * sr):
            a = a[:int(cap * sr)]
        # 音量归一化：让 f/th 这种很轻的音和其他音一样响
        peak = float(np.percentile(np.abs(a), 99.5))
        if peak > 1e-4:
            a = a * min(0.7 / peak, 8.0)
        # 首尾淡入淡出
        n = int(0.01 * sr)
        a[:n] *= np.linspace(0, 1, n); a[-n:] *= np.linspace(1, 0, n)
        dur, cen, hf, per, rms = metrics(a, sr)
        ok, why = True, ''
        if rms < 0.004:
            ok, why = False, '合成出来是静音'
        elif kind == 'fric' and not (cen > 2500 and hf > 0.25):
            ok, why = False, '应该是嘶声但没有高频'
        elif kind == 'fricv' and not (per > 0.25 or hf > 0.2):
            ok, why = False, '既无周期也无高频'
        elif kind == 'voiced' and per < 0.35:
            ok, why = False, '浊音但没有周期'
        if not ok:
            fails.append((slug, why))
        print(f"{slug:10s} {dur:5.2f} {cen:6.0f} {hf:5.2f} {per:5.2f} {rms:5.2f}  {'OK' if ok else why}")
        # 写 wav -> mp3
        w = os.path.join(TMP, slug + '.wav')
        with wave.open(w, 'w') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
            f.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', w, '-ac', '1', '-ar', str(SR),
                        '-b:a', '64k', os.path.join(OUT, slug + '.mp3')], check=True)
        os.remove(w)
    print(f"\n生成 {len(PH) - len(fails)}/{len(PH)} 个")
    if fails:
        print("有问题:", fails)
        sys.exit(1)


if __name__ == '__main__':
    main()
