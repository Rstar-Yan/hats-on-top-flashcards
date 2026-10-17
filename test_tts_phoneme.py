#!/usr/bin/env python3
"""试验：让 Edge/Azure 的 Ana 童声直接读单个音素（SSML <phoneme>）。

edge-tts 默认把文本转义塞进 <prosody>，这里 monkeypatch mkssml 换成
带 <phoneme alphabet='ipa'> 的 SSML，看服务端认不认。
"""
import asyncio, os, sys
import edge_tts
from edge_tts import communicate as com

OUT = "/tmp/tts_phon_test"
os.makedirs(OUT, exist_ok=True)

VOICE = "en-US-AnaNeural"


def make_ssml(tc, inner):
    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>"
        f"<voice name='{tc.voice}'>"
        f"<prosody pitch='{tc.pitch}' rate='{tc.rate}' volume='{tc.volume}'>"
        f"{inner}"
        "</prosody></voice></speak>"
    )


# 劫持：communicate 里调用的是模块级的 mkssml
com.mkssml = make_ssml

TEST = {
    "ae":  ("æ", "<phoneme alphabet='ipa' ph='æ'>a</phoneme>"),
    "s":   ("s", "<phoneme alphabet='ipa' ph='s'>s</phoneme>"),
    "sh":  ("ʃ", "<phoneme alphabet='ipa' ph='ʃ'>sh</phoneme>"),
    "m":   ("m", "<phoneme alphabet='ipa' ph='m'>m</phoneme>"),
}


async def gen(name, inner):
    c = edge_tts.Communicate(inner, VOICE, rate="-10%")
    path = os.path.join(OUT, name + ".mp3")
    with open(path, "wb") as f:
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    return path


async def main():
    for name, (ipa, inner) in TEST.items():
        try:
            p = await gen(name, inner)
            print(f"{name:4s} {ipa:3s} -> {os.path.getsize(p)} bytes")
        except Exception as e:
            print(f"{name}: 失败 {type(e).__name__} {e}")


asyncio.run(main())
