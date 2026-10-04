### tools/mtp_rt.py, lines 1–111
```
1: """tools/mtp_rt.py - plan v0.3 P6: the MTP draft layer's runtime files, from the packed MTP GGUF.
2: 
3:     python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q2_0.gguf --out <Strata>/mtp-bf16/rt
4: 
5: Writes
6:   experts.bin   512 routed experts in the engine's blob layout (`include/strata/kernels/cpu/expert.hpp`): gate/up
7:                 rows interleaved (2r = gate r, 2r+1 = up r), then down rows; the Q2_0 codes in one plane and the fp16
8:                 scales in another.  A lossless relayout of the GGUF's Q2_0 blocks (same bytes, `cpu_expert_fixture.py`).
9:   dense.bin     every other tensor: the large projections quantized to Q8_0 (ggml's reference rounding) so the
10:                 engine's multi-column MMVQ can run them; the hyper-connection and router weights kept BF16; the
11:                 RMSNorm weights as F32 with the Gemma "+1" applied (vLLM GemmaRMSNorm scales by 1 + w).
12:   dense.txt     one line per tensor: name kind rows cols offset bytes   (kind = q8_0 | bf16 | f32)
13: """
14: from __future__ import annotations
15: 
16: import argparse
17: import sys
18: from pathlib import Path
19: 
20: import numpy as np
21: 
22: sys.path.insert(0, str(Path(__file__).resolve().parent))
23: from _paths import add_gguf_py  # noqa: E402
24: add_gguf_py()
25: import gguf  # noqa: E402
26: 
27: H, FF, NE = 2560, 640, 512
28: BLOB = 3 * (H * FF * 18 // 64)
29: Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
30:       "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
31:       "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}
32: 
33: 
34: def q8_0(x: np.ndarray) -> bytes:
35:     """ggml quantize_row_q8_0_ref over rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
36:     b = x.reshape(-1, 32).astype(np.float32)
37:     amax = np.abs(b).max(axis=1)
38:     d = amax / 127.0
39:     inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
40:     v = b * inv[:, None]
41:     q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
42:     out = np.empty((b.shape[0], 34), dtype=np.uint8)
43:     out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
44:     out[:, 2:] = q.view(np.uint8)
45:     return out.tobytes()
46: 
47: 
48: def blob_of(gu: np.ndarray, dn: np.ndarray) -> bytes:
49:     """gu: (1280, 720) Q2_0 rows (gate rows 0..639, up rows 640..1279); dn: (2560, 180)."""
50:     gub = gu.reshape(2 * FF, H // 64, 18)
51:     inter = np.empty_like(gub)
52:     inter[0::2] = gub[:FF]
53:     inter[1::2] = gub[FF:]
54:     dnb = dn.reshape(H, FF // 64, 18)
55:     parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
56:     out = b"".join(parts)
57:     assert len(out) == BLOB, len(out)
58:     return out
59: 
60: 
61: def main() -> int:
62:     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
63:     ap.add_argument("--gguf", required=True)
64:     ap.add_argument("--out", required=True)
65:     a = ap.parse_args()
66:     out = Path(a.out)
67:     out.mkdir(parents=True, exist_ok=True)
68:     r = gguf.GGUFReader(a.gguf)
69:     tens = {t.name: t for t in r.tensors}
70:     gu = np.asarray(tens["mtp.layers.0.mlp.experts.gate_up_proj"].data)
71:     dn = np.asarray(tens["mtp.layers.0.mlp.experts.down_proj"].data)
72:     assert gu.shape == (NE, 2 * FF, 720) and dn.shape == (NE, H, 180), (gu.shape, dn.shape)
73:     with open(out / "experts.bin", "wb") as f:
74:         for e in range(NE):
75:             f.write(blob_of(gu[e], dn[e]))
76:     lines = []
77:     off = 0
78:     with open(out / "dense.bin", "wb") as f:
79:         for name, t in tens.items():
80:             if "experts." in name:
81:                 continue
82:             short = name[len("mtp."):]
83:             short = short[len("layers.0."):] if short.startswith("layers.0.") else short
84:             data = np.asarray(t.data)
85:             if int(t.tensor_type) == 0:        # F32 norm weights: raw GemmaRMSNorm w -> 1 + w
86:                 arr = (data.astype(np.float32) + 1.0)
87:                 raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
88:             else:                              # BF16
89:                 u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
90:                 u16 = u16.reshape(data.shape[0], -1)
91:                 rows, cols = u16.shape
92:                 if short in Q8:
93:                     f32 = (u16.astype(np.uint32) << 16).view(np.float32)
94:                     raw, kind = q8_0(f32), "q8_0"
95:                 else:
96:                     raw, kind = u16.tobytes(), "bf16"
97:             pad = (-off) % 256
98:             f.write(b"\0" * pad)
99:             off += pad
100:             f.write(raw)
101:             lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
102:             off += len(raw)
103:     (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
104:     print(f"experts.bin {NE * BLOB} B, dense.bin {off} B, {len(lines)} tensors -> {out}")
105:     for l in lines:
106:         print("  " + l)
107:     return 0
108: 
109: 
110: if __name__ == "__main__":
111:     sys.exit(main())
```

### tools/mtp_pack.py, lines 1–193
```
1: """tools/mtp_pack.py - plan v0.3 P6 prep: the MTP draft block as a Strata GGUF, from the BF16 tensors.
2: 
3: llama.cpp's qwen4exp converter drops the MTP head (`supports_mtp_export = False`), and the flyweight MTP GGUF is
4: no longer on this PC, so Strata packs its own from the 31 `mtp.*` tensors fetched by tools/mtp_fetch.py:
5: 
6:     python tools/mtp_pack.py --src Desktop/Strata/mtp-bf16 --experts q2_0 --out mtp-q2_0.gguf
7:     python tools/mtp_pack.py --src ... --experts q4_0 --out mtp-q4_0.gguf       (acceptance comparison arm)
8: 
9: Layout: dense tensors (attention, indexer, hyper-connections, shared expert, router, fc/norms) stay BF16 (F32 for
10: 1-D norms), ~0.18 GB. The routed experts keep the checkpoint's fused layout - `gate_up_proj` [512, 1280, 2560] and
11: `down_proj` [512, 2560, 640], quantized along the last (input) axis - in one of:
12: 
13:     q2_0   64-element blocks, grid {-1, 0, 1, 2} x d. The scale is chosen per block to MINIMIZE squared error over
14:            that grid (the ggml reference sets d = max|w| and never uses the +2 level). Same format as the main
15:            model's experts, so Strata's CPU VNNI kernel and GPU hit kernel serve it unchanged. ~0.71 GB.
16:     q4_0   ggml reference rounding. ~1.42 GB.       q8_0   ggml reference. ~2.67 GB.
17: 
18: This is round-to-nearest, not GSQ: the plan picks the expert format by MEASURED draft acceptance (P0.3/P6), not
19: by this file's reconstruction error, which is reported per tensor only as a sanity check. No model runs here.
20: """
21: from __future__ import annotations
22: 
23: import argparse
24: import hashlib
25: import json
26: import sys
27: import time
28: from pathlib import Path
29: 
30: import numpy as np
31: 
32: HERE = Path(__file__).resolve().parent
33: sys.path.insert(0, str(Path(__file__).resolve().parent))
34: from _paths import add_gguf_py  # noqa: E402
35: add_gguf_py()
36: import gguf  # noqa: E402  (the pinned llama.cpp gguf-py, which knows Q2_0 = type 42)
37: 
38: EXPERT_TENSORS = ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj")
39: 
40: 
41: class _Experts:
42:     """Index an expert out of a BF16 memmap as float32 on demand."""
43:     def __init__(self, raw):
44:         self.raw = raw
45: 
46:     def __getitem__(self, e):
47:         return (np.asarray(self.raw[e]).astype(np.uint32) << 16).view(np.float32)
48: 
49: 
50: def load_bf16(path: Path, shape: list[int]) -> np.ndarray:
51:     raw = np.fromfile(path, dtype="<u2")
52:     if raw.size != int(np.prod(shape)):
53:         raise ValueError(f"{path}: {raw.size} values, expected shape {shape}")
54:     return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
55: 
56: 
57: # ------------------------------------------------------------------------------------------------ quantizers
58: def q2_0(w: np.ndarray) -> np.ndarray:
59:     """[..., n] float32 -> bytes of Q2_0 blocks (fp16 d, 16 bytes of 2-bit codes, code = q + 1, 4 per byte)."""
60:     x = w.reshape(-1, 64).astype(np.float32)
61:     amax = np.abs(x).max(axis=1, keepdims=True)
62:     best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
63:     best_d = np.zeros_like(amax)
64:     # Candidate scales span amax/2 (the +2 level reaches the max) to amax (the ggml reference); 17 steps.
65:     for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
66:         d = amax * f
67:         inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
68:         q = np.clip(np.rint(x * inv), -1, 2)
69:         err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
70:         better = err < best_err
71:         best_err = np.where(better, err, best_err)
72:         best_d = np.where(better, d, best_d)
73:     d16 = best_d.astype(np.float16)
74:     d = d16.astype(np.float32)                                   # quantize against the STORED scale
75:     inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
76:     codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)          # 0..3
77:     c = codes.reshape(-1, 16, 4)
78:     packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
79:     out = np.empty((x.shape[0], 18), dtype=np.uint8)
80:     out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
81:     out[:, 2:] = packed
82:     return out.reshape(-1)
83: 
84: 
85: def q4_0(w: np.ndarray) -> np.ndarray:
86:     """ggml quantize_row_q4_0_ref: d = max / -8 (signed max), code = clamp(round(x/d + 8.5) truncated, 0, 15)."""
87:     x = w.reshape(-1, 32).astype(np.float32)
88:     idx = np.abs(x).argmax(axis=1)
89:     mx = x[np.arange(x.shape[0]), idx][:, None]
90:     d = mx / -8.0
91:     inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
92:     q = np.minimum(15, np.trunc(x * inv + 8.5)).astype(np.uint8)
93:     packed = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
94:     out = np.empty((x.shape[0], 18), dtype=np.uint8)
95:     out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
96:     out[:, 2:] = packed
97:     return out.reshape(-1)
98: 
99: 
100: def q8_0(w: np.ndarray) -> np.ndarray:
101:     x = w.reshape(-1, 32).astype(np.float32)
102:     d = np.abs(x).max(axis=1, keepdims=True) / 127.0
103:     inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
104:     q = np.rint(x * inv).astype(np.int8)
105:     out = np.empty((x.shape[0], 34), dtype=np.uint8)
106:     out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
107:     out[:, 2:] = q.view(np.uint8)
108:     return out.reshape(-1)
109: 
110: 
111: def dequant(kind: str, blob: np.ndarray, n: int) -> np.ndarray:
112:     if kind == "q2_0":
113:         b = blob.reshape(-1, 18)
114:         d = b[:, :2].copy().view(np.float16).astype(np.float32)
115:         qs = b[:, 2:]
116:         codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32)
117:         return ((codes - 1.0) * d).reshape(-1)[:n]
118:     if kind == "q4_0":
119:         b = blob.reshape(-1, 18)
120:         d = b[:, :2].copy().view(np.float16).astype(np.float32)
121:         qs = b[:, 2:]
122:         q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)
123:         return ((q - 8.0) * d).reshape(-1)[:n]
124:     b = blob.reshape(-1, 34)
125:     d = b[:, :2].copy().view(np.float16).astype(np.float32)
126:     return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)[:n]
127: 
128: 
129: QUANT = {"q2_0": (q2_0, gguf.GGMLQuantizationType.Q2_0, 64, 18),
130:          "q4_0": (q4_0, gguf.GGMLQuantizationType.Q4_0, 32, 18),
131:          "q8_0": (q8_0, gguf.GGMLQuantizationType.Q8_0, 32, 34)}
132: 
133: 
134: def main() -> int:
135:     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
136:     ap.add_argument("--src", required=True, help="directory written by tools/mtp_fetch.py fetch")
137:     ap.add_argument("--experts", choices=sorted(QUANT), default="q2_0")
138:     ap.add_argument("--out", required=True)
139:     ap.add_argument("--check-experts", type=int, default=8, help="experts per tensor used for the error report")
140:     a = ap.parse_args()
141:     src = Path(a.src)
142:     manifest = json.loads((src / "mtp-manifest.json").read_text())
143:     fn, qtype, block, block_bytes = QUANT[a.experts]
144:     w = gguf.GGUFWriter(a.out, "qwen4exp-mtp")
145:     w.add_string("strata.mtp.source", "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors")
146:     w.add_string("strata.mtp.source_sha256", hashlib.sha256(
147:         json.dumps({t["name"]: t["sha256"] for t in manifest}, sort_keys=True).encode()).hexdigest())
148:     w.add_string("strata.mtp.expert_format", a.experts)
149:     w.add_string("strata.mtp.expert_quantizer", "per-block MSE scale search" if a.experts == "q2_0" else "ggml reference")
150:     report = []
151:     for t in sorted(manifest, key=lambda t: t["name"]):
152:         name, shape = t["name"], t["shape"]
153:         t0 = time.time()
154:         if name in EXPERT_TENSORS:
155:             # Streamed one expert at a time from a read-only view: the fused gate_up tensor alone is 6.7 GB as
156:             # float32, so it is never materialized whole.
157:             raw = np.memmap(src / t["file"], dtype="<u2", mode="r", shape=tuple(shape))
158:             x = _Experts(raw)
159:             if shape[-1] % block:
160:                 raise ValueError(f"{name}: inner dim {shape[-1]} not a multiple of {block}")
161:             parts = [fn(x[e]) for e in range(shape[0])]            # one expert at a time bounds memory
162:             blob = np.concatenate(parts)
163:             errs = []
164:             for e in range(min(a.check_experts, shape[0])):
165:                 ref = x[e].reshape(-1)
166:                 got = dequant(a.experts, parts[e], ref.size)
167:                 errs.append(float(np.sqrt(((got - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12)))
168:             # gguf-py takes the quantized bytes with the ELEMENT shape; ggml order is innermost-first.
169:             w.add_tensor(name, blob, raw_shape=list(shape[:-1]) + [shape[-1] // block * block_bytes], raw_dtype=qtype)
170:             report.append({"tensor": name, "format": a.experts, "bytes": int(blob.size),
171:                            "relative_rms_error_first_experts": round(float(np.mean(errs)), 5),
172:                            "seconds": round(time.time() - t0, 1)})
173:         elif len(shape) == 1:
174:             x = load_bf16(src / t["file"], shape)
175:             w.add_tensor(name, x.astype(np.float32))
176:             report.append({"tensor": name, "format": "F32", "bytes": x.size * 4})
177:         else:
178:             bf = np.fromfile(src / t["file"], dtype="<u2").reshape(shape)
179:             w.add_tensor(name, bf, raw_dtype=gguf.GGMLQuantizationType.BF16)
180:             report.append({"tensor": name, "format": "BF16", "bytes": bf.size * 2})
181:         print(f"{name}: {report[-1]['format']} {report[-1]['bytes'] / 1e6:.1f} MB", flush=True)
182:     w.write_header_to_file()
183:     w.write_kv_data_to_file()
184:     w.write_tensors_to_file()
185:     w.close()
186:     total = sum(r["bytes"] for r in report)
187:     Path(a.out + ".report.json").write_text(json.dumps({"total_bytes": total, "tensors": report}, indent=1))
188:     print(f"wrote {a.out}: {total / 1e9:.3f} GB of tensor data")
189:     return 0
190: 
191: 
192: if __name__ == "__main__":
193:     sys.exit(main())
```

### tools/gguf_reader.py, lines 1–142
```
1: """Minimal GGUF v3 header reader - no ggml, no gguf-py.
2: 
3: Written because the `gguf` PyPI package cannot open this artifact: its `GGMLQuantizationType`
4: enum has no member for type 42 (`Q2_0`), which is newer than the library. That is itself a P0.S6
5: finding - the format this engine is specialized for is ahead of the standard tooling.
6: 
7: This is the reference implementation for `src/artifact/` (architecture §15: "GGUF v3 reader (mmap,
8: no ggml dependency)"). It parses the header only; tensor data is never read.
9: """
10: from __future__ import annotations
11: 
12: import dataclasses
13: import pathlib
14: import struct
15: from typing import Any
16: 
17: GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
18: 
19: # ggml_type values. 42 = Q2_0, the GSQ-RCO routed-expert encoding, absent from gguf-py as of 0.19.
20: GGML_TYPES: dict[int, str] = {
21:     0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3", 6: "Q5_0", 7: "Q5_1",
22:     8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
23:     15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
24:     21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
25:     28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
26:     34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4", 37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8",
27:     39: "MXFP4", 40: "NVFP4", 41: "Q4_0_8_8", 42: "Q2_0",
28: }
29: 
30: # GGUF metadata value type ids
31: GGUF_META = {
32:     0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2), 4: ("u32", 4), 5: ("i32", 4),
33:     6: ("f32", 4), 7: ("bool", 1), 8: ("string", None), 9: ("array", None),
34:     10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8),
35: }
36: 
37: # ggml block geometry for the encodings this model uses: (block elements, bytes per block)
38: # This is what makes a per-tensor byte count checkable, and it is the contract the kernels share.
39: BLOCK_GEOMETRY: dict[str, tuple[int, int]] = {
40:     "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8),
41:     "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
42:     "Q8_0": (32, 34), "Q8_1": (32, 36),
43:     "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176),
44:     "Q6_K": (256, 210), "Q8_K": (256, 292),
45:     "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98), "IQ1_S": (256, 50),
46:     "IQ4_NL": (32, 18), "IQ3_S": (256, 110), "IQ2_S": (256, 82), "IQ4_XS": (256, 136),
47:     "IQ1_M": (256, 56), "Q2_0": (64, 18), "MXFP4": (32, 17), "NVFP4": (64, 36),
48: }
49: 
50: 
51: @dataclasses.dataclass
52: class TensorInfo:
53:     name: str
54:     shape: list[int]
55:     type_id: int
56:     type_name: str
57:     offset: int
58: 
59:     @property
60:     def elements(self) -> int:
61:         n = 1
62:         for d in self.shape:
63:             n *= d
64:         return n
65: 
66:     def expected_bytes(self) -> int | None:
67:         geom = BLOCK_GEOMETRY.get(self.type_name)
68:         if geom is None:
69:             return None
70:         block_elems, block_bytes = geom
71:         if self.elements % block_elems:
72:             return None
73:         return self.elements // block_elems * block_bytes
74: 
75: 
76: class GGUFFile:
77:     def __init__(self, path: pathlib.Path):
78:         self.path = pathlib.Path(path)
79:         self.metadata: dict[str, Any] = {}
80:         self.tensors: list[TensorInfo] = []
81:         self.version = 0
82:         self.alignment = 32
83:         with self.path.open("rb") as fh:
84:             self._parse(fh)
85:         self.data_start = self._data_start
86: 
87:     # ------------------------------------------------------------------ internals
88:     def _parse(self, fh) -> None:
89:         magic = struct.unpack("<I", fh.read(4))[0]
90:         if magic != GGUF_MAGIC:
91:             raise ValueError(f"{self.path.name}: not a GGUF file (magic {magic:#x})")
92:         self.version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
93:         if self.version != 3:
94:             raise ValueError(f"{self.path.name}: GGUF v{self.version}, this reader handles v3")
95:         for _ in range(n_kv):
96:             key = self._str(fh)
97:             self.metadata[key] = self._value(fh)
98:         for _ in range(n_tensors):
99:             name = self._str(fh)
100:             (n_dims,) = struct.unpack("<I", fh.read(4))
101:             shape = list(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
102:             type_id, offset = struct.unpack("<IQ", fh.read(12))
103:             self.tensors.append(TensorInfo(name, shape, type_id,
104:                                            GGML_TYPES.get(type_id, f"type{type_id}"), offset))
105:         align = self.metadata.get("general.alignment")
106:         if isinstance(align, int) and align:
107:             self.alignment = align
108:         pos = fh.tell()
109:         self._data_start = (pos + self.alignment - 1) // self.alignment * self.alignment
110: 
111:     def _str(self, fh) -> str:
112:         (n,) = struct.unpack("<Q", fh.read(8))
113:         return fh.read(n).decode("utf-8", "replace")
114: 
115:     def _value(self, fh):
116:         (t,) = struct.unpack("<I", fh.read(4))
117:         name, size = GGUF_META[t]
118:         if name == "string":
119:             return self._str(fh)
120:         if name == "array":
121:             (et,) = struct.unpack("<I", fh.read(4))
122:             (count,) = struct.unpack("<Q", fh.read(8))
123:             ename, esize = GGUF_META[et]
124:             if ename == "string":
125:                 return [self._str(fh) for _ in range(count)]
126:             fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
127:                    "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}[ename]
128:             raw = fh.read(esize * count)
129:             return list(struct.unpack(f"<{count}{fmt}", raw))
130:         fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
131:                "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}[name]
132:         return struct.unpack(f"<{fmt}", fh.read(size))[0]
133: 
134:     # ------------------------------------------------------------------ helpers
135:     def by_type(self) -> dict[str, int]:
136:         out: dict[str, int] = {}
137:         for t in self.tensors:
138:             out[t.type_name] = out.get(t.type_name, 0) + 1
139:         return dict(sorted(out.items(), key=lambda kv: -kv[1]))
140: 
141:     def find(self, needle: str) -> list[TensorInfo]:
142:         return [t for t in self.tensors if needle in t.name]
```

### tools/mtp_fetch.py, lines 1–35
```
1: """tools/mtp_fetch.py - plan v0.3, P0.3/P6: the MTP block from the BF16 checkpoint, without the checkpoint.
2: 
3: The GSQ-RCO GGUF ships no MTP head. The BF16 checkpoint (Qwen/Qwen3.8-Flash-Next, 360 GB in 131 shards) does:
4: 31 `mtp.*` tensors scattered over 28 shards. Safetensors puts a JSON header (name -> dtype, shape, byte range)
5: at the start of each shard, so HTTP range requests can read the headers and then only the MTP tensors.
6: 
7:     python tools/mtp_fetch.py inventory --out DIR          # headers only (a few KB per shard)
8:     python tools/mtp_fetch.py fetch --out DIR [--only SUBSTR]  # the MTP tensors themselves, resumable
9:     python tools/mtp_fetch.py verify --out DIR             # the fetched tensors against SHA256 (exit 3: bad ones)
10: 
11: `fetch` writes one raw file per tensor plus `mtp-manifest.json` (dtype, shape, source shard, byte range,
12: sha256). It never downloads anything but the ranges named in the headers. Nothing here runs a model.
13: 
14: #327: a mirror or proxy that ignores the Range header answers 200 with the whole shard - its JSON header and
15: unrelated tensors - and a proxy may cut that to the requested length, so neither the size nor a hash of what was
16: downloaded catches it (the drafter then accepts nothing, silently).  A range read therefore needs a 206 whose
17: Content-Range is the range asked for, and every tensor of the pinned revision is checked against SHA256 below.
18: """
19: import argparse
20: import hashlib
21: import json
22: import os
23: import struct
24: import sys
25: import time
26: import urllib.error
27: import urllib.request
28: 
29: # #214: a fixed commit of the checkpoint (its `sha` from https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next
30: # on 2026-09-30), so every install reads the same tensors; STRATA_MTP_REVISION overrides it (e.g. main).  When the
31: # repository no longer has it, the current files are read instead, with a message (resolve_repo).
32: PINNED_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
33: REVISION = os.environ.get("STRATA_MTP_REVISION") or PINNED_REVISION
34: REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % REVISION
35: PINNED = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % PINNED_REVISION   # SHA256's revision
```

### tools/mtp_fetch.py, lines 205–256
```
205: def fetch(out, only):
206:     inv_path = os.path.join(out, "mtp-inventory.json")
207:     tdir = os.path.join(out, "tensors")
208:     inv = None
209:     if os.path.exists(inv_path):
210:         try:
211:             with open(inv_path, encoding="utf-8") as f:
212:                 inv = json.load(f)
213:         except ValueError:
214:             inv = None
215:     # The byte ranges belong to one revision: an inventory read from another repository or revision is read again,
216:     # and a tensor fetched from there is not resumed - it is kept only when it is the pinned revision's (sha256).
217:     same = isinstance(inv, dict) and inv.get("repo") == REPO
218:     rows = inv["tensors"] if same else inventory(out)
219:     os.makedirs(tdir, exist_ok=True)
220:     manifest = []
221:     chunk = 64 << 20
222:     for r in rows:
223:         if only and only not in r["name"]:
224:             continue
225:         path = os.path.join(tdir, r["name"] + ".bin")
226:         want = SHA256.get(r["name"]) if pinned() else None
227:         if os.path.exists(path) and not same and (
228:                 want is None or os.path.getsize(path) != r["bytes"] or sha256_of(path) != want):
229:             os.remove(path)
230:         for attempt in range(2):
231:             have = os.path.getsize(path) if os.path.exists(path) else 0
232:             if have > r["bytes"]:
233:                 os.remove(path)
234:                 have = 0
235:             with open(path, "ab") as f:
236:                 pos = r["start"] + have
237:                 while pos <= r["end"]:
238:                     end = min(pos + chunk - 1, r["end"])
239:                     f.write(get(REPO + r["shard"], pos, end))
240:                     pos = end + 1
241:                     print("%s %.0f%%" % (r["name"], 100 * (pos - r["start"]) / r["bytes"]), file=sys.stderr)
242:             if os.path.getsize(path) != r["bytes"]:
243:                 sys.exit("%s: size %d != %d" % (path, os.path.getsize(path), r["bytes"]))
244:             digest = sha256_of(path)
245:             if want is None or digest == want:
246:                 break
247:             os.remove(path)                         # wrong bytes (#327): the whole tensor again, once
248:             if attempt:
249:                 sys.exit("%s: sha256 %s is not the checkpoint's %s - the download source sends wrong data (a mirror "
250:                          "that ignores Range requests?)" % (r["name"], digest, want))
251:             print("%s: wrong bytes (sha256 %s...), fetching it again" % (r["name"], digest[:12]), file=sys.stderr)
252:         manifest.append(dict(r, file=os.path.relpath(path, out), sha256=digest))
253:     with open(os.path.join(out, "mtp-manifest.json"), "w", encoding="utf-8") as f:
254:         json.dump(manifest, f, indent=1)
255: 
256: 
```

### setup.py, lines 2408–2456
```
2408: OLD_DRAFT_VOCABS = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}   # to 0.1.26: 27 Han tokens
2409: 
2410: 
2411: DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin"}
2412: 
2413: 
2414: def saved_draft_vocab(cfg_path: Path) -> str | None:
2415:     """The draft subset a model's config chose earlier (--draft-vocab), or None: a setup run again without the flag
2416:     rewrites the config, and would otherwise put the default subset back."""
2417:     try:
2418:         v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
2419:     except (OSError, ValueError, AttributeError):
2420:         return None
2421:     return v if v in DRAFT_VOCABS else None
2422: 
2423: 
2424: def mtp_corrupt(mtp: Path, env=None) -> bool:
2425:     """#327: True when the MTP tensors an install fetched are not the pinned checkpoint's (tools/mtp_fetch.py verify,
2426:     which hashes only files that changed since they last checked out).  A mirror that ignored range requests left the
2427:     shards' starts there instead, and the draft layer built from them accepted nothing - with no error anywhere."""
2428:     if not (mtp / "tensors").is_dir():
2429:         return False
2430:     r = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(mtp)], env=env,
2431:                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
2432:     return r.returncode == 3
2433: 
2434: 
2435: def refresh_draft_vocab(rt: Path, choice: str = "cjk") -> None:
2436:     """The draft layer's token subset in the MTP folder: `cjk` (data/draft_vocab.bin, since 0.1.27, #137), `en`
2437:     (data/draft_vocab_en.bin, the English/code subset before it: ~110 MiB less VRAM, English answers 1-2% faster) or
2438:     `cyrillic` (data/draft_vocab_cyrillic.bin: English/code and the whole Cyrillic script, for Ukrainian, Russian,
2439:     Bulgarian, Serbian... answers).
2440:     Copied when missing or when a shipped subset other than the chosen one is there; a subset made by hand is kept."""
2441:     new, dst = ROOT / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
2442:     if not new.exists() or not rt.is_dir():
2443:         return
2444:     if dst.exists():
2445:         old = hashlib.sha256(dst.read_bytes()).hexdigest()
2446:         shipped = OLD_DRAFT_VOCABS | {hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
2447:                                       for f in DRAFT_VOCABS.values() if (ROOT / "data" / f).exists()}
2448:         if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
2449:             return
2450:         ok("draft layer: the token subset " + {"cjk": "with Chinese, Japanese and Korean",
2451:                                                "cyrillic": "with the Cyrillic script"}.get(choice,
2452:                                                                                           "for English and code (less VRAM)"))
2453:     shutil.copyfile(new, dst)
2454: 
2455: 
2456: def ensure_engine_for(cards, cfg_path: Path, cfg: dict, yes: bool) -> dict:
```

### serve/server.py, lines 614–633
```
614: def gpu_list(cfg: dict) -> list[int]:
615:     """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
616:     numbers them; [] when it names none."""
617:     g = cfg.get("gpu")
618:     if g is None or g == "":
619:         return []
620:     items = g if isinstance(g, (list, tuple)) else str(g).split(",")
621:     return [int(str(x).strip()) for x in items if str(x).strip() != ""]
622: 
623: 
624: def engine_args(cfg: dict) -> list[str]:
625:     """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
626:     config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32")."""
627:     args = list(cfg["args"])
628:     if len(gpu_list(cfg)) > 1 and "--layer-split" not in args:
629:         args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
630:     # opt-in: an auto split runs on the first card alone when it holds every profiled expert and the KV
631:     if len(gpu_list(cfg)) > 1 and cfg.get("split_skip_if_fits") and "--split-skip-if-fits" not in args:
632:         args.append("--split-skip-if-fits")
633:     return args
```

### serve/server.py, lines 649–668
```
649:     return gpu_list(cfg)
650: 
651: 
652: def child_env(cfg: dict) -> dict:
653:     """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
654:     compiled it) first on the library search path."""
655:     env = dict(os.environ)
656:     if hip_visible(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (hip_visible)
657:         env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in hip_visible(cfg))
658:     elif gpu_list(cfg):                              # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
659:         env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
660:         env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
661:     for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries (AMD: the GEMM tuning table)
662:         env[str(k)] = str(v)
663:     dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
664:     if dirs:
665:         var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
666:         env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
667:     return env
668: 
```

### include/strata/kernels/cpu/expert.hpp, lines 35–66
```
35: inline constexpr int FF = 640;      // expert intermediate width
36: inline constexpr int NE = 512;      // routed experts per layer
37: inline constexpr int QK = 64;       // QK2_0: weights per fp16 scale
38: inline constexpr int QKA = 32;      // QK8_1/QK8_0: ACTIVATION elements per scale
39: inline constexpr int BB = 18;       // bytes per 64-weight Q2_0 block: 2 fp16-scale + 16 codes
40: inline constexpr int MAXC = H / QKA;             // 80 activation chunks across the widest reduction
41: inline constexpr int ROW_GU = H * 2 / 8;         // 640 B of codes per gate/up row
42: inline constexpr int ROW_D = FF * 2 / 8;         // 160 B of codes per down row
43: inline constexpr int SC_GU = H / QK;             // 40 weight blocks per gate/up row
44: inline constexpr int SC_D = FF / QK;             // 10 weight blocks per down row
45: inline constexpr size_t BLOB = 3ull * (H * FF * BB / QK);   // 1,382,400
46: 
47: // ---- the blob's internal layout (one expert) ----
48: inline constexpr size_t O_GU_CODES = 0;
49: inline constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
50: inline constexpr size_t O_GU_SCALES = O_D_CODES + 1ull * H * ROW_D;
51: inline constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;
52: 
53: /// One activation in Strata's planar storage: `QKA` elements per chunk.
54: /// The default legacy quantizer uses FP32 scales and half-away rounding. It is
55: /// not the pinned ggml CPU Q8_0 contract; expert_set_oracle_q8_0 selects that
56: /// experimental contract (FP16-rounded scales and x86 nearest-even codes).
57: ///
58: /// The historical function name `act_quant_q8_1` does not describe a native
59: /// ggml block layout. One 64-weight block always spans two 32-element chunks.
60: ///
61: /// `hx[k] = scale[k] * sum[k]` is the weight-independent correction the Q2_0 identity needs
62: /// (`sum (c-1) d_w xhat d_x = d_w (d_x*sum(c*xhat) - d_x*sum(xhat))`), precomputed once per layer rather than
63: /// once per row: 640 rows would otherwise recompute it 640 times.
64: struct ActQ {
65:     alignas(64) int8_t q[H];
66:     float scale[MAXC];
```

### include/strata/kernels/iq_kernels.hpp, lines 1–69
```
1: // include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
2: // IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
3: // UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
4: //
5: // The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
6: // ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
7: // means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
8: #pragma once
9: 
10: #include <cstddef>
11: #include <cstdint>
12: 
13: namespace strata::kernels {
14: 
15: /// ggml type ids handled here.
16: bool iq_supported(int ggml_type) noexcept;
17: /// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
18: bool embed_type_supported(int ggml_type) noexcept;
19: /// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
20: size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;
21: 
22: /// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
23: void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);
24: 
25: /// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
26: void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
27: 
28: /// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
29: void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
30: void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
31: /// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
32: /// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
33: void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
34:                    int64_t n_embd, float* out, void* stream);
35: /// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
36: /// uses: row 2r = gate row r, row 2r+1 = up row r.
37: void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
38:                        void* stream);
39: 
40: /// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
41: struct NativeExpertLayout {
42:     int gu_type = -1, d_type = -1;
43:     int64_t n_embd = 0, n_ff = 0;
44:     size_t gu_row = 0, d_row = 0;       // bytes per row
45:     size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
46:     size_t bytes = 0;                   // the whole blob
47: };
48: NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
49: /// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
50: /// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
51: bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;
52: 
53: /// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
54: size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);
55: 
56: /// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
57: /// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
58: /// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
59: void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
60:                            const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
61:                            int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream);
62: 
63: /// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
64: /// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
65: /// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
66: void iq_set_old_kernels(bool old);
67: bool iq_old_kernels();
68: 
69: }  // namespace strata::kernels
```

### include/strata/core/mtp.hpp, lines 135–172
```
135:     cudaStream_t cs_ = nullptr;
136:     cudaGraphExec_t prefill_exec_[9] = {};
137:     cudaGraphExec_t prefill_dev_exec_[9] = {};
138:     int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
139:     int64_t pf_cap_ = 0;          ///< its capacity in ints
140:     cudaGraphExec_t round_exec_[9] = {};
141: 
142:     struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
143:     std::vector<Tensor> tensors_;
144:     uint8_t* dense_ = nullptr;
145:     uint8_t* experts_ = nullptr;
146:     void* state_arena_ = nullptr;
147:     QsaState st_;
148:     void* arena_ = nullptr;
149: 
150:     // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
151:     // the drafts out
152:     int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
153:     int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
154:     int32_t *h_out_ = nullptr, *m_out_ = nullptr;
155:     float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
156:     // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
157:     uint8_t* dhead_ = nullptr;
158:     int32_t* dvocab_ = nullptr;
159:     int64_t n_dvocab_ = 0;
160:     std::string rt_dir_;
161:     int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
162:     int64_t prompt_len_ = 0;
163:     float* probs_ = nullptr;
164:     // device
165:     int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
166:     float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
167:     float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
168:     float* xn_ = nullptr;
169:     uint8_t* xq_ = nullptr;
170:     float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
171:     float* attn_scratch_ = nullptr;
172:     float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
```

### src/core/mtp.cpp, lines 143–222
```
143:                       int64_t window) {
144:     cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
145:     g_ = &g;
146:     ss_ = &ss;
147:     max_t_ = max_t;
148:     rt_dir_ = rt_dir;
149:     if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
150:     // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole “drafter files” cost; reporting
151:     // them apart from the rest of the stage is what makes the next regression visible.
152:     const auto t_files = std::chrono::steady_clock::now();
153:     // ---- the index and the dense weights
154:     {
155:         std::ifstream idx(rt_dir + "/dense.txt");
156:         if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
157:         std::string line;
158:         while (std::getline(idx, line)) {
159:             if (line.empty()) continue;
160:             std::istringstream is(line);
161:             Tensor t;
162:             is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
163:             if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
164:             tensors_.push_back(t);
165:         }
166:         std::vector<uint8_t> blob;
167:         if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
168:         const cudaError_t alloc = cudaMalloc((void**) &dense_, blob.size());
169:         if (alloc != cudaSuccess) {
170:             size_t free_bytes = 0, total_bytes = 0;
171:             const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
172:             err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
173:                   "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
174:                   (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
175:             return false;
176:         }
177:         cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
178:         vram_ += blob.size();
179:     }
180:     // ---- the 512 routed experts, one blob each
181:     {
182:         const uint64_t bytes = (uint64_t) g.n_expert * strata::kernels::cpu::BLOB;
183:         // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
184:         // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
185:         FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
186:         if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
187:         struct Closer {
188:             FILE* f;
189:             ~Closer() { if (f != nullptr) std::fclose(f); }
190:         } closer{f};
191:         if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
192:         std::vector<uint8_t> chunk(64u << 20);
193:         for (uint64_t off = 0; off < bytes;) {
194:             const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
195:             if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
196:             cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
197:             off += n;
198:         }
199:         vram_ += bytes;
200:     }
201:     const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
202:                               "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
203:                               "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
204:     for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }
205: 
206:     // ---- the layer's own K/V (dense attention: no indexer state is read)
207:     const strata::kernels::QsaShapes s = shapes_of(g);
208:     const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
209:     // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
210:     // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
211:     // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
212:     int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
213:     // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
214:     // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
215:     const bool kv_hybrid_was = qsa_kv_hybrid();
216:     const bool kv_int8_was = qsa_kv_int8();
217:     qsa_set_kv_hybrid(false);
218:     if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
219:     uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
220:     if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
221:     if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
222:         if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
```

### src/core/mtp.cpp, lines 258–284
```
258:         mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
259:         lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
260:         xn_ = b.take<float>(T * HC * N);
261:         xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
262:         qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
263:         kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
264:         attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
265:         attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
266:         logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
267:         shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
268:         sample_ = b.take<float>(T * N);
269:         hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
270:         grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
271:         grp_counts_ = b.take<int32_t>(4);
272:         hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
273:         hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
274:         sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
275:         x_bf16_ = b.take<uint16_t>(N);
276:         out_ids_ = b.take<int32_t>(T + 4);
277:         probs_ = b.take<float>(T + 4);
278:         dummy_inj_ = b.take<float>(HC);
279:     };
280:     Bump count;
281:     carve(count);
282:     if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
283:     cudaMemset(arena_, 0, count.used);
284:     Bump real;
```

### src/core/mtp.cpp, lines 522–553
```
522:             fused_gr_read_multi(fa, T, xn_, cs);
523:         }
524:         // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
525:         for (int t = 0; t < T; ++t) {
526:             bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
527:             if (native_router_enabled()) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
528:             else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
529:         }
530:         moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
531:                            grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
532:         quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
533:         moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
534:                        hit_xs_, hit_scratch_, parts_, cs);
535:         NativeSharedWeights nsw;
536:         nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
537:         nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
538:         nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
539:         nsw.q8_1 = xq_;
540:         const SForm none{};
541:         for (int t = 0; t < T; ++t) {
542:             f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, cs);
543:             shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
544:                           nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
545:                           N, g.n_ff, 32, cs, mixed_ + t * N, &nsw);
546:             if (native_moe_combine_enabled())
547:                 native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
548:             else
549:                 moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
550:             gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
551:         }
552:         // ---- the final mixer and the main model's head
553:         for (int t = 0; t < T; ++t)
```

### src/kernels/cuda/iq_kernels.cu, lines 1244–1290
```
1244:         case 13: dq_q5_k(vx, ibs, y, tid); break;
1245:         case 7: dq_q5_1(vx, ibs, y, tid); break;
1246:         case 8: dq_q8_0(vx, ibs, y, tid); break;
1247:         case 30: dq_bf16(vx, ibs, y, tid); break;
1248:         default: break;
1249:     }
1250: }
1251: 
1252: // flat: superblock i -> y + 256 i
1253: template<typename dst_t>
1254: __global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y) {
1255:     const int64_t i = blockIdx.x;
1256:     dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
1257: }
1258: // gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
1259: __global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
1260:                                   __half* __restrict__ y) {
1261:     const int64_t i = blockIdx.x;
1262:     const int parity = blockIdx.y;
1263:     const int64_t r = i / per_row, c = i % per_row;
1264:     dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
1265: }
1266: 
1267: // the types dq_dispatch dequantizes
1268: bool is_iq(int t) {
1269:     return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
1270:            t == 12 || t == 13 || t == 7 || t == 8;
1271: }
1272: // values per block of the types the grouped expert kernels take (0 = none)
1273: int gu_qk(int t) {
1274:     switch (t) {
1275: #define STRATA_QK(T) case T: return Fmt<T>::qk;
1276:         STRATA_GU_FMTS(STRATA_QK)
1277: #undef STRATA_QK
1278:         default: return 0;
1279:     }
1280: }
1281: int d_qk(int t) {
1282:     switch (t) {
1283: #define STRATA_QK(T) case T: return Fmt<T>::qk;
1284:         STRATA_D_FMTS(STRATA_QK)
1285: #undef STRATA_QK
1286:         default: return 0;
1287:     }
1288: }
1289: 
1290: bool env_on(const char* name) {
```

### src/kernels/cuda/iq_kernels.cu, lines 1417–1455
```
1417: bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
1418:     const int qg = gu_qk(gu_type), qd = d_qk(d_type);
1419:     return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
1420:            n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
1421: }
1422: 
1423: NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
1424:     NativeExpertLayout L;
1425:     L.gu_type = gu_type;
1426:     L.d_type = d_type;
1427:     L.n_embd = n_embd;
1428:     L.n_ff = n_ff;
1429:     L.gu_row = iq_row_bytes(gu_type, n_embd);
1430:     L.d_row = iq_row_bytes(d_type, n_ff);
1431:     L.up_off = (size_t) n_ff * L.gu_row;
1432:     L.down_off = 2 * L.up_off;
1433:     L.bytes = L.down_off + (size_t) n_embd * L.d_row;
1434:     return L;
1435: }
1436: 
1437: size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
1438:     const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
1439:     return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
1440: }
1441: 
1442: void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
1443:                            const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
1444:                            int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
1445:     if (cap_groups <= 0 || cap_entries <= 0) return;
1446:     cudaStream_t s = (cudaStream_t) stream;
1447:     const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
1448:     float* gate = (float*) scratch;
1449:     float* up = (float*) ((uint8_t*) scratch + fa);
1450:     float* h = (float*) ((uint8_t*) scratch + 2 * fa);
1451:     block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
1452:     const auto* X = (const block_q8_1*) x_q8_1;
1453:     const dim3 ggu((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) cap_groups);
1454:     switch (L.gu_type) {
1455: #define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
```

### src/program/generate.cpp, lines 2133–2175
```
2133:     }
2134: 
2135:     // ---- --split-skip-if-fits: before any later stage loads, does CUDA0 alone hold every profiled pair?  What it
2136:     // still has to allocate on one GPU is the whole session (the KV of every layer), the drafter and the head
2137:     // (kDrafterMib below), the verify windows and the reserve; the prompt path borrows from the cache.  If the
2138:     // profile's pairs fit in what is left, a split would only add the hand-offs: run on CUDA0 alone.
2139:     if (multi_gpu && split_auto && o.split_skip_if_fits) {
2140:         std::vector<std::pair<int32_t, int32_t>> prof;
2141:         int64_t pslots = 0;
2142:         std::string perr;
2143:         const bool remote = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
2144:         if (remote || o.expert_profile.empty() ||
2145:             !strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, prof, pslots, perr)) {
2146:             std::fprintf(stderr, "strata generate: --split-skip-if-fits: %s; the split stays\n",
2147:                          remote ? "remote expert caches are in use" : perr.empty() ? "no expert profile" : perr.c_str());
2148:         } else {
2149:             const auto& lay = strata::kernels::cpu::expert_layout();
2150:             int64_t pairs_bytes = 0;
2151:             for (const auto& pr : prof)
2152:                 pairs_bytes += native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
2153:             size_t fb = 0, tb = 0;
2154:             cudaMemGetInfo(&fb, &tb);
2155:             const int64_t session = (int64_t) strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers);
2156:             const int64_t held_back = session + (((int64_t) o.vram_reserve_mib + 1000 + 96) << 20);   // + drafter/head, windows
2157:             const int64_t room = (int64_t) fb - held_back;
2158:             cudaDeviceProp dp{};
2159:             cudaGetDeviceProperties(&dp, 0);
2160:             if (pairs_bytes <= room) {
2161:                 std::fprintf(stderr, "strata generate: layer split skipped (--split-skip-if-fits): CUDA0 (%s) holds all "
2162:                                      "%zu profiled pairs (%.2f GiB) with the session (%.2f GiB, %lld-token context), "
2163:                                      "the drafter and the reserve: %.2f GiB free, %.2f GiB to spare - one GPU\n",
2164:                              dp.name, prof.size(), (double) pairs_bytes / 1073741824.0, (double) session / 1073741824.0,
2165:                              (long long) o.max_context, (double) fb / 1073741824.0,
2166:                              (double) (room - pairs_bytes) / 1073741824.0);
2167:                 multi_gpu = false;
2168:                 split_auto = false;
2169:                 split_devs.clear();
2170:                 split_at.clear();
2171:                 o.layer_split.clear();
2172:             } else {
2173:                 std::fprintf(stderr, "strata generate: --split-skip-if-fits: CUDA0 (%s) would hold only %.2f of the "
2174:                                      "profile's %.2f GiB (%.2f GiB free, %.2f GiB for the session, drafter and "
2175:                                      "reserve): the split stays\n", dp.name,
```

### src/program/generate.cpp, lines 2293–2320
```
2293:     // to 6 tokens, 74.1 MiB of device buffers", on every boot) and the drafter is loaded on ONE stage - the
2294:     // last one, which is also the only one that holds the head.  On the two identical 8 GB cards that GiB was
2295:     // the entire difference between CUDA0's cache and CUDA1's: 814 slots against 188, 2026-09-30.  Both of
2296:     // those allocations are already made before a stage's cache is sized, so what has to be held back here is
2297:     // the windows and - only on the stage that carries them - the drafter and the head.
2298:     const int64_t kWindowMib = 96;       // the verify windows; 75 MiB measured, rounded up
2299:     const int64_t kDrafterMib = 1000;    // the MTP drafter (839 MiB) + the head, on the last stage only
2300:     // #340: with the own prompt buffers chosen by the split's rule (not asked for with --no-prefill-borrow) the
2301:     // boundary is searched as the borrowing configuration would (no reserve): the reserve then only makes the caches
2302:     // smaller, which measured cost no decode (K=28 on 9070 XT + R9700: 58.4 tok/s own vs 58.5 borrowing), while a
2303:     // search with the reserve moved the boundary to K=32 and decode to 54.8. STRATA_SPLIT_OWN_PLACE=reserve: the
2304:     // search sees the reserve.
2305:     static const bool place_with_reserve = [] {
2306:         const char* v = std::getenv("STRATA_SPLIT_OWN_PLACE");
2307:         return v != nullptr && std::string(v) == "reserve";
2308:     }();
2309:     auto stage_room = [&](int dev, bool later, bool drafter, bool search = false) -> int64_t {
2310:         const strata::core::OnDevice on(dev);
2311:         size_t fb = 0, tb = 0;
2312:         if (const cudaError_t e = cudaMemGetInfo(&fb, &tb); e != cudaSuccess)
2313:             std::fprintf(stderr, "strata generate: layer split: CUDA%d free memory: %s\n", dev < 0 ? 0 : dev,
2314:                          cudaGetErrorString(e));
2315:         const int64_t pf = search && split_own_auto && !place_with_reserve ? 0 : split_pf_mib;
2316:         const int64_t reserve = ((int64_t) o.vram_reserve_mib + pf + (later ? kWindowMib : 0) +
2317:                                  (drafter ? kDrafterMib : 0)) << 20;
2318:         return std::max<int64_t>((int64_t) fb - reserve, 0);
2319:     };
2320:     if (multi_gpu && split_auto) {
```

### src/program/generate.cpp, lines 2880–2905
```
2880:                      (long long) prefilled, (long long) (per_layer ? xcache.slots() : want));
2881:     }
2882: 
2883:     for (auto& stp : stages) {
2884:         GpuStage& st = *stp;
2885:         const auto& lay = strata::kernels::cpu::expert_layout();
2886:         // the drafter and the head are already allocated by now (they load above, before this), so what is left
2887:         // to hold back is the windows - and `free_b` has already lost the drafter.
2888:         const int64_t room = stage_room(st.dev, true, false);
2889:         const strata::core::OnDevice on(st.dev);
2890:         std::vector<int64_t> sized;
2891:         int64_t used = 0;
2892:         for (const auto& pr : st.profile) {
2893:             const int64_t b = native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
2894:             if (used + b > room) break;
2895:             used += b;
2896:             sized.push_back((int64_t) lay.blob_bytes(pr.first));
2897:         }
2898:         if (sized.empty() ||
2899:             !(native_pack ? st.cache.open_sized(sized, g.n_layers, g.n_expert, err)
2900:                           : st.cache.open((int64_t) sized.size(), g.n_layers, g.n_expert, (int64_t) lay.max_blob, err))) {
2901:             std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
2902:                          sized.empty() ? "no room" : err.c_str());
2903:             return 1;
2904:         }
2905:         int64_t filled = 0;
```
