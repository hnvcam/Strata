# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Saved local MTP comparison

Before repeating the IQ4_XS Q2 versus Q4_K_M MTP comparison, read
[the saved full-vocabulary results](bench/results/2026-10-03-iq4-xs-mtp/full-vocab/report.md).
Measured 2026-10-03 on the i5-13500, RTX 5070 Ti + RTX 4060, 64 GB RAM: both MTP
models ran on the 5070 Ti with all 248,320 draft tokens and identical settings.
Show these results first when asked to redo the comparison; avoid repeating it merely
to recover forgotten measurements. Raw measurements are saved beside the report.
The experimental code, models and configs were removed and original Strata restored.
Historical test paths in those records are not active configurations. Do not apply
benchmark settings or start more experiments unless the user asks.

The user subsequently chose full draft vocabulary for the restored Q2 setup.
`strata-iq4_xs.json` labels it `full`; its configured MTP runtime contains all
248,320 token IDs in `draft_vocab.bin`. Keep this custom list when restoring settings;
do not replace it with the English subset. Original setup preserves custom lists.
