# 3ds_nintendo_llm

> **Work in progress** — This project is in early development. Expect breaking changes.

A from-scratch LLM inference engine for the Nintendo 3DS, written in C++17.

The goal is to run a small language model (targeting DistilGPT-2, 82M parameters) directly on 3DS hardware, with no external dependencies beyond the standard library and libctru.

## Status

| Component | Status |
|---|---|
| Tokenizer (BPE) | Done |
| Model weight loader | Planned |
| Forward pass (transformer) | Planned |
| Sampler | Planned |
| 3DS UI | Planned |

## Architecture

- **Tokenizer** : BPE tokenizer compatible with DistilGPT-2, loaded from a custom binary format (`.bin`). Uses GPT-2's byte-level encoding and binary search on merge rules.
- **Model** : int8 quantized weights in a custom binary format, designed to fit within the 3DS memory budget (~96MB on old 3DS, ~200MB on New 3DS).

## Building (host, for testing)

Requires `g++` with C++17 support.

```bash
make        # build
make run    # build and run
make re     # clean rebuild
```

The tokenizer test binary expects `tokenizer.bin` in the working directory. You can generate it from the Python converter in `model converter/`.

## Target hardware

- Nintendo 3DS / New Nintendo 3DS
- ARM11 MPCore, 268 MHz (old) / 804 MHz (New 3DS)
- 128 MB RAM (old) / 256 MB RAM (New 3DS)
- Built with devkitARM / libctru (3DS build not yet implemented)
