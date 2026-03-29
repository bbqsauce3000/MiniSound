# MiniSound

Note that NumPy is needed, install with `pip install numpy`, then run the example with `python example.py `

MiniSound is a compact, single‑file audio synthesis and sequencing engine implemented in Python.  
It provides waveform generation, envelopes, filtering, effects processing, multi‑track scheduling,  
and WAV export in under 600 lines of code.

MiniSound is designed so that **song files are ordinary Python programs**.  
A “song” is simply a script that imports MiniSound, schedules events, and produces a `.wav` file.  
If a user wants additional behavior or new features, they may implement them directly in the  
song script or modify the engine itself.

---

## Overview

MiniSound includes:

- Pulse, triangle, and noise oscillators  
- ADSR envelope generation  
- Vibrato modulation  
- Low‑pass, high‑pass, and saturation stages  
- Bit‑crushing and downsampling  
- Stereo panning and unison detuning  
- Chorus, delay, comb, and all‑pass filters  
- Reverb  
- Compression, limiting, and tape saturation  
- Oversampling and anti‑alias filtering  
- Track‑based event scheduling  
- Master bus processing  
- WAV rendering and export

---

## Documentation

Detailed explanations and advanced usage notes are provided in:

- `documentation.txt`  
- `documentationadvanced.txt`

---
## Features to add...
A actual, proper DSL?

---
## License

This project is released under **The Unlicense**.  
It is dedicated to the public domain.  
You may use, modify, distribute, and build upon it without restriction.
