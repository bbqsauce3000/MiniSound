# MiniSound

Note that NumPy is needed, install with `pip install numpy`, then run the example with `python example.py `

MiniSound is designed so that **song files are ordinary Python programs**.  
A “song” is simply a script that imports MiniSound, schedules events, and produces a `.wav` file.  
If a user wants additional behavior or new features, they may implement them directly in the  
song script or modify the engine itself.

---
## Example
```
from minisound import *

song = Song(bpm=140)
lead = song.add_track("lead")

lead.note("C5")
lead.note("D5")
lead.note("G5")

song.save("example.wav")
```
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
