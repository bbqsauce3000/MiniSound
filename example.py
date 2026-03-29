from minisound import Song

song = Song(bpm=120, sr=44100)

track = song.add_track(name="track", gain=1.0, pan=0.0, send=0.0)

track.note("C5", dur=1)
track.note("E5", dur=1)
track.note("G5", dur=1)
track.note("C6", dur=1)

song.save("example.wav")
