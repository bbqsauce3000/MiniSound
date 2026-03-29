import numpy as np
import wave
import random
import math

class MiniSoundError(Exception):
    pass

def ensure(cond, msg):
    if not cond:
        raise MiniSoundError(msg)

BASE_NOTES = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}

def parse_note_name(name):
    ensure(len(name)>=2, f"Invalid note '{name}'")
    letter = name[0].upper()
    ensure(letter in BASE_NOTES, f"Invalid note letter '{letter}'")
    idx = len(name)-1
    while idx>0 and name[idx].isdigit():
        idx -= 1
    accidental = name[1:idx+1]
    octave = name[idx+1:]
    ensure(octave.isdigit(), f"Invalid octave '{name}'")
    semitone = BASE_NOTES[letter]
    for c in accidental:
        if c == '#': semitone += 1
        elif c == 'b': semitone -= 1
        else: raise MiniSoundError(f"Invalid accidental '{name}'")
    return semitone, int(octave)

def note_to_freq(note):
    semitone, octave = parse_note_name(note)
    midi = 12*(octave+1)+semitone
    return 440.0 * (2 ** ((midi-69)/12))

def crush(x, bits=None):
    if bits is None: return x
    levels = 2**bits
    return np.round(x*levels)/levels

def downsample(x, factor=None):
    if factor is None or factor<=1: return x
    return x[::factor].repeat(factor)[:len(x)]

def adsr_envelope(n, sr, attack=0.005, decay=0.02, sustain=0.85, release=0.03):
    env = np.ones(n, dtype=np.float32)
    a = min(int(sr * attack), n)
    d = min(int(sr * decay), max(n - a, 0))
    r = min(int(sr * release), n)
    if a > 0:
        env[:a] = np.linspace(0, 1.0, a)
    end = a + d
    if d > 0 and end <= n:
        env[a:end] = np.linspace(1.0, sustain, d)
    sustain_start = end
    sustain_end = max(n - r, sustain_start)
    if sustain_end > sustain_start:
        env[sustain_start:sustain_end] = sustain
    if r > 0 and sustain_end < n:
        env[sustain_end:] = np.linspace(sustain, 0, n - sustain_end)
    return env

def apply_vibrato(x, sr, depth=0.003, speed=5.0):
    n = len(x)
    t = np.arange(n) / sr
    vib = np.sin(2*np.pi*speed*t) * depth
    return x * (1 + vib)

def lowpass(x, amt=0.15):
    if len(x)==0: return x
    return np.concatenate([[x[0]], x[:-1]*(1-amt) + x[1:]*amt])

def highpass(x, amt=0.02):
    if len(x)==0: return x
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1,len(x)):
        y[i] = amt*(y[i-1] + x[i] - x[i-1])
    return y

def saturate(x, amt=0.15):
    return np.tanh(x * (1 + amt))

def pan_stereo(mono, pan):
    left = mono * (1 - pan)
    right = mono * (1 + pan)
    return np.column_stack([left, right])

def pulse(freq, dur, vol, sr, duty=0.18, bits=None, rate=None, unison=1, spread=0.02):
    n = max(1, int(dur*sr))
    t = np.linspace(0, dur, n, endpoint=False)
    out = np.zeros((n,), dtype=np.float32)
    for i in range(unison):
        det = (i - (unison-1)/2) * spread
        duty_i = max(0.05, min(0.95, duty * (1 + random.uniform(-0.05, 0.05))))
        freq_i = freq * (1 + det)
        wave_mono = ((t*freq_i)%1.0 < duty_i).astype(np.float32)*2 - 1
        out += wave_mono
    out /= max(1, unison)
    env = adsr_envelope(n, sr)
    out *= env * (vol*0.6)
    out = apply_vibrato(out, sr)
    out = lowpass(out)
    out = saturate(out)
    out = crush(out, bits)
    out = downsample(out, rate)
    return np.column_stack([out, out])

def tri(freq, dur, vol, sr, bits=None, rate=None, unison=1, spread=0.01):
    n = max(1, int(dur*sr))
    t = np.linspace(0, dur, n, endpoint=False)
    out = np.zeros((n,), dtype=np.float32)
    for i in range(unison):
        det = (i - (unison-1)/2) * spread
        freq_i = freq * (1 + det)
        wave_mono = 2*np.abs(2*((t*freq_i)%1.0)-1)-1
        out += wave_mono
    out /= max(1, unison)
    env = adsr_envelope(n, sr)
    out *= env * (vol*0.55)
    out = apply_vibrato(out, sr)
    out = lowpass(out)
    out = saturate(out)
    out = crush(out, bits)
    out = downsample(out, rate)
    return np.column_stack([out, out])

def noise(dur, vol, sr, bits=None, rate=None):
    n = max(1, int(dur*sr))
    wave_mono = np.random.uniform(-1,1,n)
    env = adsr_envelope(n, sr, attack=0.003, decay=0.02, sustain=0.5, release=0.03)
    wave_mono *= env * (vol*0.4)
    wave_mono = lowpass(wave_mono, amt=0.25)
    wave_mono = crush(wave_mono, bits)
    wave_mono = downsample(wave_mono, rate)
    return np.column_stack([wave_mono, wave_mono])

def chorus_stereo(buf, sr, depth=0.0025, speed=0.8):
    n = len(buf)
    t = np.arange(n) / sr
    max_delay = int(sr * 0.03)
    out = np.zeros_like(buf)
    lfo_L = (0.5 + 0.5*np.sin(2*np.pi*speed*t)) * depth * max_delay
    lfo_R = (0.5 + 0.5*np.cos(2*np.pi*speed*t)) * depth * max_delay
    for i in range(n):
        dL = lfo_L[i]
        dR = lfo_R[i]
        iL = max(0, i - dL)
        iR = max(0, i - dR)
        iL0 = int(iL)
        iL1 = min(iL0 + 1, n - 1)
        fracL = iL - iL0
        iR0 = int(iR)
        iR1 = min(iR0 + 1, n - 1)
        fracR = iR - iR0
        left = (1-fracL)*buf[iL0,0] + fracL*buf[iL1,0]
        right = (1-fracR)*buf[iR0,1] + fracR*buf[iR1,1]
        out[i,0] = 0.7*buf[i,0] + 0.3*left
        out[i,1] = 0.7*buf[i,1] + 0.3*right
    return out

def delay_stereo(buf, sr, delay_sec=0.25, feedback=0.25, wet=0.25, pingpong=False):
    n = len(buf)
    delay = int(delay_sec * sr)
    out = np.zeros((n+delay*4,2), dtype=np.float32)
    out[:n] += buf
    for i in range(n):
        out[i+delay] += buf[i] * feedback
        if pingpong:
            out[i+delay,0] += buf[i,1] * feedback
            out[i+delay,1] += buf[i,0] * feedback
    result = out[:n] * (1-wet) + out[:n] * wet
    return result

def comb_filter(x, delay_samples, feedback):
    if delay_samples <= 0: return x
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[i]
        if i - delay_samples >= 0:
            out[i] += out[i-delay_samples] * feedback
    return out

def allpass_filter(x, delay_samples, feedback):
    if delay_samples <= 0:
        return x
    feedback = max(-0.7, min(0.7, feedback))
    out = np.zeros_like(x)
    for i in range(len(x)):
        inp = x[i]
        if i - delay_samples >= 0:
            delayed = out[i - delay_samples]
            out[i] = delayed + feedback * (inp - delayed)
        else:
            out[i] = inp
    return out

def schroeder_reverb(buf, sr, room_size=0.8, wet=0.25):
    mono = buf.mean(axis=1)
    n = len(mono)
    comb_delays = [int(sr * d * room_size) for d in (0.0297, 0.0371, 0.0411, 0.0437)]
    comb_out = np.zeros(n, dtype=np.float32)
    for d in comb_delays:
        comb_out += comb_filter(mono, max(1,d), feedback=0.7*room_size)
    comb_out /= len(comb_delays)
    ap1 = allpass_filter(comb_out, int(sr*0.005), feedback=0.5)
    ap2 = allpass_filter(ap1, int(sr*0.0017), feedback=0.5)
    left = np.roll(ap2, int(sr*0.003))
    right = np.roll(ap2, int(sr*0.006))
    reverb = np.column_stack([left, right]) * wet
    return buf * (1-wet) + reverb

def rms_level(buf, window=1024):
    if len(buf)==0: return 0.0
    mono = buf.mean(axis=1)
    return np.sqrt(np.mean(mono*mono))

def compressor(buf, threshold=0.6, ratio=3.0, attack_ms=10, release_ms=50, sr=44100):
    if len(buf)==0: return buf
    mono = buf.mean(axis=1)
    out = np.zeros_like(buf)
    gain = 1.0
    attack = max(1, int(sr * attack_ms/1000.0))
    release = max(1, int(sr * release_ms/1000.0))
    for i in range(len(mono)):
        level = abs(mono[i])
        if level > threshold:
            target = 1.0 / (1.0 + (level-threshold)*(ratio-1))
        else:
            target = 1.0
        if target < gain:
            gain += (target - gain) / attack
        else:
            gain += (target - gain) / release
        out[i,0] = buf[i,0] * gain
        out[i,1] = buf[i,1] * gain
    return out

def soft_limiter(buf, threshold=0.95):
    if len(buf)==0: return buf
    return np.tanh(buf / threshold) * threshold

def stereo_widen(buf, amount=0.25):
    if len(buf)==0: return buf
    mid = (buf[:,0] + buf[:,1]) * 0.5
    side = (buf[:,0] - buf[:,1]) * (1+amount)
    left = mid + side*0.5
    right = mid - side*0.5
    return np.column_stack([left, right])

def make_layer(freq, dur, vol, sr, wave="pulse", bits=None, rate=None, duty=0.18, unison=1, spread=0.02, chorus=False):
    if wave == "pulse":
        buf = pulse(freq, dur, vol, sr, duty=duty, bits=bits, rate=rate, unison=unison, spread=spread)
    elif wave == "pulse50":
        buf = pulse(freq, dur, vol, sr, duty=0.5, bits=bits, rate=rate, unison=unison, spread=spread)
    elif wave == "tri":
        buf = tri(freq, dur, vol, sr, bits=bits, rate=rate, unison=unison, spread=spread)
    elif wave == "noise":
        buf = noise(dur, vol, sr, bits=bits, rate=rate)
    else:
        raise MiniSoundError(f"Invalid wave '{wave}'")
    if chorus:
        buf = chorus_stereo(buf, sr)
    return buf

def mix_layers(layers):
    if not layers:
        return np.zeros((1,2), dtype=np.float32)
    L = max(len(l) for l in layers)
    out = np.zeros((L,2), dtype=np.float32)
    for l in layers:
        out[:len(l)] += l
    return out

def db_to_linear(db):
    return 10 ** (db / 20.0)

def linear_to_db(x, floor=-120.0):
    x = max(x, 1e-12)
    return max(floor, 20.0 * math.log10(abs(x)))

def rms(buf):
    if len(buf)==0: return 0.0
    mono = buf.mean(axis=1)
    return np.sqrt(np.mean(mono*mono))

def peak_level(buf):
    if len(buf)==0: return 0.0
    return np.max(np.abs(buf))

def tape_saturate(x, drive=0.8, mix=0.5):
    driven = np.tanh(x * (1 + drive))
    return x * (1-mix) + driven * mix

def oversample(buf, factor=2):
    if factor <= 1: return buf
    n = len(buf)
    idx = np.linspace(0, n-1, n*factor)
    left = np.interp(idx, np.arange(n), buf[:,0])
    right = np.interp(idx, np.arange(n), buf[:,1])
    return np.column_stack([left, right])

def anti_alias_lowpass(buf, cutoff=0.45):
    if len(buf)==0: return buf
    amt = max(0.001, min(0.45, cutoff))
    out = np.zeros_like(buf)
    out[0] = buf[0]
    for i in range(1,len(buf)):
        out[i] = out[i-1] * (1-amt) + buf[i] * amt
    return out

def soft_knee_compressor(buf, threshold_db=-12.0, ratio=3.0, attack_ms=10, release_ms=100, sr=44100):
    if len(buf)==0: return buf
    mono = buf.mean(axis=1)
    out = np.zeros_like(buf)
    env = 1.0
    attack = max(1, int(sr * attack_ms/1000.0))
    release = max(1, int(sr * release_ms/1000.0))
    thresh_lin = db_to_linear(threshold_db)
    for i in range(len(mono)):
        level = abs(mono[i])
        if level <= thresh_lin:
            target_gain = 1.0
        else:
            over = level / thresh_lin
            gain_db = - (20.0 * math.log10(over)) * (1 - 1/ratio)
            target_gain = db_to_linear(gain_db)
        if target_gain < env:
            env += (target_gain - env) / attack
        else:
            env += (target_gain - env) / release
        out[i,0] = buf[i,0] * env
        out[i,1] = buf[i,1] * env
    return out

def lookahead_limiter(buf, threshold=0.98, makeup_db=0.0):
    if len(buf)==0: return buf
    peak = peak_level(buf)
    if peak <= threshold:
        if makeup_db != 0.0:
            return buf * db_to_linear(makeup_db)
        return buf
    gain = threshold / peak
    out = buf * gain
    if makeup_db != 0.0:
        out *= db_to_linear(makeup_db)
    return out

def auto_gain_compensate(buf, target_rms=0.1):
    cur = rms(buf)
    if cur <= 1e-6: return buf
    gain = target_rms / cur
    return buf * gain

class Track:
    def __init__(self, bpm=120, sr=44100, name="track", gain=1.0, pan=0.0, send=0.0):
        self.bpm = bpm
        self.sr = sr
        self.events = []
        self.time = 0.0
        self.name = name
        self.gain = gain
        self.pan = pan
        self.send = send
        self.comp_threshold_db = -12.0
        self.comp_ratio = 3.0
        self.comp_attack = 10
        self.comp_release = 100
        self.last_rms = 0.0
        self.last_peak = 0.0

    def beats(self, n):
        return (60.0/self.bpm)*n

    def note(self, name, dur=1, vol=0.4, wave="pulse", bits=None, rate=None, layers=None, send=None, unison=1, spread=0.02, chorus=False):
        d = self.beats(dur)
        base_freq = note_to_freq(name)
        base_freq *= (1 + random.uniform(-0.002, 0.002))
        vol *= (1 + random.uniform(-0.10, 0.10))

        if bits is not None:
            bits = max(3, int(bits * (1 + random.uniform(-0.15, 0.15))))
        if rate is not None:
            rate = max(1, int(rate * (1 + random.uniform(-0.20, 0.20))))

        send_amt = self.send if send is None else send

        if layers is None:
            audio = make_layer(base_freq, d, vol, self.sr, wave=wave, bits=bits, rate=rate, unison=unison, spread=spread, chorus=chorus)
            pan = random.uniform(-0.3, 0.3) + self.pan
            audio = pan_stereo(audio[:,0], pan)
        else:
            layer_bufs = []
            for L in layers:
                w = L.get("wave", wave)
                detune_cents = L.get("detune", 0.0)
                detune_ratio = 2 ** (detune_cents / 1200.0)
                f = base_freq * detune_ratio
                lv = vol * L.get("vol", 1.0)
                lb = L.get("bits", bits)
                lr = L.get("rate", rate)
                duty = L.get("duty", 0.18)
                l_unison = L.get("unison", unison)
                l_spread = L.get("spread", spread)
                l_chorus = chorus or L.get("chorus", False)
                buf = make_layer(f, d, lv, self.sr, wave=w, bits=lb, rate=lr, duty=duty, unison=l_unison, spread=l_spread, chorus=l_chorus)
                pan_off = L.get("pan", 0.0)
                pan = self.pan + pan_off
                buf = pan_stereo(buf[:,0], pan)
                layer_bufs.append(buf)
            audio = mix_layers(layer_bufs)

        self.events.append((audio * self.gain, self.time, send_amt))
        self.time += d + random.uniform(-0.002, 0.002)

    def sample(self, filename, dur=1.0, vol=0.4, pitch=1.0, send=None):
        send_amt = self.send if send is None else send
        if dur is None:
            data = load_wav(filename, self.sr)
            if pitch!=1.0:
                n = int(len(data)/pitch)
                idx = np.linspace(0,len(data)-1,n)
                data = np.stack([
                    np.interp(idx,np.arange(len(data)),data[:,0]),
                    np.interp(idx,np.arange(len(data)),data[:,1])
                ],axis=1)
            audio = data*vol*self.gain
            audio = pan_stereo(audio[:,0], self.pan)
            self.events.append((audio,self.time,send_amt))
            self.time += len(audio)/self.sr
            return
        d = self.beats(dur)
        audio = sample_play(filename,d,vol,self.sr,pitch_shift=pitch)*self.gain
        audio = pan_stereo(audio[:,0], self.pan)
        self.events.append((audio,self.time,send_amt))
        self.time += d

    def chord(self, names, dur=1, vol=0.3, wave="pulse", layers=None, send=None):
        d = self.beats(dur)
        parts = []
        for n in names.split():
            base_freq = note_to_freq(n)
            if layers is None:
                parts.append(make_layer(base_freq, d, vol, self.sr, wave=wave))
            else:
                layer_bufs = []
                for L in layers:
                    w = L.get("wave", wave)
                    detune_cents = L.get("detune", 0.0)
                    detune_ratio = 2 ** (detune_cents / 1200.0)
                    f = base_freq * detune_ratio
                    lv = vol * L.get("vol", 1.0)
                    lb = L.get("bits", None)
                    lr = L.get("rate", None)
                    duty = L.get("duty", 0.18)
                    l_unison = L.get("unison", 1)
                    l_spread = L.get("spread", 0.02)
                    l_chorus = L.get("chorus", False)
                    buf = make_layer(f, d, lv, self.sr, wave=w, bits=lb, rate=lr, duty=duty, unison=l_unison, spread=l_spread, chorus=l_chorus)
                    pan_off = L.get("pan", 0.0)
                    pan = self.pan + pan_off
                    buf = pan_stereo(buf[:,0], pan)
                    layer_bufs.append(buf)
                parts.append(mix_layers(layer_bufs))

        L = max(len(p) for p in parts)
        out = np.zeros((L,2),dtype=np.float32)
        for p in parts:
            out[:len(p)] += p
        out *= self.gain
        send_amt = self.send if send is None else send
        self.events.append((out,self.time,send_amt))
        self.time += d

    def arp(self, names, steps=8, dur=0.25, vol=0.25, wave="pulse", layers=None, send=None):
        seq = names.split()
        for i in range(steps):
            self.note(seq[i%len(seq)], dur=dur, vol=vol, wave=wave, layers=layers, send=send)

    def _apply_echo(self, buf, delay_ms=110, feedback=0.12):
        delay = int(self.sr*delay_ms/1000)
        if delay<=0 or delay>=len(buf): return buf
        echo = np.zeros_like(buf)
        echo[delay:] = buf[:-delay]*feedback
        return buf + echo

    def render(self):
        if not self.events:
            return np.zeros((1,2),dtype=np.float32), np.zeros((1,2),dtype=np.float32)
        end = max(at+len(a)/self.sr for a,at,_ in self.events)
        total = int(end*self.sr)+1
        buf = np.zeros((total,2),dtype=np.float32)
        send_buf = np.zeros((total,2),dtype=np.float32)
        for audio,at,send_amt in self.events:
            s = int(at*self.sr)
            buf[s:s+len(audio)] += audio
            if send_amt and send_amt>0:
                send_buf[s:s+len(audio)] += audio * send_amt
        buf = soft_knee_compressor(buf, threshold_db=self.comp_threshold_db, ratio=self.comp_ratio,
                                   attack_ms=self.comp_attack, release_ms=self.comp_release, sr=self.sr)
        buf = self._apply_echo(buf)
        self.last_rms = rms(buf)
        self.last_peak = peak_level(buf)
        buf *= self.gain
        send_buf *= self.gain
        return buf, send_buf

class Song:
    def __init__(self, bpm=120, sr=44100):
        self.bpm = bpm
        self.sr = sr
        self.tracks = []
        self.master_volume_db = -3.0
        self.master_saturation_drive = 0.6
        self.master_saturation_mix = 0.25
        self.master_target_rms = 0.08
        self.master_limiter_threshold = 0.98
        self.master_limiter_makeup_db = 0.0

    def add_track(self, name="track", gain=1.0, pan=0.0, send=0.0):
        t = Track(bpm=self.bpm, sr=self.sr, name=name, gain=gain, pan=pan, send=send)
        self.tracks.append(t)
        return t

    def render(self):
        if not self.tracks:
            return np.zeros((1,2), dtype=np.float32)
        rendered = [t.render() for t in self.tracks]
        bufs = [r[0] for r in rendered]
        sends = [r[1] for r in rendered]
        L = max(len(r) for r in bufs)
        mix = np.zeros((L,2), dtype=np.float32)
        send_mix = np.zeros((L,2), dtype=np.float32)
        for r in bufs:
            mix[:len(r)] += r
        for s in sends:
            send_mix[:len(s)] += s
        reverb_return = schroeder_reverb(send_mix, self.sr, room_size=0.85, wet=0.35)
        mix += reverb_return * 0.6
        mix_os = oversample(mix, factor=2)
        mix_os = tape_saturate(mix_os, drive=self.master_saturation_drive, mix=self.master_saturation_mix)
        mix_os = anti_alias_lowpass(mix_os, cutoff=0.22)
        mix = mix_os[::2][:len(mix)]
        mix = soft_knee_compressor(mix, threshold_db=-14.0, ratio=2.0, attack_ms=10, release_ms=120, sr=self.sr)
        mix = stereo_widen(mix, amount=0.18)
        mix = lookahead_limiter(mix, threshold=self.master_limiter_threshold, makeup_db=self.master_limiter_makeup_db)
        mix *= db_to_linear(self.master_volume_db)
        mix = auto_gain_compensate(mix, target_rms=self.master_target_rms)
        mix = np.clip(mix, -0.9999, 0.9999)
        return mix.astype(np.float32)

    def save(self, filename):
        audio = self.render()
        int16 = (audio*32767).astype(np.int16)
        with wave.open(filename,"wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(self.sr)
            f.writeframes(int16.tobytes())

def load_wav(filename, target_sr=44100):
    with wave.open(filename,"rb") as f:
        sr = f.getframerate()
        nchan = f.getnchannels()
        frames = f.readframes(f.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)/32767.0
        data = data.reshape(-1, nchan)
        if sr != target_sr:
            idx = np.linspace(0, len(data)-1, int(len(data)*target_sr/sr))
            left = np.interp(idx, np.arange(len(data)), data[:,0])
            if nchan>1:
                right = np.interp(idx, np.arange(len(data)), data[:,1])
            else:
                right = left
            data = np.column_stack([left, right])
        return data

def sample_play(filename, dur_seconds, vol, sr, pitch_shift=1.0):
    data = load_wav(filename, sr)
    n = int(dur_seconds * sr)
    if pitch_shift != 1.0:
        idx = np.linspace(0, len(data)-1, int(len(data)/pitch_shift))
        data = np.column_stack([
            np.interp(idx, np.arange(len(data)), data[:,0]),
            np.interp(idx, np.arange(len(data)), data[:,1])
        ])
    if len(data) < n:
        pad = np.zeros((n-len(data),2), dtype=np.float32)
        data = np.vstack([data, pad])
    return data[:n] * vol
