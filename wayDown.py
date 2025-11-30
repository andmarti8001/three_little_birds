import sys
import numpy as np
import sounddevice as sd
from pynput import keyboard

# ---------------- NOTE FREQUENCIES ----------------
NOTE_FREQUENCIES = {
    "A0": 27.50, "A#0": 29.14, "B0": 30.87,
    "C1": 32.70, "C#1": 34.65, "D1": 36.71, "D#1": 38.89, "E1": 41.20, "F1": 43.65, "F#1": 46.25, "G1": 49.00, "G#1": 51.91,
    "A1": 55.00, "A#1": 58.27, "B1": 61.74,
    "C2": 65.41, "C#2": 69.30, "D2": 73.42, "D#2": 77.78, "E2": 82.41, "F2": 87.31, "F#2": 92.50, "G2": 98.00, "G#2": 103.83,
    "A2": 110.00, "A#2": 116.54, "B2": 123.47,
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65,
    "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30,
    "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99, "G#5": 830.61,
    "A5": 880.00, "A#5": 932.33, "B5": 987.77,
    "C6": 1046.50, "C#6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22,
    "A6": 1760.00, "A#6": 1864.66, "B6": 1975.53,
    "C7": 2093.00, "C#7": 2217.46, "D7": 2349.32, "D#7": 2489.02, "E7": 2637.02, "F7": 2793.83, "F#7": 2959.96, "G7": 3135.96, "G#7": 3322.44,
    "A7": 3520.00, "A#7": 3729.31, "B7": 3951.07,
    "C8": 4186.01
}

# ---------------- YOUR ARRANGEMENTS (FILL THESE) ----------------
# You can put note names like "G3", "C4", "E5" etc.
NOTES_SNOWMAN = [
    ['A#3', 'C#4', 'F4'],
    ['A#3', 'C#4', 'F#3'],
    ['A#3', 'D#3', 'F#3'],
    ['A#3', 'C#4', 'F4'],
    ['A#3', 'C#4', 'F#3'],
    ['A#3', 'D#3', 'F#3'],
]
NOTES_ELF = [
    'A#4',
    'D#5',
    'D#5',
    'C#5',
    'A#4',
    'G#4',
    'A#4',
    'A#4',
    'A#4',
    'A#4',
    'A#4',
    'A#4',
    'D#5',
    'D#5',
    'C#5',
    'A#4',
    'G#4',
    'A#4',
    'A#4',
    'A#4',
    'A#4',
    'A#4',
]


NOTES_SANTA = [
    'G4',
    'C5',
    'B4',
    'C5',
    'D5',
    'E5',
    'E5',
    'D5',
    'B4',
    'A4',
    'G4',
    'E5',
    'D5',
    'E5',
    'F5',
    'G5',
    'C5',
    'C5',
    'C5',
    'A5',
    'A5',
    'G5',
    'E5',
    'D5',
    'C5',
    'C5',
    'D5',
    'C5',
]

NOTES_REINDEER = ["A4"]

SAMPLE_RATE = 44100

# Each instrument's current freq (or None if idle)
active_freqs = {
    "snowman": None,
    "elf": None,
    "santa": None,
    "reindeer": None,
}

phases = {
    "snowman": [],
    "elf": [],
    "santa": [],
    "reindeer": [],
}


# Instruments share the same logic of cycling through arrays
class Instrument:
    def __init__(self, notes, name):
        self.notes = notes
        self.name = name
        self.index = 0
        self.is_pressed = False

Snowman = Instrument(NOTES_SNOWMAN, "snowman")
Elf     = Instrument(NOTES_ELF,     "elf")
Santa   = Instrument(NOTES_SANTA,   "santa")
Reindeer= Instrument(NOTES_REINDEER,"reindeer")

# Map physical keys → instruments
KEYMAP = {
    'a': Snowman,
    'f': Elf,
    'j': Santa,
    ';': Reindeer,
}

# ---------------- AUDIO CALLBACK (TRUE POLYPHONY) ----------------
def audio_callback(outdata, frames, time, status):
    if status:
        print("Audio status:", status)

    t = np.arange(frames, dtype=np.float32)
    mix = np.zeros(frames, dtype=np.float32)

    for name, freqs in active_freqs.items():
        if freqs is None:
            continue

        # freqs is a list of oscillator frequencies
        phases_list = phases[name]
        chord_mix = np.zeros(frames, dtype=np.float32)

        for i, freq in enumerate(freqs):
            phase = phases_list[i]
            phase_inc = (2 * np.pi * freq) / SAMPLE_RATE

            ph = phase + phase_inc * t
            tone = 0.22 * np.sin(ph, dtype=np.float32)

            phases_list[i] = ph[-1] % (2 * np.pi)
            chord_mix += tone

        # Add this instrument's sum to the main mix
        mix += chord_mix

    # Avoid clipping
    mix = np.clip(mix, -1.0, 1.0)

    # stereo
    outdata[:] = np.column_stack((mix, mix))

# ---------------- START / STOP NOTE ----------------
def start_note(inst: Instrument):
    if inst.is_pressed or len(inst.notes) == 0:
        return

    inst.is_pressed = True

    # Get next entry
    note_data = inst.notes[inst.index]

    # Convert single note -> list
    if isinstance(note_data, str):
        freqs = [NOTE_FREQUENCIES[note_data]]
    elif isinstance(note_data, list):
        freqs = [NOTE_FREQUENCIES[n] for n in note_data]
    else:
        print("Invalid note data:", note_data)
        return

    # Reset phase accumulators (one per oscillator)
    phases[inst.name] = [0.0 for _ in freqs]

    # Store the active frequencies for this instrument
    active_freqs[inst.name] = freqs

    print(f"{inst.name} ON → {note_data}")

def stop_note(inst: Instrument):
    if not inst.is_pressed:
        return

    inst.is_pressed = False

    # Turn off all oscillators on that instrument
    active_freqs[inst.name] = None
    phases[inst.name] = []

    # Advance note index
    inst.index = (inst.index + 1) % len(inst.notes)
    print(f"{inst.name} OFF → next index {inst.index}")

# ---------------- KEYBOARD LISTENER ----------------
def on_press(key):
    try:
        ch = key.char
    except AttributeError:
        # special key (ESC, etc.)
        if key == keyboard.Key.esc:
            print("ESC pressed, exiting…")
            stream.stop()
            stream.close()
            sys.exit(0)
        return

    if ch in KEYMAP:
        inst = KEYMAP[ch]
        start_note(inst)

def on_release(key):
    try:
        ch = key.char
    except AttributeError:
        return

    if ch in KEYMAP:
        inst = KEYMAP[ch]
        stop_note(inst)

# ---------------- MAIN ----------------
def main():
    print("🎄 True Polyphonic Christmas Quartet (no GUI)")
    print("Controls:")
    print("  A = Snowman (bass)")
    print("  F = Elf (high)")
    print("  J = Santa (harmony)")
    print("  ; = Reindeer (melody)")
    print("Hold any combo (A/F/J/;) to play multiple at once.")
    print("ESC = Quit\n")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    # Start the audio stream
    try:
        stream = sd.OutputStream(
            channels=2,
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            blocksize=512
        )
        stream.start()
        print("Audio stream started.")
    except Exception as e:
        print("Audio startup error:", e)
        sys.exit(1)

    main()
