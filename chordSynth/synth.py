import numpy as np
import sounddevice as sd
from pynput import keyboard
import threading

# =========================
# USER CONFIG: BANK SCALES
# =========================
# Each bank should be a list of 7 MIDI note numbers (degrees 1..7).
# For now they are EMPTY ARRAYS for you to fill.
# Example for C major (if you want later):
# bank_scales['a'] = [60, 62, 64, 65, 67, 69, 71]

bank_scales = {
    'a': [],  # Bank A
    'w': [],  # Bank W
    's': [],  # Bank S
    'e': [],  # Bank E
    'd': [],  # Bank D
    'r': [],  # Bank R
    'f': [],  # Bank F
}

bank_scales['a'] = [58, 60, 62, 63, 65, 67, 69]      # Bb Ionian (Bb major)
bank_scales['w'] = [60, 62, 63, 65, 67, 69, 70]      # C Dorian
bank_scales['s'] = [62, 63, 65, 67, 69, 70, 72]      # D Phrygian
bank_scales['e'] = [63, 65, 67, 69, 70, 72, 74]      # Eb Lydian
bank_scales['d'] = [65, 67, 69, 70, 72, 74, 75]      # F Mixolydian
bank_scales['r'] = [67, 69, 70, 72, 74, 75, 77]      # G Aeolian (natural minor)
bank_scales['f'] = [69, 70, 72, 74, 75, 77, 79]      # A Locrian


# =========================
# AUDIO / SYNTH SETTINGS
# =========================

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

# Simple synth envelope
ATTACK_TIME = 0.01   # seconds
RELEASE_TIME = 0.05  # seconds

# Master volume
MASTER_GAIN = 0.4

# =========================
# GLOBAL STATE
# =========================

# current bank key: one of 'a','w','s','e','d','r','f' or None
current_bank = None

# octave offset in octaves (not semitones)
current_octave_offset = 0
MIN_OCTAVE_OFFSET = -2
MAX_OCTAVE_OFFSET = 2

# active notes: midi -> dict with phase, env_level, env_state
# env_state in {"attack", "sustain", "release"}
active_notes = {}

# which midi notes each key is holding (for release logic)
key_to_midis = {}

# lock to protect active_notes & key_to_midis
state_lock = threading.Lock()

# =========================
# KEY MAPPINGS
# =========================

# Chord-scale bank select keys
BANK_KEYS = {'a', 'w', 's', 'e', 'd', 'r', 'f'}

# Playable keys mapped to degree index 0..6
PLAYABLE_KEY_TO_DEGREE = {
    'j': 0,  # degree 1
    'i': 1,  # degree 2
    'k': 2,  # degree 3
    'o': 3,  # degree 4
    'l': 4,  # degree 5
    'p': 5,  # degree 6
    ';': 6,  # degree 7
}

OCTAVE_DOWN_KEY = 'v'
OCTAVE_UP_KEY = 'b'


# =========================
# UTILS
# =========================

def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def add_note_for_key(key_char, midi):
    """Register a note as active for a given key."""
    with state_lock:
        # Track which notes this key is responsible for
        key_to_midis.setdefault(key_char, [])
        if midi not in key_to_midis[key_char]:
            key_to_midis[key_char].append(midi)

        # Create or re-attack the note
        if midi not in active_notes:
            active_notes[midi] = {
                "phase": 0.0,
                "env_level": 0.0,
                "env_state": "attack",
            }
        else:
            # If already there, make sure it's not stuck in release
            if active_notes[midi]["env_state"] == "release":
                active_notes[midi]["env_state"] = "attack"


def release_notes_for_key(key_char):
    """Put all notes associated with this key_char into release state."""
    with state_lock:
        midi_list = key_to_midis.pop(key_char, [])
        for midi in midi_list:
            if midi in active_notes:
                active_notes[midi]["env_state"] = "release"


# =========================
# AUDIO CALLBACK
# =========================

def audio_callback(outdata, frames, time, status):
    if status:
        print("Audio callback status:", status)

    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE

    with state_lock:
        if not active_notes:
            outdata[:] = 0.0
            return

        # Output signal
        sig = np.zeros(frames, dtype=np.float32)

        # We will collect notes to remove after the block
        notes_to_remove = []

        for midi, note in active_notes.items():
            phase = note["phase"]
            env_level = note["env_level"]
            env_state = note["env_state"]

            freq = midi_to_freq(midi)
            omega = 2.0 * np.pi * freq

            # Generate phase array
            phase_array = phase + omega * t
            # Basic waveform: sine + a bit of saw-ish harmonic
            wave = np.sin(phase_array) + 0.3 * np.sin(2 * phase_array)

            # Envelope per sample
            if env_state == "attack":
                target = 1.0
                total_samples = max(1, int(ATTACK_TIME * SAMPLE_RATE))
                step = (target - env_level) / total_samples
                env_vec = env_level + step * np.arange(frames)
                # Clamp
                env_vec = np.clip(env_vec, 0.0, 1.0)
                env_level_end = env_vec[-1]
                if env_level_end >= 0.999:
                    env_state = "sustain"
                    env_level_end = 1.0
                    env_vec[:] = 1.0

            elif env_state == "sustain":
                env_vec = np.full(frames, env_level, dtype=np.float32)
                env_level_end = env_level  # should be 1.0

            elif env_state == "release":
                target = 0.0
                total_samples = max(1, int(RELEASE_TIME * SAMPLE_RATE))
                step = (target - env_level) / total_samples
                env_vec = env_level + step * np.arange(frames)
                env_vec = np.clip(env_vec, 0.0, 1.0)
                env_level_end = env_vec[-1]
                if env_level_end <= 0.001:
                    # fully released, schedule for removal
                    notes_to_remove.append(midi)

            else:
                # Unknown state, mute
                env_vec = np.zeros(frames, dtype=np.float32)
                env_level_end = 0.0

            # Apply envelope
            wave *= env_vec

            # Add to mix
            sig += wave.astype(np.float32)

            # Store back phase and envelope state
            note["phase"] = (phase_array[-1] + omega / SAMPLE_RATE) % (2.0 * np.pi)
            note["env_level"] = env_level_end
            note["env_state"] = env_state

        # Remove fully released notes
        for midi in notes_to_remove:
            active_notes.pop(midi, None)

    # Normalize by number of active notes and apply master gain
    with state_lock:
        num_active = max(1, len(active_notes))

    sig *= (MASTER_GAIN / num_active)

    # Write mono to out
    outdata[:, 0] = sig
    if outdata.shape[1] > 1:
        outdata[:, 1] = sig


# =========================
# KEYBOARD HANDLERS
# =========================

def on_press(key):
    global current_bank, current_octave_offset

    try:
        char = key.char
    except AttributeError:
        # Ignore special keys like Shift, Ctrl, etc.
        return

    char = char.lower()

    # If key is already pressed and holding notes, do nothing (avoid retrigger spam)
    with state_lock:
        if char in key_to_midis and key_to_midis[char]:
            return

    # Bank selection
    if char in BANK_KEYS:
        current_bank = char
        print(f"Bank '{char}' selected.")
        return

    # Octave controls
    if char == OCTAVE_DOWN_KEY:
        if current_octave_offset > MIN_OCTAVE_OFFSET:
            current_octave_offset -= 1
            print(f"Octave: {current_octave_offset:+d}")
        return

    if char == OCTAVE_UP_KEY:
        if current_octave_offset < MAX_OCTAVE_OFFSET:
            current_octave_offset += 1
            print(f"Octave: {current_octave_offset:+d}")
        return

    # Playable keys
    if char in PLAYABLE_KEY_TO_DEGREE:
        degree_index = PLAYABLE_KEY_TO_DEGREE[char]

        if current_bank is None:
            print("No bank selected yet. Press one of: a w s e d r f")
            return

        scale = bank_scales.get(current_bank, [])
        if len(scale) != 7:
            print(f"Bank '{current_bank}' is not populated with 7 notes yet.")
            return

        base_midi = scale[degree_index]
        midi = base_midi + current_octave_offset * 12
        add_note_for_key(char, midi)
        return


def on_release(key):
    try:
        char = key.char
    except AttributeError:
        return

    char = char.lower()

    if char in PLAYABLE_KEY_TO_DEGREE:
        release_notes_for_key(char)


# =========================
# MAIN
# =========================

def main():
    print("Chord-Scale Bank Synth")
    print("======================")
    print("Bank select keys: a w s e d r f")
    print("  Each bank has an array of 7 MIDI notes (fill in bank_scales at top).")
    print("")
    print("Playable keys (degrees 1..7 of current bank):")
    print("  j i k o l p ;  -> degrees 1 2 3 4 5 6 7")
    print("")
    print("Octave down: v")
    print("Octave up  : b")
    print("")
    print("Usage:")
    print("  1) Fill bank_scales['a'], ['w'], ... with 7 MIDI notes each.")
    print("  2) Run this script.")
    print("  3) Press a/w/s/e/d/r/f to select a bank.")
    print("  4) Play with j i k o l p ; and use v/b for octaves.")
    print("Ctrl+C in terminal to quit.\n")

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype='float32',
        callback=audio_callback
    ):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nExiting...")


if __name__ == "__main__":
    main()

