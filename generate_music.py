import pickle
import numpy as np

from music21 import instrument, note, chord, stream

from tensorflow.keras.models import load_model


# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------

model = load_model("music_model.h5")


# -------------------------------
# LOAD NOTES
# -------------------------------

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)


pitchnames = sorted(set(notes))

note_to_int = {
    note: number
    for number, note in enumerate(pitchnames)
}

int_to_note = {
    number: note
    for number, note in enumerate(pitchnames)
}


# -------------------------------
# CREATE RANDOM START SEQUENCE
# -------------------------------

sequence_length = 100

start = np.random.randint(
    0,
    len(notes) - sequence_length - 1
)

pattern = [
    note_to_int[n]
    for n in notes[
        start:start + sequence_length
    ]
]


# -------------------------------
# GENERATE MUSIC
# -------------------------------

prediction_output = []

print("Generating music...")

for _ in range(300):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction_input = (
        prediction_input /
        float(len(pitchnames))
    )

    prediction = model.predict(
        prediction_input,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(index)

    pattern = pattern[1:]


# -------------------------------
# CONVERT NOTES TO MIDI
# -------------------------------

offset = 0

output_notes = []

for pattern in prediction_output:

    if (
        "." in pattern
        or pattern.isdigit()
    ):

        notes_in_chord = pattern.split(".")

        notes_in_chord = [
            int(n)
            for n in notes_in_chord
        ]

        new_chord = chord.Chord(
            notes_in_chord
        )

        new_chord.offset = offset

        output_notes.append(
            new_chord
        )

    else:

        new_note = note.Note(pattern)

        new_note.offset = offset

        new_note.storedInstrument = (
            instrument.Piano()
        )

        output_notes.append(
            new_note
        )

    offset += 0.5


# -------------------------------
# SAVE GENERATED MUSIC
# -------------------------------

midi_stream = stream.Stream(
    output_notes
)

midi_stream.write(
    "midi",
    fp="generated_music.mid"
)

print("\nMusic generated successfully!")
print("Saved as: generated_music.mid")
