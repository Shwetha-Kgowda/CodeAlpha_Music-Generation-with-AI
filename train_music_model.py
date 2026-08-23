import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# -------------------------------
# STEP 1: READ MIDI FILES
# -------------------------------

def get_notes():
    notes = []

    midi_folder = "midi_songs"

    for file in os.listdir(midi_folder):

        if file.endswith(".mid") or file.endswith(".midi"):

            file_path = os.path.join(midi_folder, file)

            print("Processing:", file)

            try:
                midi = converter.parse(file_path)

                parts = instrument.partitionByInstrument(midi)

                if parts:
                    notes_to_parse = parts.parts[0].recurse()
                else:
                    notes_to_parse = midi.flat.notes

                for element in notes_to_parse:

                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))

                    elif isinstance(element, chord.Chord):
                        notes.append(
                            ".".join(str(n) for n in element.normalOrder)
                        )

            except Exception as e:
                print("Error reading file:", file, e)

    return notes


# -------------------------------
# STEP 2: CREATE TRAINING SEQUENCES
# -------------------------------

notes = get_notes()

print("\nTotal notes:", len(notes))

if len(notes) == 0:
    print("No MIDI notes found!")
    print("Please add MIDI files inside the midi_songs folder.")
    exit()


# Save notes for music generation later
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)


# Get unique notes
pitchnames = sorted(set(notes))

note_to_int = {
    note: number
    for number, note in enumerate(pitchnames)
}

sequence_length = 100

network_input = []
network_output = []

for i in range(
    0,
    len(notes) - sequence_length
):

    sequence_in = notes[
        i:i + sequence_length
    ]

    sequence_out = notes[
        i + sequence_length
    ]

    network_input.append(
        [note_to_int[char] for char in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )


n_patterns = len(network_input)

print("Training patterns:", n_patterns)
print("Unique notes:", len(pitchnames))


# -------------------------------
# STEP 3: PREPARE DATA
# -------------------------------

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(len(pitchnames))

network_output = to_categorical(
    network_output,
    num_classes=len(pitchnames)
)


# -------------------------------
# STEP 4: BUILD LSTM MODEL
# -------------------------------

model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(
            network_input.shape[1],
            network_input.shape[2]
        ),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        256,
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dropout(0.3))

model.add(
    Dense(256, activation="relu")
)

model.add(
    Dense(
        len(pitchnames),
        activation="softmax"
    )
)


# -------------------------------
# STEP 5: COMPILE MODEL
# -------------------------------

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)


# -------------------------------
# STEP 6: TRAIN MODEL
# -------------------------------

print("\nTraining started...\n")

model.fit(
    network_input,
    network_output,
    epochs=50,
    batch_size=64
)


# Save trained model
model.save("music_model.h5")

print("\nModel saved successfully!")
print("File: music_model.h5")
