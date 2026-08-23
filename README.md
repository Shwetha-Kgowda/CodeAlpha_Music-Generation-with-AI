# CodeAlpha_Music-Generation-with-AI

Build a AI music generator that:

Collects MIDI files
Uses music21 to extract notes and chords
Converts music into sequences
Trains an LSTM neural network
Generates new note sequences
Saves the generated music as a .mid file

INSTALL LIBRARIES:
tensorflow
music21
numpy

Step 1: Add MIDI training data

Create a folder named:

midi_songs

Put classical MIDI files inside it, for example:

midi_songs/
├── song1.mid
├── song2.mid
├── song3.mid
└── song4.mid

You should ideally use 50–100+ MIDI files for better results.


How to run
1. Open VS Code terminal
cd Music-Generation-AI
2. Train the AI model
python train_music_model.py

After training, these files will be created:

notes.pkl
music_model.h5
3. Generate music
python generate_music.py

Output:

generated_music.mid

You can open the generated MIDI file using VLC, MuseScore, GarageBand, or any MIDI player.
