# MIDI to GP3 Tab Conversion

This project aims to create a machine learning model that converts MIDI files into GP3 guitar tab files with the most efficient fingering possible. The project utilizes sequence-to-sequence (Seq2Seq) models to achieve this goal.

## Model inputs and outputs

Inputs from midi file & tab

Note n (int) of length l (int) played at time t (int) is played on string s (int or letter ?) at BPM b (int) considering the previous notes n-x of length l-x

inference will be (n-3, l-3, n-2, l-2, n-1, l-1, n, l, b) => s
