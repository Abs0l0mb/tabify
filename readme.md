# MIDI to GP3 Tab Conversion

This project aims to create a machine learning model that converts MIDI files into GP3 guitar tab files with the most efficient fingering possible. The project utilizes sequence-to-sequence (Seq2Seq) models to achieve this goal.

## Model inputs and outputs

As input, you can pass any midi file, with a range corresponding to the range of a guitar in standard tuning (E2-E6).

## Dataset

The model has been trained on hundred of thousands of separate tracks in gp3 format.

To create the dataset, you can scrape tabs on specialized websites ; once you have all your gp3 files, the dataset_creation.py file takes as input the folder containing those files, and extract each track into a separate gp3 file, to a folder of your choice. 

For label creation, you can use the gp3_to_midi.py to convert a gp3 tab to midi file.

Once you have your datas and labels, you are free to train your own model !

The model proposed in this repo will probably be an LSTM to capture long term dependencies between notes.

## Roadmap

(done) Fix voice (0 only) + dédoublonnage

Export JSONL par morceau (events triés, accords triés)

Baseline DP/Viterbi + métriques

Définir score jouabilité (même simple)

Ensuite seulement : modèle ML/NN + reranking jouabilité