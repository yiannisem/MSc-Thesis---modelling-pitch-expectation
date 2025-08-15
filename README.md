# MSc-Thesis---comparing-IDyOM-and-Music-Transformer-for-pitch-expectation-modelling


This repository contains code, data, analysis scripts and results for my MSc thesis, which compares the performance of the IDyOM variable-order Markov model and the Music Transformer in modelling melodic pitch expectation.

## Repository Structure

```
.
├── MusicTransformer-Pytorch-private
│    ├── probe_prediction_scripts/                                                           # Scripts for running the whole pipeline from initial IDyOM and Transformer predictions to forming the table used in statistical analysis
│    │    ├── extract_relevant_probs_from_schell_cpitch_idyom_output.py                       # Get the required diatonic pitches for modelling the Schellenberg experiment from IDyOM output (cpitch viewpoint)
│    │    ├── rpr_ftuned_schell_cpitch_add_human_data.py                                             # Add the human ratings to the merged cpitch IDyOM & Transformer probabilities and IC table
│    │    ├── rpr_ftuned_schell_add_relevant_transformer_probs_to_cpitch_idyom_table.py       # Add the Transformer Schellenberg probability predictions to the already existing table containing the cpitch IDyOM probabilities and IC values.
│    │    ├── rpr_ftuned_schell_add_transformer_ic_to_merged_cpitch_table.py                         # Add the Transformer Schellenberg ic predictions to the already existing table containing IDyOM probbilities & ICs and Transformer probabilities
│    │    ├── rpr_ftuned_schell_get_transformer_midi_47_to_84_and_normalise.py                # Take transformer probe probabilities for midi 0-127, keep 47-84 and normalise (this step is done primarily for easy inspection of the probabilities; normalisation isn't necessary)
│    │    └── schell_cpitch_add_idyom_ic.py                                                   # Add Information Content (IC) to the table containing IDyOM probabilities for Schellenberg (cpitch viewpoint)
│    ├── probe_prediction-results/                                                           # Contains outputs of the scripts in probe_prediction_scripts
│    ├── schellenberg_midi_stimuli_with_probe/                                                # Contains the Schellenberg stimuli in midi format, with an arbitrary appended final note to ensure compatibility with probe prediction scripts
│    ├── analyse_midi_schell_modified_for_probe_prediction.py                                # Outputs 4 csv files with transformer predictions; rpr_ftuned_schell_transformer_probe_probs.csv is used for the next step of the   pipeline(modified code from Kern repo's analysis_midi.py)
└── TrainedModels/
     ├── MTMaestro_with_RPR/                                                                # Output of training the Music Transformer on Maestro with rpr=True (contains results, weights, params, etc.)
     └── MTMaestroRPR_finetuned_on_MCCC/                                                    # Output of fine-tuning the Music Transformer on MCCC with rpr=True (contains results, weights, params, etc.)

```
