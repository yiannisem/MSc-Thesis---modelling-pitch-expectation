# MSc-Thesis---comparing-IDyOM-and-Music-Transformer-for-pitch-expectation-modelling


This repository contains code, data, analysis scripts and results for my MSc thesis, which compares the performance of the IDyOM variable-order Markov model and the Music Transformer in modelling melodic pitch expectation.

## Repository Structure

```
.
├── probe_prediction_scripts/                                                           # Scripts for running the whole pipeline from initial IDyOM and Transformer predictions to forming the table used in statistical analysis
│   ├── extract_relevant_probs_from_schell_cpitch_idyom_output.py                       # Get the required diatonic pitches for modelling the Schellenberg experiment from IDyOM output (cpitch viewpoint)
│   └── schell_cpitch_add_idyom_ic.py                                                   # Add Information Content (IC) to the table containing IDyOM probabilities for Schellenberg (cpitch viewpoint)
```
