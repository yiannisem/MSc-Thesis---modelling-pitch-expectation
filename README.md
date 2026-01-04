# MSc Thesis - Modelling Pitch Expectation in Melody: A Comparison of Variable-Order Markov and Transformer-Based Approaches

Abstract—Musical expectation is widely regarded as a key mechanism underlying the emotional impact of music. Past work has modelled expectation with rules-based approaches, such as Narmour’s Implication–Realisation model, or statistical learning methods like the Information Dynamics of Music (IDyOM) which have shown stronger explanatory power. More recently, deep
neural models such as the Music Transformer have demonstrated success in symbolic music generation, raising questions about their potential for modelling expectation. This study directly compared IDyOM and the Music Transformer in predicting probe-tone ratings from experiments by Cuddy and Lunney (1995) and Schellenberg (1996). IDyOM was tested with basic and
derived viewpoint configurations, while Transformer probabilities were obtained from a model trained on MAESTRO and finetuned on the Monophonic Corpus of Complete Compositions. Regression analyses showed that the Transformer significantly outperformed IDyOM for short two-note contexts, whereas both models performed comparably for longer melodic contexts. These
findings suggest that while the Music Transformer offers an advantage when higher-level structural information is absent, IDyOM remains a competitive model for longer melodic sequences, and their comparison helps render the Transformer’s predictions more interpretable.

## Paper and Poster

This repository contains the paper with the full details of the research, as well as a poster which I presented at DMRN+20 in London in December, 2025, for a more brief synopsis of my work.

## Video Presentation

A video presentation summarising my research paper can be found here: https://youtu.be/ii7TDOAyz4M.

## Instructions for running

### Obtaining IDyOM predictions

- Obtain .dat file with IDyOM results for the desired stimulus using the wanted IDyOM configurations and viewpoints. Guidance on this can be found at https://github.com/mtpearce/idyom/wiki, which provides relevant documentation and a tutorial.

### Training and fine-tuning the Music Transformer

- The MAESTRO and MCCC datasets need to be downloaded to facilitate training and fine-tuning. The MAESTRO dataset is available here: https://magenta.withgoogle.com/datasets/maestro (use MAESTRO v2.0.0). The MCCC is available here: https://osf.io/dg7ms/. Direct the scripts in the following steps to the appropriate paths for these two corpuses.
- Pre-process the MAESTRO dataset using preprocess_midi.py. Includes tokenisation and splitting into training, validation and test sets.
- Train the Music Transformer on MAESTRO using train.py (training parameters detailed in paper)
- Pre-process the MCCC corpus using preprocess_midi_MCCC.py
- Likewise, fine-tune on MCCC using train.py

### Obtaining Music Transformer predictions

- Preprocess stimuli to get from kern to midi format using kern_to_midi_convert.py (or simply access converted files from cudlun_midi_stimuli_with_probe and schellenberg_midi_stimuli_with_probe folders)
- Use analyse_midi_schell_modified_for_probe_prediction.py to get the Music Transformer's predictions on the chosen stimuli (chosen input path if wanting to predict on a stimuli dataset other than Schellenberg)

### Manipulating predictions to prepare for statistical analysis

Note: the scripts mentioned in this section are located in the probe_prediction_scripts folder.

- To get the relevant probabilities from IDyOM's output in the form required by this pipeline, use extract_relevant_probs_from_schell_cpitch_idyom_output.py if working with Schellenberg data or extract_cudlun_probes_midi_54_to_78_from_idyom_output.py if working with Cuddy and Lunney (the differences are due to the former requiring diatonic pitches within an octave and the latter all chromatic pitches within an octave)
- Use cudlun_add_idyom_ic.py on the csv obtained in the previous step to add the Information Content (IC) to the table.

Depending on if the choice of stimuli dataset, use 1 or 2

1 - Schellenberg: 
- Use rpr_ftuned_schell_get_transformer_midi_47_to_84_and_normalise.py on the Transformer prediction output for the probe tones, and the output of this is inputted into rpr_ftuned_schell_add_relevant_transformer_probs_to_cpitch_idyom_table.py to merge with the aforementioned IDyOM table with the wanted probabilities.
- Use rpr_ftuned_schell_add_transformer_ic_to_merged_table.py to add the IC values to the output table of the previous step.
- Use rpr_ftuned_schell_cpitch_add_human_data.py to add the human ratings to the output table from the previous step.

2 - Cuddy and Lunney:
- Use rpr_ftuned_cudlun_add_relevant_transformer_probs_to_idyom_table.py to add the required probabilities from the Transformer's prediction output to the IDyOM table with the wanted probabilities.
- Use rpr_ftuned_cudlun_add_transformer_ic_to_merged_cpitch_table.py to add the IC values to the output table of the previous step.
- Use rpr_ftuned_cudlun_add_human_data.py to add the human ratings to the output table from the previous step.

### Statistical analysis

- From the folder named statistical_analysis_scripts, use schell_all_regressions_analyses.py to do the relevant regression analyses on the table obtained for the Schellenberg data, or cudlun_all_regressions_analyses.py for the Cuddy and Lunney data.
- Then compare correlations following the website cited in the research paper (or an alternative way)


