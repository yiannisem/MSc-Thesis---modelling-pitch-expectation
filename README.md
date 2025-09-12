# MSc-Thesis---comparing-IDyOM-and-Music-Transformer-for-pitch-expectation-modelling

This repository contains the research paper and code for my MSc thesis, which compares the performance of the IDyOM variable-order Markov model and the Music Transformer in modelling melodic pitch expectation. It is an adaptations of the work of Kern et al. (2022) - repository available through https://data.ru.nl/collections/di/dccn/DSC_3018045.02_116.

## Abstract

Abstract—Musical expectation is widely regarded as a key
mechanism underlying the emotional impact of music. Past work
has modelled expectation with rules-based approaches, such as
Narmour’s Implication–Realisation model, or statistical learning
methods like the Information Dynamics of Music (IDyOM) which
have shown stronger explanatory power. More recently, deep
neural models such as the Music Transformer have demonstrated
success in symbolic music generation, raising questions about
their potential for modelling expectation. This study directly
compared IDyOM and the Music Transformer in predicting
probe-tone ratings from experiments by Cuddy and Lunney
(1995) and Schellenberg (1996). IDyOM was tested with basic and
derived viewpoint configurations, while Transformer probabilities
were obtained from a model trained on MAESTRO and finetuned
on the Monophonic Corpus of Complete Compositions.
Regression analyses showed that the Transformer significantly
outperformed IDyOM for short two-note contexts, whereas both
models performed comparably for longer melodic contexts. These
findings suggest that while the Music Transformer offers an
advantage when higher-level structural information is absent,
IDyOM remains a competitive model for longer melodic sequences,
and their comparison helps render the Transformer’s
predictions more interpretable.

## Research paper

[MSc_Thesis_research_paper_12-09.pdf](https://github.com/user-attachments/files/22298095/MSc_Thesis_research_paper_12-09.pdf)
