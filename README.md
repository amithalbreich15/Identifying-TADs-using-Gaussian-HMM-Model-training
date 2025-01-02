**Identifying Topologically Associating Domains (TADs) Using a Gaussian Hidden Markov Model (HMM)**

**Overview**

This project presents a computational approach to identify and characterize Topologically Associating Domains (TADs) within chromosomal structures using a Gaussian Hidden Markov Model (HMM). The study leverages high-resolution chromatin interaction data from Hi-C experiments to reveal the 3D genome architecture. Our HMM-based methodology effectively models chromatin interaction patterns and predicts TAD boundaries with high accuracy.

**Key Features**

Data Input: Hi-C datasets processed and normalized to account for biases such as GC content and fragment lengths.
Scoring Metrics: Includes the Directionality Index (DI) and Insulation Scores to quantify chromatin interactions.

HMM Design:
Three-state architecture: Forward (F), Backward (B), and Middle (M) states to model genomic interactions.
Gaussian probability distributions for state emissions, ensuring flexibility and precision in pattern recognition.

Algorithms:
Baum-Welch Algorithm: Used for model training and parameter estimation.
Viterbi Algorithm: Applied for sequence prediction to identify TAD boundaries.
Comparative Analysis: Evaluates TAD conservation across different tissue types and cell lines.

Results
Successfully identified TAD boundaries with high consistency across datasets.
Highlighted the conserved nature of TAD structures across tissue types, underscoring their role in chromosomal organization.
Demonstrated the potential of HMMs in genomic data analysis.

Applications
Genomic Research: Enhances understanding of chromatin architecture and its role in gene regulation.
Disease Studies: Provides insights into the disruption of TAD boundaries linked to genetic disorders.

Computational Biology: Demonstrates the utility of HMMs for analyzing complex genomic data.

**How to Use**

Input: Prepare normalized Hi-C data in matrix form.
Run: Train the Gaussian HMM on DI scores using the provided scripts.
Output: Predicted TAD boundaries and state transition maps.
Visualization: Use the included plotting tools for boundary visualization and comparison across datasets.
Future Enhancements

Integration of additional genomic features (e.g., ChIP-seq, ATAC-seq) for improved boundary prediction.
Evaluation of different bin and window sizes for higher-resolution TAD identification.
Extension to multi-species analysis to uncover evolutionary patterns.

**Authors**
Amit Halbreich, David Zuravin, Noam Delbari, Yaniv Pasternak, Omer Mushlion, Elisheva Morgenstern
