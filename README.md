# RIS CNN: Training Data and Model Training Code

This repository contains training and test data, together with the code to train
and evaluate the convolutional neural network model presented in our papers.
The CNN predicts total tangential electric and magnetic fields of reconfigurable
intelligent surfaces (RISs).

This work was carried out by Yuanzhi Liu and Costas D. Sarris at the
Edward S. Rogers Sr. Department of Electrical and Computer Engineering,
University of Toronto, Toronto, ON, Canada.

## Background

RISs control wave propagation by selecting the state of each unit cell. Mutual
coupling makes each cell's response depend on the surrounding configuration,
which periodic unit-cell models cannot fully capture. Full-wave simulation can
account for these effects, but repeating it for many configurations is costly.
Our [CNN approach](https://doi.org/10.1109/TAP.2025.3532123) learns these coupling
effects to predict near-zone fields and enable efficient scattered-field
computation. The [codebook optimization study](https://doi.org/10.1109/LAWP.2025.3585743)
uses this model to design scattering patterns, including suppressing sidelobes
and quantization lobes in binary RISs.

## Contents

- [CNN code and instructions](cnn/README.md): model architecture, training,
  sample selection, and test evaluation.
- [Training and test data](cnn/data/README.md): 24,000 training samples
  (2,000 per size/incident-angle group) and 550 test samples.
- [BibTeX citations](CITATION.bib): references for both papers.

## Getting started

Use Python 3.9 through 3.12. From the repository root:

```bash
python -m pip install -r cnn/requirements.txt
python cnn/train.py
python cnn/evaluate.py
```

All required training and test data are included. See [cnn/README.md](cnn/README.md)
for model and training configuration, dataset checks, and evaluation commands.

## Citation

If you use this code or the training/test data in your research, please cite
either or both of the following papers, as relevant to your work:

1. Yuanzhi Liu and Costas D. Sarris, "A Convolutional Neural Network Based Method
   for Accurate Computation of Scattered Fields From Reconfigurable Intelligent
   Surfaces," IEEE Transactions on Antennas and Propagation, vol. 73, no. 4,
   pp. 2615-2627, April 2025. [DOI: 10.1109/TAP.2025.3532123](https://doi.org/10.1109/TAP.2025.3532123).
2. Yuanzhi Liu and Costas D. Sarris, "Optimization of Reconfigurable Intelligent
   Surface Codebooks Using a Mutual Coupling Aware CNN Model of Scattered Fields,"
   IEEE Antennas and Wireless Propagation Letters, vol. 24, no. 9,
   pp. 3188-3192, September 2025. [DOI: 10.1109/LAWP.2025.3585743](https://doi.org/10.1109/LAWP.2025.3585743).

BibTeX entries are provided in [CITATION.bib](CITATION.bib).

## Contact

Yuanzhi Liu: [yuanzhi.liu@alumni.utoronto.ca](mailto:yuanzhi.liu@alumni.utoronto.ca)

## License

- Code and documentation outside `cnn/data/`: [MIT License](LICENSE).
- Dataset and accompanying files under `cnn/data/`: [Creative Commons Attribution
  4.0 International (CC BY 4.0)](cnn/data/LICENSE).

When sharing the data, credit Yuanzhi Liu and Costas D. Sarris, provide a link to
CC BY 4.0, and indicate any changes. The full license text is included in the
repository.
