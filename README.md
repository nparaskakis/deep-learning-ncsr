# deep-learning-ncsr

Repository for the project of the course "Deep Learning" in the MSc in Data Science of the NCSR "Demokritos".

# General description

In this project, we try to make a classifier for the dataset FSC22, which is a dataset containing forest sounds. We experiment with various architectures:

- Convolutional neural networks, where as feature representations for our audio files we use melspectrograms.
- Fully connected neural networks, where as feature representations for our audio files we use audio features.

The citation of the dataset FSC22 is the following:

Bandara, M.; Jayasundara, R.; Ariyarathne, I.; Meedeniya, D.; Perera, C. Forest Sound Classification Dataset: FSC22. Sensors 2023, 23, 2032. https://doi.org/10.3390/s23042032

The dataset itself was downloaded from https://www.kaggle.com/datasets/irmiot22/fsc22-dataset.

We also experiment with transfer learning from models trained on dataset FSD50K, which is a dataset containing various sound events.

The citation of the dataset FSC22 is the following:

E. Fonseca, X. Favory, J. Pons, F. Font and X. Serra, "FSD50K: An Open Dataset of Human-Labeled Sound Events," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 829-852, 2022, doi: 10.1109/TASLP.2021.3133208.

The dataset itself was downloaded from https://annotator.freesound.org/fsd/release/FSD50K/.

# Main folders description


<!-- Run as:

python main.py --architecture cnn1 --features melspectrograms

or

python main.py --architecture cnn1 --features audiofeatures

You can replace cnn1 with cnn2 also -->