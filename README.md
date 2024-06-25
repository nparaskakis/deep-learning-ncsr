# deep-learning-ncsr

Repository for the project of the course "Deep Learning" in the MSc in Data Science of the NCSR "Demokritos".

**Authors:**
- **Nikolaos Paraskakis / I.D.: 2321**
- **Dimitrios Tselentis / I.D.: 2325**

June 2024

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

We also experiment with transfer learning from models trained on dataset UrbanSound8K, which is a dataset containing various urban sound events.

The citation of the dataset UrbanSound8K is the following:

J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

The dataset itself was downloaded from https://urbansounddataset.weebly.com/urbansound8k.html.

# Main folders description

Below, you will find a brief description for the main folders of this repository:
- **baseline**: code to create melspectrograms and audiofeatures of FSC22, and code to train models on FSC22 original dataset
- **augmented**: code to augment FSC22, code to create melspectrograms and audiofeatures of augmented FSC22, and code to train models on augmented FSC22 dataset
- **transfer_learning_v1**: code to create melspectrograms and audiofeatures of FSD50K, code to train models on FSD50K original dataset
- **transfer_learning_v2**: code to train (pretrained) models on FSC22 original dataset (transfer learning)
- **transfer_learning_v3**: code to create melspectrograms and audiofeatures of UrbanSound8K, and code to train models on UrbanSound8K original dataset
- **training_notebooks**: contains the notebooks that were used to train all the models in google colab.

In the root of this repository you will find also the following files:
- **report.ipynb**: deliverable notebook demonstrating the things mentioned in the presentation pdf file
- **requiremnets.txt**: requirements txt file for pip install
- **deep_learning_presentation.pdf**: the pdf file of the presentation of this project
- **kaggle.json**: this is a json file containing the credentials needed to download the directory with all the data used in this project. The data are uploaded in a private kaggle repository.