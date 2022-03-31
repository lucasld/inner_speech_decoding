# Inner Speech Decoding

The target of this project is to classify EEG data recordings with a CNN architecture. As the data provided is insufficient for training up to a high accuracy, we focus on methods to increace learning with few datapoints. The methods applied include PCA, transfer-training and for testing the k-cross validation.

The Project was conducted as a final project in the course "Implementing Artificial Neural Networks with Tensorflow" in 2021/22 at the University of Osnabrueck by Fabienne Kock, Lucas Liess-Duquesne and Sascha Mühlinghaus. 

For further information please refer to our [paper]().

### Dataset
We will use a [dataset](https://openneuro.org/datasets/ds003626) published by Nicolas Nieto, Victoria Peterson, Hugo Rufiner, Juan Kamienkowski, Ruben Spies.
A detailed description can be found [here](https://www.biorxiv.org/content/10.1101/2021.04.19.440473v1.full).

### How to get started yourself:
#### Clone this Repository
`git clone https://github.com/lucasld/inner_speech_decoding.git`
#### Download the Dataset
`aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ds003626-download/`
or use [these](https://openneuro.org/datasets/ds003626/versions/2.1.0/download) instructions.
(the downloaded folder should be placed into the project folder and be renamed to 'dataset')
#### Create an Environment
The provided 'environment.yml' includes all the required packages and libraries.

### Results 

The best average accuracy we achieved for classifying the data correctly was XY% with 25% being a random classification. 
The best single subject classification was XY% with 25% being a random classification. 

For more detailed results please refer to our [paper]().

### Naming Conventions
