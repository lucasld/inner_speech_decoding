# Inner Speech Decoding

The target of this project is to classify EEG data recordings with a CNN architecture. As the data provided is insufficient for training up to a high accuracy, we focus on methods to increace learning with few datapoints. The methods applied include PCA, transfer-training and for testing the k-cross validation.

The Project was conducted as a final project in the course "Implementing Artificial Neural Networks with Tensorflow" in 2021/22 at the University of Osnabrueck by Fabienne Kock, Lucas Liess-Duquesne and Sascha Mühlinghaus. 

For further information please refer to our [report](https://github.com/lucasld/inner_speech_decoding/blob/main/Inner_Speech_Project_Report.pdf).

### Dataset
We will use a [dataset](https://openneuro.org/datasets/ds003626) published by Nicolas Nieto, Victoria Peterson, Hugo Rufiner, Juan Kamienkowski, Ruben Spies.
A detailed description can be found [here](https://www.biorxiv.org/content/10.1101/2021.04.19.440473v1.full).

### How to get started yourself:
#### Clone this Repository
`git clone https://github.com/lucasld/inner_speech_decoding.git`
#### Download the Dataset
Use this command to download the dataset into the project folder:
`aws s3 sync --no-sign-request s3://openneuro.org/ds003626 dataset/`
or use [these](https://openneuro.org/datasets/ds003626/versions/2.1.0/download) instructions.
#### Create an Environment
The provided 'environment.yml' includes all the required packages and libraries.
