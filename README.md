# Inner Speech Decoding

Decoding EEG-data of Inner Speech to explore the potential of Brain Computer Interfaces.

### Dataset
We will use a [dataset](https://openneuro.org/datasets/ds003626) published by Nicolas Nieto, Victoria Peterson, Hugo Rufiner, Juan Kamienkowski, Ruben Spies.
A detailed description can be found [here](https://www.biorxiv.org/content/10.1101/2021.04.19.440473v1.full).

### How to get started:
#### Clone this repository
`git clone https://github.com/lucasld/inner_speech_decoding.git`
#### Download the dataset
`aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ds003626-download/ && mv ds003626-download dataset`
or use [these](https://openneuro.org/datasets/ds003626/versions/2.1.0/download) instructions.
(the downloaded folder should be placed into the project folder and be renamed to 'dataset')

[latex](https://sharelatex.gwdg.de/project/6214980ed0e70d008e1712b8/invite/token/0ac8314082fe570c96281b0abaf636062d64c8e1d5bf696d?project_name=Inner%20Speech%20Dokumentation&user_first_name=Fabienne)
