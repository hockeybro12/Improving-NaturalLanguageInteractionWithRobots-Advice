# Improving-NaturalLanguageInteractionWithRobots-Advice
Code for the NAACL 2019 Paper: [Improving Natural Language Interaction with Robots Using Advice](https://aclweb.org/anthology/papers/N/N19/N19-1195/)

### Requirements

All code has been tested with Python 2.7 and Tensorflow 1.5. You can create an Anaconda environment to run the code like so: `conda create -n advice_env python=2.7 anaconda pip tensorflow-gpu=1.5`.


### Baseline Model

You can run the basline [Bisk et. al](https://www.aclweb.org/anthology/papers/N/N16/N16-1089/) models. They also released their code, but ours is slightly different (in Tensorflow instead of Julia for example), and achieves slightly better performance. To run the model, do: 

`python BaselineModel.py`

To run for target coordinate prediction and saving the model in a folder called savedModels, do:

`python BaselineModel.py --target=2 --model_save_path=savedModels/model.ckpt`

Other parameters can be found at the top of the file.

The results should match the ones found in our paper. Our end-to-end model with advice is a simple extension of this model and is described in the paper.


### Pre-trained models

Code coming soon.


### End-to-End Model with Advice

Code coming soon.
