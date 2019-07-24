# Improving-NaturalLanguageInteractionWithRobots-Advice
Code for the NAACL 2019 Paper: [Improving Natural Language Interaction with Robots Using Advice](https://aclweb.org/anthology/papers/N/N19/N19-1195/)

All files are commented. Please read the paper and the code files. If you have any questions, please create an Issue.

### Requirements

All code has been tested with Python 2.7 and Tensorflow 1.5. You can create an Anaconda environment to run the code like so: `conda create -n advice_env python=2.7 anaconda pip tensorflow-gpu=1.5`. Please use the data from [Bisk et. al](https://www.aclweb.org/anthology/papers/N/N16/N16-1089/).


### Baseline Model

You can run the basline [Bisk et. al](https://www.aclweb.org/anthology/papers/N/N16/N16-1089/) models. They also released their code, but ours is slightly different (in Tensorflow instead of Julia for example). Our code is based on theirs, though. To run the model, do: 

`python BaselineModel.py`

To run for target coordinate prediction and saving the model in a folder called savedModels, do:

`python BaselineModel.py --target=2 --model_save_path=savedModels/model.ckpt`

Other parameters can be found at the top of the file.

The results should match the ones found in our paper. Our end-to-end model with advice is a simple extension of this model and is described in the paper.


### Pre-trained models

The explanation for these models is provided in Section 2.3 of the paper, under the `Advice Grounding` section. In this code release, we only release the code to run these models on restrictive advice, not corrective. These pre-trained/grounding models are crucical to understand the advice text, especially in the input-specific model self-generated advice case.

You must run these models and save them in order to load them for the end-to-end model with advice.

To run the model to understand the 4 advice regions, saving the model in the `savedModels` directory:

`python PreTrainedModel.py`

To run the model for input-specific model self-generated advice (and thus have the model understand the text of many more regions):

`python PreTrainedModel.py --self_generated_advice=True`

Both models should achieve 99.99% accuracy, as described in the paper.


### End-to-End Model with Advice

The explanation for these models is provided in Section 2.4 of the paper. In short, this is the baseline [Bisk et. al](https://www.aclweb.org/anthology/papers/N/N16/N16-1089/) model, along with advice. 

In order to run this, you must have a pre-trained model saved. We will assume you ran it and saved it at: `savedModels/pre_trained_advice_advice/model.ckpt`. The tokens are saved at `saved_tokens/tokens.npy`. These are where the pretrainedmodel.py script saves it by default.

You also need the data files. We will assume it is saved at: `data/STxyz_Blank/...`

Now, you can run the end to end model:

`python BaselineModelAdvice.py`


### End-to-End Model with Input Specific Self-Generated Advice

In order to run input specific self-generated advice, you must first generate it. To that, run the following command. This will save the advice in a file `test_advice.npy`. The FLAG `test_advice_save_file` controls where it is saved.

`python BaselineModelAdvice.py --generate_advice=True`

Once the advice has been saved, you can use it at test time to run the end-to-end model with input specific self-generated advice. To do this, run:

`python BaselineModelAdvice.py --self_generated_advice=True`

