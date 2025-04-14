# Dreamer

Dreamer is a visual Model-Based Reinforcement algorithm, that learns a world model which captures latent dynamics from high-level pixel images and trains a control agent entirely in imagined rollouts from the learned world model.

This work is my attempt at reproducing Dreamerv1 & v2 papers in pytorch specifically for continuous control tasks in deepmind control suite.

#### Differences in implementation:

 1. Replay Buffer manages episodes instead of transitions, making sure that we don't mix episodes when sampling
 2. Although less flexible, Convolution models where layed out step by step for readibility
 3. This branch includes EfficientNetB0 as an image encoder to be chosen

#### Implementation of the Dreamer agent with Pretrained Vision Embeddings (1M Steps)
<img src="https://github.com/user-attachments/assets/a842f33f-9847-4323-9300-950630fce714" width="150">
<img src="https://github.com/user-attachments/assets/619ce071-4dea-45a6-be0f-4ceaa7cb0e26" width="150">
<img src="https://github.com/user-attachments/assets/a2bd3690-9e1c-4518-86b4-18b880f8523b" width="150">
<img src="https://github.com/user-attachments/assets/736bb65d-62d2-45da-948c-e66ad21d1639" width="150">


## Code Structure
Code structure is similar to original work by Danijar Hafner in Tensorflow

`dreamer.py`  - main function for training and evaluating dreamer agent

`utils.py`    - Logger, miscellaneous utility functions

`models.py`   - All the NNs for world model and actor

`replay_buffer.py` - Experience buffer for training world model

`env_wrapper.py`  - Gym wrapper for Dm_control suite

Runs can be configured from the config.json

## Installation

Run:
`conda env create -f environment.yml`

#### For training
You can select any block index from 1 to 6 with 'efficientnet' activated, and 'cnn' will use the original image encoder.

The "freeze_encoder" flag activates gradient flow to the pretrained encoder or deactivates it.

Finally, you can pick the embedding size, with 1024 being the original used on the CNN

`python dreamer.py --config config.json --train --env walker-walk --encoder_type <'efficientnet' or 'cnn'> --obs_embed_size=1024 --block_index=<1-6> --seed=1 --freeze_encoder`
#### For Evaluation
`python dreamer.py --config config.json --evaluate --restore --checkpoint_path '<your_ckpt_path>'`



## Acknowledgements
This code is heavily inpsired by following open-source works

dreamer by Danijar Hafner : https://github.com/danijar/dreamer

dreamer-pytorch by yusukeurakami : https://github.com/yusukeurakami/dreamer-pytorch

Dreamerv2 by Rajghugare : https://github.com/RajGhugare19/dreamerv2

Dreamer by adityabingi : https://github.com/adityabingi/Dreamer
