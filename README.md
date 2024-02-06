# HotGestures

This is the recognizer used in the TVCG2023 paper [HotGestures](https://ieeexplore.ieee.org/document/10269004) using `Python 3.8` and `Pytorch 1.7.0`.

# Training
Two separate models, one for each hand. To train a recognizer:
1. Download the HotGestures dataset [here](https://doi.org/10.17863/CAM.97131).
2. Change the data paths in `train_two_hands.py` to the local path on your computer.
3. Change the `model_fold` paths in `train_two_hands.py` and `train.py` to a local directory
4. Run `train_two_hands.py`

The best models should be saved in `model_fold`. 

# Testing
Run `test_two_hands.py`. The script uses the unsegmented data from `/online_seq` to generate predictions using the specified models. Errors are calculated in terms of Levenshtain distances. 

# Acknowledgement
If you find this work useful please kindly cite us at:
```
@ARTICLE{10269004,
  author={Song, Zhaomou and Dudley, John J. and Kristensson, Per Ola},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={HotGestures: Complementing Command Selection and Use with Delimiter-Free Gesture-Based Shortcuts in Virtual Reality}, 
  year={2023},
  volume={29},
  number={11},
  pages={4600-4610},
  doi={10.1109/TVCG.2023.3320257}}
```
