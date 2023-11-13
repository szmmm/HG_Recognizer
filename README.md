# HotGestures

This is the recognizer used in HotGestures using `Python 3.8` and `Pytorch 1.7.0`.

# Training
Two separate models, one for each hand. To train a recognizer:
1. Download the HotGestures dataset
2. Change the data paths in `train_two_hands.py` to the local path on your computer.
3. Change the `model_fold` paths in `train_two_hands.py` and `train.py` to a local directory
4. Run `train_two_hands.py`

The best models should be saved in `model_fold`. 

# Testing
Run `test_two_hands.py`. The script uses the unsegmented data from `/online_seq` to generate predictions using the specified models. Errors are calculated in terms of Levenshtain distances. 
