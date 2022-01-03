# SoccerRobotPerception 
Detection of Goalpost, Ball, Robot and Segmentation of field and fieldlines with a two headed network.

# Requirements
1. PyTorch
2. Numpy
3. Matplotlib
4. OpenCV
5. Scikit-Image

# Training

To train the model, please run

```
$ python train.py --lr=<learning_rate> --epochs=<epochs> --blob_dir=<Detection images path> --seg_dir=<Segmentation images path>
```

# Visualization

An additional visualizations.ipynb is added to perform the testing and visualize the results. Please run this file to see the visualizations


Our sincere thanks for Hafez Farazi from University of Bonn for his continous support and advice during this project.
