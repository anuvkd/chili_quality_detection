# Chili Quality Detection using CNN

This Python script uses a convolutional neural network (CNN) implemented with TensorFlow/Keras to detect chili quality grades (Grade1 to Grade4) based on an image dataset. The dataset should be organized into directories for each grade, and the script splits it into training and validation sets, trains the model, and evaluates its performance.

## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn (sklearn)
- numpy
- matplotlib (optional for plotting training history)

## Installation

### Clone the Repository

```bash
git clone https://github.com/anuvkd/chili_quality_detection
cd chili_quality_detection
```

### Install Dependencies:

```bash
pip install tensorflow scikit-learn numpy matplotlib
```

## Dataset Setup

### Download Dataset:

Download the dataset from [here](https://drive.google.com/drive/folders/1_dwy5JSWnpF1mgfY572-SY5JSV1RCAZR?usp=sharing)
Extract the dataset into a directory (dataset) on your local machine.

#### Organize the dataset into subdirectories for each chili grade:

markdown
Copy code
dataset
├── Grade1
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Grade2
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Grade3
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
└── Grade4
├── image1.jpg
├── image2.jpg
└── ...

#### Training:

```bash
python3 validation.py
```

The script will split the dataset into training and validation sets, train the CNN model, and save the trained model as chili_quality_detection_model.h5.
Evaluation:

After training, the script evaluates the model's accuracy on the validation set.
Model Details

#### Model Architecture:

The CNN model architecture consists of convolutional layers followed by max pooling, flattening, dense layers, dropout for regularization, and a softmax output layer for classification into 4 grades.

#### Training Parameters:

Adjust parameters like batch_size, epochs, and model architecture in chili_quality.py according to your dataset size and computational resources.

### Run code

```bash
python3 chili_quality.py
```
