# Color Classifier
This is an AI that i made to classify colors
Given 3 inputs (Red, Green, Blue) the program will
output one of these labels:
- **Red** 
- **Green**
- **Blue**
- **Orange**
- **Yellow**
- **Pink**
- **Purple**
- **Brown**
- **White**
- **Black**

## Neural Network
I used pytorch to create the neural network
The Neural Net is a feed forward network
It has 3 inputs, 2 hidden layers with 6 neurons each
and 10 outputs

## Data
I used Daniel Shifftman's [CrowdSourceColorData](https://github.com/CodingTrain/CrowdSourceColorData)
data to train on 

## Installation
`pip install pygame`

`pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html`