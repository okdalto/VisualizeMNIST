# VisualizeMnist
This project is real-time visualization of a network recognizing digits from user's input.
I trained a network using MNIST dataset and parsed the weight data in python. With this data, I implemented my own custom matrix multiply function and activation functions and other functions that are needed to run the network in Processing. At first trail, because MNIST dataset is preprocessed for numbers to be in the center of the images, there was a precision problem when the user's input is placed little bit far away from the center. I used a data augmentation technic in the training process to resolve this problem. 

# Installation
To run this code, you need [Processing](https://www.processing.org/download/) IDE and a library named [peasycam](http://mrfeinberg.com/peasycam/).
