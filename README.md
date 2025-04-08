# IMAGE-CLASSIFICATION-MODEL

COMPANY:CODTECH IT SOLUTIONS

NAME:RAKSHITHA P

INTERN ID: CT04WS08 

DOMAIN:MACHINE LEARNING

DURATION: 4 WEEEKS

MENTOR: NEELA SANTOSH

##This Python code builds and trains a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images from the **CIFAR-10 dataset** — a dataset of 60,000 32×32 color images across 10 classes (like airplanes, cats, and cars). It starts by loading and normalizing the dataset so that pixel values range between 0 and 1 for better training performance. Then, a CNN model is defined with three convolutional layers (with ReLU activation), two max-pooling layers to reduce dimensionality, a flattening layer to convert 2D features to 1D, and two dense (fully connected) layers — the last one using softmax to output probabilities for 10 classes. The model is compiled using the **Adam optimizer** and **sparse categorical crossentropy** as the loss function. It is trained for 10 epochs using the training data and validated on the test data. Finally, the model’s test accuracy is printed, and a plot is generated to show how the training and validation accuracy evolved over each epoch.

#output:
![Image](https://github.com/user-attachments/assets/095f3009-65ef-4c71-a4ab-92c5241f8710)
