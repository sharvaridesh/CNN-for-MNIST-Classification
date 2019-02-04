# CNN-for-MNIST-Classification

## Dataset 
source: http://yann.lecun.com/exdb/mnist/

## Architecture of the Network

- We create 32, 33 convolutional filters with ReLU (Rectified Linear Unit) node activations. 
- After this, we still have a height and width of 28 nodes. 
- We then perform down-sampling by applying a 22 max pooling operation with a stride of 2. 
- Layer 2 consists of the same structure, but now with 64 filters channels and another stride-2 max pooling for down-sampling. 
- We then flatten the output to get a fully connected layer with 1000 nodes. These layers will use ReLU node activations. 
- Finally, we use a softmax classification layer to output the 10 digit probabilities.

## Results

Hyperparameter tuning is done to observe optimum performance of the model. For hyperparameter tuning, cross-validation is done. This can be visualized from the image below. 

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/cross-validation.JPG) |
|:--:|
|*Cross Validation Visualization*|

### Hyperparameter Tuning

The learning rates taken into consideration for this model are 0.001,0.003,0.01,0.03.For each of these learning rates, training and validation accuracy and, training and validation loss are observed.

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/accuracy_batchsize64.JPG) |
|:--:|
|*Accuracy visualization for batch size 64*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/loss_batchsize64.JPG) |
|:--:|
|*Loss visualization for batch size 64*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/accuracy_batchsize128.JPG) |
|:--:|
|*Accuracy visualization for batch size 128*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/loss_batchsize128.JPG) |
|:--:|
|*Loss visualization for batch size 128*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/accuracy_batchsize256.JPG) |
|:--:|
|*Accuracy visualization for batch size 256*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/loss_batchsize256.JPG) |
|:--:|
|*Loss visualization for batch size 256*|

### Activation Functions

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/accuracy_relu.JPG)
|:--:|
|*Model Accuracy for ReLU activation function*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/loss_relu.JPG)
|:--:|
|*Model Loss for ReLU activation function*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/accuracy_sigmoid.JPG)
|:--:|
|*Model Accuracy for Sigmoid activation function*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/loss_sigmoid.JPG)
|:--:|
|*Model Loss for Sigmoid activation function*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/accuracy_tanh.JPG)
|:--:|
|*Model Accuracy for tanH activation function*|

![](https://github.com/sharvaridesh/CNN-for-MNIST-Classification/blob/master/results/loss_tanh.JPG)
|:--:|
|*Model Loss for tanh activation function*|

- The training accuracy is maximum for the activation function tanH. 
- The validation accuracy is maximum for activation function ReLU.

### Testing Set Results

The test set accuracy that is obtained on this model for learning rate=0.001 and batch size=128 is **99.31 percent**.

### Model Optimization

For this model, maximum accuracy is obtained for, 
- Learning rate of **0.001**
- Batch size of **128** 
- Activation function **ReLU** 

For these values, the cost function is minimized.
