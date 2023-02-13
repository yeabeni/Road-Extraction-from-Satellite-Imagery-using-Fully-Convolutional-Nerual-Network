For the implementation of the proposed method, I used Keras open source deep learning 
library running on top of TensorFlow as backend. The framework experimented on a 
machine with specifications: Intel ® Core i5-7500 CPU at 3.40GHz 4-core with NVIDIA 
GeForce 1080 GTX GPU and 16GB RAM. I perform the experiments on a 
Massachusetts road dataset which is augmented, since the annotated images that are 
available are not enough for deep learning training, with a class of 2. I set batch size to 
5 and because of the classes that each image is classified, binary cross-entropy loss 
function is used as the cost function. The network is trained for 40 epochs; hence, the 
model with the best validation accuracy was selected and used for evaluation.
