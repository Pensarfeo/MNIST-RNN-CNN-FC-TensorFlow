
Little repo with examples of Recursive, Convolutional, and Fully Connected neural networks applyed to the MNIST Dataset.

## status
Code is still messy... needs more user fredlyness

## structure
###nn.py
    Main script: it calls everything it needs and runs the training and testing session

###dataset.py
    It providers utilities to generate the training and testing deta for the feed_dict ``feed_dict_gen``; it will also download the data if not present and format it.

###trainer.py
    It defines a function that sets up all the training graph

###utils.py
    Provides some utility functions

####models/layers.py
    functions to sets up the layers for the NN

###models/cnn.py
    sets up the cnn graph

###models/fc.py
    sets up the fc graph

###models/rnn.py
    sets up the rnn graph

## Tensorboard
to view the logs in tb run
```
    tensorboard --logdir ./tmp
```