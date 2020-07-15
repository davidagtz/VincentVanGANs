# Deep Learning Notes

## Batch Normalization

-   This prevents the vanishing/exploding gradient problem in which the models learns to slowly or to quickly.
-   It does this by first normalizing (mean) then standardizing (standard deviation) the data to be 0 and 1 respectively.
-   In order to do this, each node must now also carry an alpha and gamma which are new variables to learn. These represent the predicted true mean and true standard deviation.
-   Greatly speeds up training by making gradient descent easier (i.e. less steep/flat zones in the function)

## ReLU

-   max(0, x)
-   This is better than sigmoid/tanh because it does not diminish values that are either small or large.
-   Variants
    -   Leaky ReLU - Piecewise function. < 0 => alpha \* x. >= 0, x. This helps prevent zeroing out of small values.

## CrossEntropy Loss

-   Just use it
-   Binary or categorical depending on amount of output neurons

## Regularization

-   Methods that prevent overfitting the model
-   L2 Regularization
    -   Adds a term to the loss function
    -   Penalizes large weights
    -   One parameter that scales the term added
-   Dropout
    -   Randomly selects nodes to not be trained on
    -   Helps with generalization
    -   One parameter that detemines frequency of dropout

## Problems I have encountered

-   Convergence on high loss
    -   probably too high of learning rates
    -   probably due to vanishing or exploding gradient
