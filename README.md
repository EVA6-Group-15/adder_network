# SUM(MNIST, Random Number) Detector

## Objective
A Neural Netwrok which 
* Take two inputs:
    * an image from MNIST dataset, and
    * a random number between 0 and 9
* Gives two outputs:
    * the "number" that was represented by the MNIST image, and
    * the "sum" of this number with the random number that was generated and sent as the input to the network

![Exmaple](images/expected_viz.png)

## Data Representation
A Custom class `MNISTRandomNumber` is created which take two Inputs.
The MNIST Data either Train or Test Data and the Trasnformation to be performed on these images.

The class have a \_\_getitem\_\_() method, which will index the MNIST data and also created a Random Number
between 0 and 9.

The Random Number is than converted to a one hot encoded vector whcih also side MNIST image is sent as a tuple of
input from the class.
On the other hand the MNIST label and the **sum** of MNIST label the random number genrated is added and the result along
with MNIST label is send a tuple of labels.

### Intuition behind the Data
The Data is 