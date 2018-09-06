# Gorprop+

[![Build Status](https://travis-ci.com/bestbug456/gorpropplus.svg?branch=master)](https://travis-ci.org/bestbu456/gorpropplus)&nbsp;
[![codecov](https://codecov.io/gh/bestbug456/gorpropplus/branch/master/graph/badge.svg)](https://codecov.io/gh/bestbug456/gorpropplus)&nbsp;
[![Go Report Card](https://goreportcard.com/badge/github.com/bestbug456/gorpropplus)](https://goreportcard.com/report/github.com/bestbug456/gorpropplus)&nbsp;

This project contains the Neural Network called "rprop+" (resilient backpropagation with weight backtracking) with He-et-al Initialization. The project is pure go: no external dependences!

##### From [Wikipedia](https://en.wikipedia.org/wiki/Rprop "Wikipedia")
Rprop+, short for resilient backpropagation, is a learning heuristic for supervised learning in feedforward artificial neural networks. This is a first-order optimization algorithm. This algorithm was created by Martin Riedmiller and Heinrich Braun in 1992.

##### [Paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417 "Paper") abstract
A new learning algorithm for multilayer feedforward networks, RPROP, is proposed. To overcome the inherent disadvantages of pure gradient-descent, RPROP performs a local adaptation of the weight-updates according to the behaviour of the errorfunction. In substantial difference to other adaptive techniques, the effect of the RPROP adaptation process is not blurred by the unforseeable influence of the size of the derivative but only dependent on the temporal behaviour of its sign. This leads to an efficient and transparent adaptation process. The promising capabilities of RPROP are shown in comparison to other wellknown adaptive techniques.

##### [He-et-al Initialization](https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e "He-et-al Initialization") discussion
This method of initializing became famous through a paper submitted in 2015 by He et al, and is similar to Xavier initialization, with the factor multiplied by two. In this method, the weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently. The weights are still random but differ in range depending on the size of the previous layer of neurons. This provides a controlled initialisation hence the faster and more efficient gradient descent.



###### Why I should use this algorithm?
Rprop+ can find significantly faster than a standard back propagation algorithm, apart from that we really focus in order to produce code with "zero footprints" memory, we have many other ideas for reducing the footprint but we have already done a lot of work!

###### Ok cool, what should I know in order to start?
It's easy to use our library. You have just to create a new `NeuralNetworkArguments` variable and pass it to our library! In order to maintain the flexibility, you can set a lot of arguments. You should be particular care about `ActivationFunction`, `DerivateActivation`, `ErrorFunction`, and `DerivateError` inputs since they are the activation function and the error function the NN will use. We provided the standard Logistic and Hyperbolic Tangent as activation function and SSE function as error function, but you can implement one of this function by yourself! Just respect the interface requirement! This following example creates a new NeuralNetwork, train it, validate it and predict a new sample.


###### Example
```
    funct main(){
        // create arguments to pass to the library
    
        args := NeuralNetworkArguments{
            HiddenLayer:        []int{2},
            InputSize:          3,
            OutputSize:         2,
            Threshold:          0.01,
            StepMax:            100,
            LifeSignStep:       100,
            LinearOutput:       false,
            Minus:              10,
            Plus:               100,
            ActivationFunction: gorpropplus.Logistic,
            DerivateActivation: gorpropplus.DerivateLogistic,
            ErrorFunction:      gorpropplus.SSE,
            DerivateError:      gorpropplus.DerivateSSE,
        }
    
        // Get a fresh new neural network
        NN, err := NewNeuralNetworkAndSetup(args)
        if err != nil {
            Log.Printf("Error while creating a new neural network: %s", err.Error())
        }
        // Train the neural network
        err = nn.Train(inputData,outputData)
        if err != nil {
            Log.Printf("Error while training the neural network: %s", err.Error())
        }
        // Validate the train
        confusionMatrix,err:=nn.Validate(validationSetInput,ValidationSetOutput)
        if err != nil {
            Log.Printf("Error while training the neural network: %s", err.Error())
        }
        // Predict a new sample
        prediction,err:=nn.Predict(input)
        if err != nil {
            Log.Printf("Error while training the neural network: %s", err.Error())
        }
        Log.Printf("Prediction result: %v", prediction)
    }
```
    
###### Ok it seems easy. But how about performance?
Glad you ask. Even if the Golang compiler do a lot of work we work really hard in order to optimise our code. Look at this benchmark:

```
BenchmarkTrain-4                                           10000        174518 ns/op      152150 B/op       2573 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4        1000000          1326 ns/op         496 B/op         12 allocs/op
BenchmarkComputeNet-4                                     500000          3134 ns/op        1464 B/op         33 allocs/op
BenchmarkCalculateGradients-4                             500000          2673 ns/op        1664 B/op         40 allocs/op
BenchmarkPredictWithLinearOutputTRUE-4                   2000000           565 ns/op         400 B/op         14 allocs/op
BenchmarkPredictWithLinearOutputFALSE-4                  3000000           502 ns/op         360 B/op         12 allocs/op
```

###### This library seem very cool! Who create it?
This library was created by me (bestbug) and the best Data Scientist I've ever met [Franca Marinelli](https://www.linkedin.com/in/franca-marinelli-30b086126/ "Franca Marinelli")

###### If I found a bug or something wrong?
You can create a pull request to us (really appreciate) or if you don't have any idea about what is going wrong you can always open an issue here on GitHub!
