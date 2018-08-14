# Gorprop+

This project contain the Neural Network called "rprop+" (resilient backpropagation with weight backtracking). The project at the moment have the train and the predict function completed and tested, the validte function will be create soon.

##### From [Wikipedia](https://en.wikipedia.org/wiki/Rprop "Wikipedia")
Rprop+, short for resilient backpropagation, is a learning heuristic for supervised learning in feedforward artificial neural networks. This is a first-order optimization algorithm. This algorithm was created by Martin Riedmiller and Heinrich Braun in 1992.

##### [Paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417 "Paper") abstract
To overcome the inherent disadvantages of pure gradient-descent, RPROP performs a local adaptation of the weight-updates according to the behaviour of the errorfunction. In substantial difference to other adaptive techniques, the effect of the RPROP adaptation process is not blurred by the unforseeable influence of the size of the derivative but only dependent on the temporal behaviour of its sign. This leads to an efficient and transparent adaptation process.


###### Why I should use this algoritm?
Rprop+ can find significantly faster than a standard back propagation algorithm, a part from that we really focus in order to produce code with "zero footprint" memory, we have many other idea for reduce the footprint but we have already do a lot of work!

###### Ok cool, what should I know in order to start?
It's esay to use our libray. You have just to create a new `NeuralNetworkArguments` variable and pass it to our library! In order to mantain the flexibility, you can set a lot of arguments. You should be particular care about `ActivationFunction`, `DerivateActivation`, `ErrorFunction`, and `DerivateError` inputs since they are the activation function and the error function the NN will use. We provided the standard Logistic and Iperbolic Tangent as activation function and SSE function as error function, but you can implement one of this function by yourself! Just respect the interface requirment! This following example create a new NeuralNetwork, train it, validate it and predict a new sample.


######Example
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
    
###### Ok it seems esay. But how about performance?
Glad you ask. Even if the Golang compiler do a lot of work we work really hard in order to optimise our code. Look at this benchmark:

```
BenchmarkTrain-4                                            5000        330654 ns/op      164370 B/op       2780 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4        1000000          1530 ns/op         496 B/op         12 allocs/op
BenchmarkComputeNet-4                                     300000          3616 ns/op        1464 B/op         33 allocs/op
BenchmarkCalculateGradients-4                             500000          3396 ns/op        1664 B/op         40 allocs/op
```

###### This library seem very cool! Who create it?
This library was created by me (bestbug) and the best Data Scientist I've ever met [Franca Marinelli](https://www.linkedin.com/in/franca-marinelli-30b086126/ "Franca Marinelli")

###### If I found a bug or something wrong?
You can create a pull request to us (really appreciate) or if you don't have any idea about what is going wrong you can always open a issue here on github!