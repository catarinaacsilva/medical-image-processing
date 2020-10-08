# Medical Image Processing

An implementation of an image processing pipeline and using machine learning algorithims capable of identify acanthocytes on blood.
The first simple implementation follows the method described in the following [paper](http://www.laccei.org/LACCEI2018-Lima/student_Papers/SP531.pdf).
Actually, the code is being improved to a more complete aproach to detect and classify these abnormal cells, to produce results more precious.

The pre-processing pipeline apply the following steps:

1. Convert image to gray scale
2. Apply 9x9 median filter to remove noise
3. Convert to binary using Otsu thresholding method
4. Filling operation to remove holes
5. Apply morphological reconstruction (elliptic shaped 9x9 kernel) to remove remove the medium-sized noise
6. Finally, canny edge detector is to apply to extract region contours

Features extracted:

1. Histogram from the chain code
2. Circularity
3. Roundness
4. Aspect-ratio
5. Solidity

Algorithms implemented:

1. kNN
2. Logistic Regression

Other algorithms used to compare results (future implementation):

1. Naive Bayes
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. Neural Network


## Requirements

The code requires the following libraries:

1. [OpenCV 4.2](https://opencv.org/)

The code also uses two other libraries, however they are distributed as single header dropin:

1. [nlohmann/json](https://github.com/nlohmann/json) for json manipulation
2. [adishavit/argh](https://github.com/adishavit/argh) for argument manipulation

Finally the code was written with C++17 features, that allow us to have access to filesystem functionalities independent from the operative system.

There was special care to improve the protability of the code.

## Compile

The code provide a [Makefile](Makefile) for compiling the code.
It should work on must of the Linux distribution.

## Execute

The code is comprise of two main programs:

1. train: used to create a kNN model
2. main: uses the previsouly learned model to classify several medical images.

In order to facilite the execution of the code the project already provides a file structure:

```console
.
+-- resources
|   +-- model    -> where the kNN models are stored
|   +-- test     -> where the images used for testing are stored
|   +-- train
|       +-- bad  -> where the anomalous instances are stored
|       +-- good -> where the healthy instances are stored
```

Finally, each main programs has several parameters.
The help message of each one of them is printed bellow:

```console
$ ./train -h
Program used to train a kNN model to identify anomalous blood cells.
usage: train [-p] [-k] [-i] [-o] [-h]

Parameters:
  -p, the preprocessig method               [default = 0]
  -m, ML model (0 - ARFF; 1 - KNN; 2 - LR)  [default = 0]
  -k, the number of nearest neighbors       [default = 1]
  -d, Minkowski distance of order p         [default = 2]
  -i, the input folder with images to train [default = './resources/train/']
  -o, the output model                      [default = './resources/model/model.json']
  -v, verbose
  -h, this help message
```

```console
$ ./main -h
Program used to identify anomalous blood cells.
usage: main [-p] [-k] [-i] [-o] [-h]

Parameters:
  -p, the preprocessig method            [default = 0]
  -m, the classification model           [default = './resources/model/model.json']
  -i, the folder with images to classify [default = './resources/test/']
  -v, verbose
  -h, this help message
```

## Authors

* **Catarina Silva** - [catarinaacsilva](https://github.com/catarinaacsilva)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



