# Medical Image Processing

A simple implementation of a image processing pipeline capable of identifiyng Acanthocytosis.
The implemenation follows the method described in the following [paper](http://www.laccei.org/LACCEI2018-Lima/student_Papers/SP531.pdf).

The pre-processing pipeline apply the following filters:

1. Converts colors images to gray scale
2. Applys a median blur filter with size 9x9
3. Converts the image to binary scale using Otsu method
4. Applies morphological open and close operations (ELLIPSE shape with size 9x9)
5. Re-applies the median blur filter
6. Segments the image with Canny edge detector or morphological gradient

After that, it uses a find contour method to segment the image into objects.
Finally, each object is classified using a kNN algorithm.
The kNN algorithm uses chain code histogram and the circularity value as features.

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


shif
nao morfologica (hibrido)
descritores sift (forma)


clustering

problema???

datasets anotados

oncologia hematologica

margens de erro
 erro 10% signif

distribuiçao das dimensoes

saber se são sedimentos

ligar opencv ao micros




