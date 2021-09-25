# xnet
xnet is a header only library which implements multilayer perceptron for data analysis.

# Contents

  This repository contains visual studio solution of the xnet project. The main files are:

  xnet.h - header file of the library

  xnet.cpp - contains definitions of the functions

  xnet_tester.cpp - file shows how to use the multilayer perceptron functions by using iris data set example. Neural network is trained on iris.csv data set and later tested

  iris.csv - iris data set

  testset.csv - contains some data for test

# How to Use
This is showed inside the file xnet_tester.cpp. 

1.) Basically an instance of xnet has to be created.

2.) Initialize the network

3.) Write a training loop

4.) Use the model on predicting data

# Features
The trained model can be saved in a .sxy file and can be read later.
