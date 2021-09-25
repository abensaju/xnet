#pragma once
//Header file of xnet

#include <random>
#include <map>
#include <math.h>

#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <stdlib.h>



namespace xnet
{

	class xnet {


	public:



		xnet();
		~xnet();


		std::vector<std::vector<double>> output_layers;          // Output of each layers
		std::vector<double> avg_avg_error;
		std::vector<int> accuracy_vector;
		// ########################### Standard matrix ########################
		typedef std::vector<double> cl;
		typedef std::vector<cl> paramatrix;

		std::vector<double> bias_layer;   //  bias for a single feed forward network
		std::vector<std::vector<double>> bias_net;    // bias for the whole network


		std::vector<double> error_netout_feed_forward;
		// ####################################################################

		// ############# train criterias ######################
		int epoch;
		int minibatch;

		double learning_rate_parameters;
		double learning_rate_bias;
		// ##############################################

		int maxlayers;   // Number of layers
		int inputsize;         // Size of input
		std::vector<int> L;

		std::vector<paramatrix> W;         // vector of matrices  only global parameter. All others die in one backpropagation


		void  layerFactory();
		void  weightInitializer();


		double  squaredError(std::vector<double> out, std::vector<double> tar, std::vector<double>& error);
		double  crossEntropy(std::vector<double> out, std::vector<double> tar, std::vector<double>& error);
		double  softmax();
		double  sigmoid(double inp);
		double  softmax_netout();
		double  squaredError_out(double value);
		void  backwardPass(std::vector<double> out, std::vector<double> tar);
		void  global_clear();                  //Clearing all storage vectors after one feedforward and backpropagation step
		void  forwardPass(std::vector<double> inp, std::vector<double> tar);
		void  write_parameters();
		void  load_parameters_feed_forward();


	};


}

