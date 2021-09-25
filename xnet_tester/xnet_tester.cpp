// xnet_tester.cpp : Small example which shows how to use the xnet multi layer perceptron for training and testing data. Here iris data set is used as example

#include <iostream>
#include "xnet.h"


// --------------- Testing ----------------


void test(xnet::xnet net)
{

	net.global_clear();


	int LINE = 50;
	std::string st;

	double lol = 0;

	int counter = 0;
	int sample = 0;
	std::ifstream file("testset.csv"); 
	std::string value;

	std::vector<double> inp;
	std::vector<double> tar;
	// ####################### Testing ################

	
	std::getline(file, st);


	std::istringstream ss(st);
	std::string token;

	while (std::getline(ss, token, ',')) {

		lol = atof(token.c_str());


		if (counter == 4)
		{
			
			if (token == "Iris-setosa")
			{
				
				tar.push_back(1);
				tar.push_back(0);
				tar.push_back(0);



			}
			else if (token == "Iris-versicolor")
			{
				
				tar.push_back(0);
				tar.push_back(1);
				tar.push_back(0);
			}
			else if (token == "Iris-virginica")
			{
				
				tar.push_back(0);
				tar.push_back(0);
				tar.push_back(1);
			}



		}
		else
		{
			
			inp.push_back(lol);
		}
		

		counter++;

	}


	file.clear();
	file.seekg(0, std::ios::beg);

	net.forwardPass(inp, tar);


	std::cout << token.c_str();
	
	std::cout << " \n";
	std::cout << std::to_string(net.output_layers[net.output_layers.size() - 1][0]).c_str();
	std::cout << " \n";
	std::cout << std::to_string(net.output_layers[net.output_layers.size() - 1][1]).c_str();
	std::cout << " \n";
	std::cout << std::to_string(net.output_layers[net.output_layers.size() - 1][2]).c_str();
	std::cout << "The flower is: " << token;
	

}

void train(xnet::xnet net)
{


	int LINE = 50;
	std::string st;

	double lol = 0;

	int counter = 0;
	int sample = 0;
	std::ifstream file("iris.csv"); 
	std::string value;

	std::vector<double> inp;
	std::vector<double> tar;


	//###################################### trainer ################################

	for (int tr = 0; tr < net.epoch; ++tr)    // für jeden epoch
	{

		std::cout << " epoch################################################ " << tr << std::endl;
		sample = 0;




		for (int samples = 0; samples < 50; ++samples)  //############# taking samples.......
		{

			for (int inc = 0; inc < 3; ++inc)
			{

				for (int i = 0; i <= samples + LINE * inc; i++)
					std::getline(file, st);
				
				sample++;

				tar.clear();
				inp.clear();

				// here parse the comma and take target and input....

				std::istringstream ss(st);
				std::string token;

				while (std::getline(ss, token, ',')) {

					lol = atof(token.c_str());


					if (counter == 4)
					{
						
						if (token == "Iris-setosa")
						{
							
							tar.push_back(1);
							tar.push_back(0);
							tar.push_back(0);



						}
						else if (token == "Iris-versicolor")
						{
							
							tar.push_back(0);
							tar.push_back(1);
							tar.push_back(0);
						}
						else if (token == "Iris-virginica")
						{
							
							tar.push_back(0);
							tar.push_back(0);
							tar.push_back(1);
						}



					}
					else
					{
						
						inp.push_back(lol);
					}
					
					counter++;

				}

				counter = 0;

				//##################################################################

				file.clear();
				file.seekg(0, std::ios::beg);

				net.forwardPass(inp, tar);
				net.backwardPass(net.output_layers[net.output_layers.size() - 1], tar);



			}



		}

		double ev_sum = 0;
		for (int er = 0; er < net.avg_avg_error.size(); ++er)
		{


			ev_sum += net.avg_avg_error[er];

		}

		ev_sum = ev_sum / net.avg_avg_error.size();



		net.avg_avg_error.clear();

		std::cout << " avg_error is:  " << ev_sum << std::endl;

	}


	

	// #####################################################################


}

int main()
{
    xnet::xnet net;
    
	//----- Initialize network -----
	
	net.learning_rate_bias = 0.01;
	net.learning_rate_parameters = 0.01;
	net.epoch = 500;
	net.inputsize = 4;


	net.L.push_back(4);  // two dimensional state
	net.L.push_back(10);
	net.L.push_back(3);  // 4 q werten für 4 actionen..... 4 richtung 
	
	net.weightInitializer();

	//-------------------------------
	
	train(net);
    

	//net.write_parameters();
	//net.load_parameters_feed_forward();

	test(net);


    std::cout << "End!\n";
}
