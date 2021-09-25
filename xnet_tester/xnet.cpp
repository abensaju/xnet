// This file contains the definitions of library functions
#include "xnet.h"

namespace xnet
{
	// ############## this should be done by user
	xnet::xnet()
	{

		epoch = 500;
		minibatch = 1;
		// ##############################################

		maxlayers = 3;   // Number of layers
		inputsize = 4;   // Size of input


		learning_rate_parameters = 0.001;
		learning_rate_bias = 0.001;

	}

	xnet::~xnet()
	{



	}

	// ########### This should be done by user

	void xnet::layerFactory()
	{

		//############## Layer definition ###################
		L.push_back(4);                  // Nodes in layer one 
		L.push_back(4);
		L.push_back(3);
		maxlayers = L.size();
		// ################################################## 

	}

	void xnet::weightInitializer()
	{


		// ##################  Go xavier initializierung ###############

		double randomNumber;
		int tmp;
		tmp = inputsize;

		std::vector<double> was;
		std::vector<std::vector<double>> wass;
		int layerSize = 0;
		double variance = 0;

		for (int k = 0; k < (L.size() - 1); ++k)
		{

			for (int j = 0; j < L[k + 1]; ++j)            //second layer Second vector place
			{

				for (int i = 0; i < L[k]; ++i)                  //  First layer First vector place
				{
					variance = double(2.0 / (L[k] + L[k + 1]));
					//std::cout << "variance is" << variance << std::endl;
					std::random_device rd;
					std::mt19937 e2(rd());

					std::normal_distribution<double> distribution(/*mean=*/0.0, /*stddev=*/variance);   //0.5
					randomNumber = distribution(e2);
					//randomNumber = 0.5;
					was.push_back(randomNumber);  //  first i- all j
					//std::cout << "The weights at" << k << "-" << i << "-" << j << ":" << randomNumber << "\n";

				}

				wass.push_back(was);
				was.clear();

			}

			W.push_back(wass);
			wass.clear();

		}

		// ################################################################

		// ############## initializing biases #############


		for (int i = 1; i < L.size(); ++i)
		{

			for (int j = 0; j < L[i]; ++j)
			{

				bias_layer.push_back(0.01);


			}

			bias_net.push_back(bias_layer);
			bias_layer.clear();

		}


		// ################################################


	}


	// ##################### Error Functions ################################

	double xnet::squaredError(std::vector<double> out, std::vector<double> tar, std::vector<double>& error)
	{

		error.clear();
		double avgerror = 0;
		double err = 0;
		for (int j = 0; j < L[L.size() - 1]; ++j)
		{


			err = tar[j] - out[j];
			error.push_back(0.5 * pow(err, 2));
			avgerror = avgerror + error[j];

		}

		avgerror = avgerror / L[L.size() - 1];

		return(avgerror);
	}


	double xnet::crossEntropy(std::vector<double> out, std::vector<double> tar, std::vector<double>& error)
	{

		error.clear();
		double avgerror = 0;
		double err = 0;
		for (int j = 0; j < L[L.size() - 1]; ++j)
		{

			err = tar[j] * log(out[j]);
			error.push_back(err);
			avgerror = avgerror + error[j];

		}

		avgerror = -avgerror;

		return(avgerror);
	}

	//#####################################################################


	// ############################# Non linearity ########################

	double xnet::softmax()
	{






		return(0.0);
	}





	// ######################## Derivative formulas ############################


	double xnet::softmax_netout()
	{




		return(0.0);
	}

	double xnet::squaredError_out(double value)
	{




		return(0.0);
	}


	double xnet::sigmoid(double inp)
	{

		double out = 0;

		out = 1 / (1 + exp(-inp));

		return(out);
	}


	// ######################################################################



	// ######################################################################

	void xnet::backwardPass(std::vector<double> out, std::vector<double> tar)
	{


		//std::cout << " 0 of output layer last :" << out[0] << std::endl;
		//std::cout << " 1 of output layer last :" << out[1] << std::endl;
		//std::cout << " 2 of output layer last :" << out[2] << std::endl;

		//tar.clear();
		//tar.push_back(1);
		//tar.push_back(0);
		//tar.push_back(0);



		// ############################
		double temp_out_netout = 0;
		std::vector<double> out_netout_layer;
		std::vector<std::vector<double>> out_netout;
		// ############################

		// ##################################
		std::vector<double> netout_W_;
		std::vector<std::vector<double>> netout_W__;
		std::vector<paramatrix> netout_W;
		// #################################

		double temp_error_W;
		std::vector<double> error_W_i;
		std::vector<std::vector<double>> error_W_j;
		std::vector<paramatrix> error_W;
		// ###########################

		//############################
		std::vector<std::vector<double>> error_netout;         // Always storing the error_netout from wach layer
		std::vector<double> error_netout_layer;
		double error_netout_temp = 0;
		//###########################

		double temp_error_out;

		// ############################

		double err = 0;
		double temp_grad; // gradient of netout by Weight
		double netout_W_temp;


		// ####################### Finding error for last layer ###############



		std::vector<double> error_out;      //differential of error w.r.t. output

		for (int j = 0; j < L[L.size() - 1]; ++j)      //second layer Second vector place
		{

			//error = squaredError(out,tar);
			error_out.push_back(out[j] - tar[j]);
			out_netout_layer.push_back(out[j] * (1 - out[j]));
			error_netout_layer.push_back((out[j] - tar[j]) * (out[j] * (1 - out[j])));
			//error_netout_layer.push_back((out[j] - tar[j]));        // Testing to alter values to see how it influences

			bias_net[bias_net.size() - 1][j] = bias_net[bias_net.size() - 1][j] - (learning_rate_bias * error_out[j] * 1);   // Direct update bias... for last layer

			for (int i = 0; i < L[(L.size() - 1) - 1]; ++i)                  //  First layer Second vector place
			{

				netout_W_temp = output_layers[(L.size() - 1) - 1][i];
				netout_W_.push_back(netout_W_temp);

				temp_error_W = (out[j] - tar[j]) * (out[j] * (1 - out[j])) * (netout_W_temp);
				error_W_i.push_back(temp_error_W);
			}     // i

			netout_W__.push_back(netout_W_);
			netout_W_.clear();

			error_W_j.push_back(error_W_i);
			error_W_i.clear();

		}    //  j

		error_W.push_back(error_W_j);
		error_W_j.clear();

		netout_W.push_back(netout_W__);
		netout_W__.clear();

		error_netout.push_back(error_netout_layer);
		error_netout_layer.clear();
		out_netout.push_back(out_netout_layer);
		out_netout_layer.clear();

		//#########################################################

		
		double tmp_error2_out1 = 0;

		for (int k = (L.size() - 2); k > 0; --k)//   weights for differnt layers - Starts from second last layer. Because last layer is already taken care. 
		{
			for (int j = 0; j < L[k]; ++j)            //second layer First vector place
			{
				//################ Out_netout #####################


				out_netout_layer.push_back(output_layers[k][j] * (1 - output_layers[k][j]));

				// ################################################

				// ########################################### Special loop for netout2_out1 ###################

				for (int f = 0; f < L[k + 1]; ++f)
				{


					// ############# Error_netout ######################
					error_netout_temp += error_netout[(L.size() - 1) - (k + 1)][f] * W[k][f][j] * out_netout_layer[j];         // Total error_netout grads Actually this is error_out
										//error_netout1 =  error2_out1      * out1_netout1
					tmp_error2_out1 += error_netout[(L.size() - 1) - (k + 1)][f] * W[k][f][j];   //  This is for the bias update

				}

				// ############ Error_netout ###################

				error_netout_layer.push_back(error_netout_temp);
				error_netout_temp = 0;

				// ############################################
				//#### Updating bias #################################

				bias_net[k - 1][j] = bias_net[k - 1][j] - (learning_rate_bias * tmp_error2_out1 * 1);
				tmp_error2_out1 = 0;
				// ####################################################

				// ###################################


				for (int i = 0; i < L[k - 1]; ++i)                  //  First layer Second vector place
				{


					// ################## netout_W ############### 

					assert((k - 1) >= 0);

					netout_W_temp = output_layers[k - 1][i];
					netout_W_.push_back(netout_W_temp);

					// ################# Error_W #################

					temp_error_W = error_netout_layer[j] * (netout_W_temp);
					error_W_i.push_back(temp_error_W);

					// ###########################################



				}   // for loop i

				// ############# netout_W ##########

				netout_W__.push_back(netout_W_);
				netout_W_.clear();

				// ################################


				// ############### Error_W ###########

				error_W_j.push_back(error_W_i);
				error_W_i.clear();

				// ##################################



			}  // for loop for j


			// ################# out_netout #########

			out_netout.push_back(out_netout_layer);
			out_netout_layer.clear();

			// ################ netout_W #############

			netout_W.push_back(netout_W__);
			netout_W__.clear();

			// ############### Error_netout ##########

			error_netout.push_back(error_netout_layer);
			error_netout_layer.clear();

			// ############ Error_W ##########

			error_W.push_back(error_W_j);
			error_W_j.clear();

			//##############################

		} // for loop for k

		// ############### Here is the part where the error_netout is extended to the input itself #########
		// Just for the first layer

		error_netout_temp = 0;
		error_netout_layer.clear();


		for (int i = 0; i < L[0]; ++i)
		{
			for (int j = 0; j < L[1]; ++j)
			{




				// ############# Error_netout ######################
				error_netout_temp += error_netout[error_netout.size() - 1][j] * W[0][j][i];         // Total error_netout grads Actually this is error_out
									//error_netout1 =  error2_out1      * out1_netout1


			}
			error_netout_layer.push_back(error_netout_temp);
			error_netout_feed_forward.push_back(error_netout_temp);     // For global access
			error_netout_temp = 0;

		}

		error_netout.push_back(error_netout_layer);
		error_netout_layer.clear();

		//std::cout << "size : " << error_netout[error_netout.size() - 1].size();

		// ##################################################################################################

		//############################### Irgenwann muss mann Dieses weights aktulisieren ##########################

		for (int k = (L.size() - 1); k > 0; --k)//   weights for differnt layers
		{

			for (int j = 0; j < L[k]; ++j)            //second layer First vector place
			{

				for (int i = 0; i < L[k - 1]; ++i)                  //  First layer Second vector place
				{

					W[k - 1][j][i] = W[k - 1][j][i] - (learning_rate_parameters * error_W[(L.size() - 1) - k][j][i]);

				}

			}

		}

		//#####################################################


	}   // Backpass



	void xnet::global_clear()                  //Clearing all storage vectors after one feedforward and backpropagation step
	{


		output_layers.clear();
		error_netout_feed_forward.clear();


	}

	void xnet::forwardPass(std::vector<double> inp, std::vector<double> tar)
	{

		global_clear();        // clears values in each layer  - output_layers

		std::vector<double> error;
		
		std::vector<double> out;
		

		output_layers.push_back(inp);       // First input is added to output of layers vector

		double sum = 0;
		for (int k = 0; k < (L.size() - 1); ++k)//   weights for differnt layers
		{
			for (int j = 0; j < L[k + 1]; ++j)            //second layer First vector place
			{
				for (int i = 0; i < L[k]; ++i)                  //  First layer Second vector place
				{

					sum += W[k][j][i] * inp[i];

				}     //  i

				//############### logistic regression ############

				sum = sigmoid(sum + bias_net[k][j]);
				//sum = 1 / (1 + exp(-sum));

				// ############################################

				out.push_back(sum);
				//std::cout << " Sum             is: " << sum;
				sum = 0;

			}      // j
			output_layers.push_back(out); // Each layer output is stored
			inp.clear();
			inp = out;
			out.clear();
		}     // k

		double sum_exp = 0;
		double err = 0;

		int j_max = 1000;
		int jmax_index = -1;
		for (int j = 0; j < L[L.size() - 1]; ++j)
		{

			//####### Finding which node out is larger ##########
			if (inp[j] < j_max)
			{
				j_max = inp[j];
				jmax_index = j;

			}
			//############

			sum_exp += exp(inp[j]);


		}

		if (tar[jmax_index] == 1)
		{

			accuracy_vector.push_back(1);

		}
		else
		{
			accuracy_vector.push_back(0);

		}


		// ###################### Calculating Error ###################################

		double avgerror = 0;
		avgerror = squaredError(inp, tar, error);

		// ####################################################################

		/*
		std::vector<double> new_inp;
		for (int j = 0; j < L[2]; ++j)
		{

			new_inp.push_back(exp(inp[j])/sum_exp);
			//err = tar[j] - new_inp[j];
			err = tar[j] - inp[j];
			error.push_back(0.5*pow(err, 2));
			avgerror = avgerror + error[j];

		}
		*/

		avg_avg_error.push_back(avgerror);


	} // forward pass

	void xnet::write_parameters()  // Save parameters of Feef forward layer
	{


		// ########### Designing XML data for networkm configuration ############


		std::ofstream myfile;
		myfile.open("param_test_feedforward.sxy");




		myfile << "<network>\n";


		for (int k = 0; k < (L.size() - 1); ++k)
		{

			myfile << "\t<Layer>\n";

			// ############# Weigth parameters #############
			myfile << "\t\t<Weights>\n";
			for (int j = 0; j < L[k + 1]; ++j)            //second layer Second vector place
			{

				myfile << "\t\t";

				for (int i = 0; i < L[k]; ++i)                  //  First layer First vector place
				{

					myfile << W[k][j][i] << ",";


				}

				myfile << "\n";

			}
			myfile << "\t\t</Weights>\n";
			// #############################################

			// ############# Bias parameters #############
			myfile << "\t\t<Bias>\n";
			myfile << "\t\t";
			for (int j = 0; j < L[k + 1]; ++j)            //second layer Second vector place
			{

				myfile << bias_net[k][j] << ",";


			}
			myfile << "\n";
			myfile << "\t\t</Bias>\n";
			myfile << "\t</Layer>\n";

		}


		myfile << "</network>\n";
		myfile.close();


	}

	void xnet::load_parameters_feed_forward()
	{

		std::vector<double> was;
		std::vector<std::vector<double>> wass;

		std::string line;
		std::ifstream myfile("param_test_feedforward.sxy");

		int line_count = 0;
		int f0, f1, f2, lgh;
		std::string str, str2;


		if (myfile.is_open())
		{

			getline(myfile, line); // <network>
			std::cout << "first read" << line;

			while (getline(myfile, line))
			{
				std::cout << line;

				if (line == "\t<Layer>")
				{

					std::cout << "New layer " << "\n";

					getline(myfile, line);  // <Weights>
					getline(myfile, line);   // First line
					while (line != "\t\t</Weights>")
					{


						// ################## Agent name extraction ##############
						std::cout << "New line " << "\n";

						while (line.length())
						{

							f0 = line.find(",");
							str = line.substr(0, f0);
							line = line.substr(f0 + 1, line.length());
							std::cout << "parameter  : " << str << "\n";
							//std::cout << "first parameter  : " << stof(str) << "\n";

							was.push_back(stof(str));



						}

						wass.push_back(was);
						was.clear();

						getline(myfile, line);


					}

					W.push_back(wass);
					wass.clear();
					//std::cout << "first parameter  : " << str2 << "\n";
					//name_agent = str;

					getline(myfile, line); // <Bias>
					getline(myfile, line);   // First line in Bias

					while (line != "\t\t</Bias>")
					{


						
						std::cout << "New line " << "\n";

						while (line.length())
						{

							f0 = line.find(",");
							str = line.substr(0, f0);
							line = line.substr(f0 + 1, line.length());
							std::cout << "parameter  : " << str << "\n";
							
							bias_layer.push_back(stof(str));


						}


						bias_net.push_back(bias_layer);
						bias_layer.clear();
						

						getline(myfile, line);


					}

					getline(myfile, line); // </Layer>
					

				}  // End of layer

			}  // End of file



		}  // If file is opened


	}



}
	
//---------------------------------------
