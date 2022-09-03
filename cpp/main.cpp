#include "sparse_model.h"
#include "sparse_layer.h"

#include <iostream>
#include <Eigen/Core>

void loadModel() {
	std::string modelConfig = "../models/config.json";
	sparse_nn::SparseModel sm(modelConfig);
	Eigen::MatrixXf inputMat = Eigen::MatrixXf::Random(10, 10);
	std::cout << inputMat << "\n" << std::endl;
	auto outputMat = sm.run(inputMat);
	std::cout << outputMat << std::endl;
	std::cout << std::endl;
	
}

int main() {
	// default constructor
	sparse_nn::SparseLayer sl;

	// load from file
	sl.loadWeightsAndBiases("../models/weights_0.csv", "../models/biases_0.csv", {8, 10});
	sl.setActivationFunction("elu");
	sl.print();

	Eigen::MatrixXf inputMat = Eigen::MatrixXf::Random(10, 10);
	std::cout << inputMat << "\n" << std::endl;
	auto outputMat = sl.run(inputMat);
	std::cout << *outputMat << std::endl;
	std::cout << std::endl;

	std::cout << "Loading model!" << std::endl;
	loadModel();
	
	return 0;
}
