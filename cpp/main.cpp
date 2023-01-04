#include "sparse/sparse_model.h"
#include "sparse/sparse_layer.h"
#include "utils/timer.h"
#include "batch_preparation/batch_preparer.h"
#include "autoencoder/autoencoder_interface_c.h"

#include <iostream>
#include <Eigen/Core>
#include <stdlib.h>

void loadModel() {
	std::string modelConfig = "../models/config.json";
	sparse_nn::SparseModel sm(modelConfig);
	Eigen::MatrixXf inputMat = Eigen::MatrixXf::Random(10, 10);
	std::cout << inputMat << "\n" << std::endl;
	auto outputMat = sm.run(inputMat);
	std::cout << outputMat << std::endl;
	std::cout << std::endl;
}

void timeBatchPreparation() {
  // parameters that determine the size of the data
  int nTimesteps = 16;
  int nStates = 9;
  int nDofsPerElement = 64;
  int nRkStages = 4;
  int nElements = 300;
  int dataSize = nElements * nDofsPerElement;
  std::vector<std::vector<sparse_nn::Timestep>> buffers;
  
  
  for (int i = 0; i < 40; ++i) {
  buffers.emplace_back(nTimesteps, 
      sparse_nn::Timestep(nStates * nRkStages, 
          sparse_nn::FullState(dataSize, 0)));
  }

  // fill buffer with random doubles
  for (auto& buffer: buffers) {
  for (auto& timestep : buffer) {
    for (auto& state : timestep) {
      for (auto& val : state) {
        val = (double)rand() / (double)rand();
      }
    }
  }
  }
  auto prep = sparse_nn::TimeBatchPreparer(nDofsPerElement, nTimesteps, nStates, nRkStages);
  Eigen::MatrixXd mat;

  for (int i = 0; i < 40; ++i) {
    sparse_nn::Timer copyToMatrixTimer("Copy to matrix");
    copyToMatrixTimer.start();
    prep.copyVectorToMatrix(mat, buffers[i]);
    copyToMatrixTimer.stop();
    copyToMatrixTimer.print();
    
    sparse_nn::Timer copyToVectorTimer("Copy to vector");
    copyToVectorTimer.start();
    prep.copyMatrixToVector(mat, buffers[39-i]);
    copyToVectorTimer.stop();
    copyToVectorTimer.print();
    std::cout << std::endl;
  }
}


int main() {
	// default constructor
	sparse_nn::SparseLayer sl;

	// load from file
	sl.loadWeightsAndBiases("../models/weights_0.csv", "../models/biases_0.csv", {10, 8});
	sl.setActivationFunction("elu");
	sl.print();

	Eigen::MatrixXf inputMat = Eigen::MatrixXf::Random(10, 10);
	std::cout << inputMat << "\n" << std::endl;
	auto outputMat = sl.run(inputMat);
	std::cout << *outputMat << std::endl;
	std::cout << std::endl;

	std::cout << "Loading model!" << std::endl;
	loadModel();

  std::cout << std::endl;
  timeBatchPreparation();
	
	return 0;
}
