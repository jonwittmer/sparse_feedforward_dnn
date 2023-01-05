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
  double *buffers[40];
  int bufferSize = nTimesteps * nStates * nDofsPerElement * nRkStages * nElements;

  // fill buffer with random doubles
  for (int i = 0; i < 2; ++i) {
    buffers[i] = (double *)malloc(bufferSize * sizeof(double)); 
  }
  for (int j = 0; j < bufferSize; ++j) {
    buffers[0][j] = (double)rand() / (double)rand();
  }

  auto prep = sparse_nn::TimeBatchPreparer(nDofsPerElement, nTimesteps, nStates, nRkStages);
  Eigen::MatrixXd mat;

  sparse_nn::Timer copyToMatrixTimer("Copy to matrix");
  copyToMatrixTimer.start();
  prep.copyVectorToMatrix(mat, buffers[0], nElements);
  copyToMatrixTimer.stop();
  copyToMatrixTimer.print();
    
  sparse_nn::Timer copyToVectorTimer("Copy to vector");
  copyToVectorTimer.start();
  prep.copyMatrixToVector(mat, buffers[1], nElements);
  copyToVectorTimer.stop();
  copyToVectorTimer.print();
  std::cout << std::endl;

  for (int j = 0; j < bufferSize; ++j) {
    if (buffers[0][j] != buffers[1][j]) {
      std::cout << buffers[0][j] << " != " << buffers[1][j] << std::endl;
      break;
    }
  }
  
  for (int i = 0; i < 2; ++i) {
    free(buffers[i]);
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
