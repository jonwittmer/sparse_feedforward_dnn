#include "sparse/sparse_model.h"
#include "sparse/sparse_layer.h"
#include "utils/timer.h"
#include "batch_preparation/batch_preparer.h"
#include "autoencoder/autoencoder_interface_c.h"

#include <iostream>
#include <Eigen/Core>
#include <omp.h>
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
  int nElements = 98;
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

void timeEncoder() {
  // parameters that determine the size of the data
  int nTimesteps = 16;
  int nStates = 9;
  int nDofsPerElement = 64;
  int nRkStages = 4;
  int nElements = 98;
  int latentDime = 64;
  sparse_nn::SparseModel encoder;
  sparse_nn::SparseModel decoder;
  
  encoder = sparse_nn::SparseModel("/work/06537/jwittmer/ls6/trained_models/time_rk/train_114/sparse_encoder/config.json");
  Eigen::MatrixXf randomData = Eigen::MatrixXf::Random(nElements * nStates, nTimesteps * nRkStages * nDofsPerElement);
  std::cout << "Encoder random data: (" << randomData.rows() << ", " << randomData.cols() << ")" << std::endl;

  for (int i = 0; i < 2; ++i) {
  sparse_nn::Timer encTimer("Encoder timer");
  encTimer.start();
  Eigen::MatrixXf output = encoder.run(randomData);
  encTimer.stop();
  encTimer.print();

  decoder = sparse_nn::SparseModel("/work/06537/jwittmer/ls6/trained_models/time_rk/train_114/sparse_decoder/config.json");
  sparse_nn::Timer decTimer("Decoder timer");
  decTimer.start();
  Eigen::MatrixXf outputDec = decoder.run(output);
  decTimer.stop();
  decTimer.print();
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


  omp_set_num_threads(56);
  std::cout << std::endl;
  #pragma omp parallel for
  for(int i = 0; i < 56; ++i) {
    timeEncoder();
  }

	return 0;
}
