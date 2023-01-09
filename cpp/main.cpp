#include "sparse/sparse_model.h"
#include "sparse/sparse_layer.h"
#include "utils/timer.h"
#include "batch_preparation/batch_preparer.h"
#include "autoencoder/autoencoder_interface_c.h"

#include <iostream>
#include <Eigen/Core>
#include <omp.h>
#include <mpi.h>
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

void timeEncoder(bool printResults) {
  int commSize;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  // parameters that determine the size of the data

  int nTimesteps = 16;
  int nStates = 9;
  int nDofsPerElement = 64;
  int nRkStages = 4;
  int nElements = 112 * 2 / commSize;
  sparse_nn::SparseModel encoder;
  sparse_nn::SparseModel decoder;
  
  encoder = sparse_nn::SparseModel("/work/06537/jwittmer/ls6/trained_models/time_rk/train_114/sparse_encoder/config.json");
  Eigen::MatrixXd randomDataD = Eigen::MatrixXd::Random(nElements * nStates, nTimesteps * nRkStages * nDofsPerElement);
  //std::cout << "Encoder random data: (" << randomData.rows() << ", " << randomData.cols() << ")" << std::endl;
  Eigen::MatrixXf randomData = randomDataD.cast<float>();

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < 1; ++i) {
  sparse_nn::Timer encTimer("Encoder timer");
  encTimer.start();
  Eigen::MatrixXf output = encoder.run(randomData);
  encTimer.stop();
  if (printResults) { encTimer.print(); }
  
  decoder = sparse_nn::SparseModel("/work/06537/jwittmer/ls6/trained_models/time_rk/train_114/sparse_decoder/config.json");
  MPI_Barrier(MPI_COMM_WORLD);
  sparse_nn::Timer decTimer("Decoder timer");
  decTimer.start();
  Eigen::MatrixXf outputDec = decoder.run(output);
  decTimer.stop();
  if (printResults) { decTimer.print(); }
  }
}

int main() {
  MPI_Init(NULL, NULL);
	// default constructor
	//sparse_nn::SparseLayer sl;

	// load from file
	//sl.loadWeightsAndBiases("../models/weights_0.csv", "../models/biases_0.csv", {10, 8});
	//sl.setActivationFunction("elu");
	//sl.print();

	//Eigen::MatrixXf inputMat = Eigen::MatrixXf::Random(10, 10);
	//std::cout << inputMat << "\n" << std::endl;
	//auto outputMat = sl.run(inputMat);
	//std::cout << *outputMat << std::endl;
	//std::cout << std::endl;

	//std::cout << "Loading model!" << std::endl;
	//loadModel();

  //std::cout << std::endl;
  //timeBatchPreparation();

  // initialize thread pool
  //omp_set_num_threads(56);
  //#pragma omp parallel for
  for(int i = 0; i < 1; ++i) {
    timeEncoder(false);
  }

  // do the timing that matters
  //#pragma omp parallel for
  //sparse_nn::Timer totalTimer("Time to do all work");
  //totalTimer.start();
  for(int i = 0; i < 1; ++i) {
    timeEncoder(true);
  }
  //totalTimer.stop();
  //totalTimer.print();

  MPI_Finalize();
	return 0;
}
