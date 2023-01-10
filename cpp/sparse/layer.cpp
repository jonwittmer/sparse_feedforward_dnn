#include "sparse/layer.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <mpi.h>

namespace sparse_nn {  	
	void Layer::loadWeightsAndBiases(const std::string weightsFilename, const std::string biasFilename,
										   const std::vector<size_t>& matrixDims) {
    MPI_Comm_rank(*nodalComm_, &localRank_);

    std::vector<Eigen::Triplet<float>> tripletList;
    std::vector<float> bias;
    if (localRank_ == 0) {
      tripletList = loadWeightsFromCsv(weightsFilename);
      bias = loadBiasesFromCsv(biasFilename);
      // check dimensions match - TF computes x.T @ matrix + bias, so matrixDims[1] is correct bias size
      assert(("Weights and bias do not have matching dimensions", matrixDims[1] == bias.size()));
    }
		
		initializeWeightsAndBiases(tripletList, bias, matrixDims);
	}

	void Layer::setActivationFunction(const std::string activation) {
		activation_ = activation;
	}

	std::vector<Eigen::Triplet<float>> Layer::loadWeightsFromCsv(const std::string filename) const {
		std::vector<Eigen::Triplet<float>> tripletList;

		std::fstream weightsFile;
		weightsFile.open(filename, std::ios::in);

		// actually store A^T for efficiency in multiplying. 
		// since C++ is row-major order, we want to store each element of a
		// batch in the rows, not the columns
		int i, j;
		float val;
		std::string item;
		while (std::getline(weightsFile, item, ',')) {
			i = std::stoi(item);
			
			std::getline(weightsFile, item, ',');
			j = std::stoi(item);
			
			std::getline(weightsFile, item);
			val = std::stof(item);

			tripletList.emplace_back(i, j, val);
		}
		weightsFile.close();
		
		return tripletList;
	}
	
	std::vector<float> Layer::loadBiasesFromCsv(const std::string filename) const {
		std::fstream biasFile;
		std::vector<float> bias;
		std::string item;
		biasFile.open(filename, std::ios::in);
		while (std::getline(biasFile, item, ',')) {
			bias.push_back(std::stof(item));
		}
		return bias;
	}
	
	void Layer::allocateOutputMat(const size_t batchSize) {
	    assert(("weights and biases must be initialized before calling allocateOutputMat", initialized_));
		
		if (batchSize <= reservedBatchSize_) {
			return;
		}

		outputMat_.resize(batchSize, bias_.rows());
		reservedBatchSize_ = batchSize;
	}
}
