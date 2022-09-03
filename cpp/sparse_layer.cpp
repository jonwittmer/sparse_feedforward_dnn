#include "sparse_layer.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#define SPARSE_NN_DEBUG 1

namespace sparse_nn {
	namespace {
		std::map<std::string, std::function<float(float)>> defineActivationFunctions() {
			std::map<std::string, std::function<float(float)>> activationMap = {
				{"relu",  [](float x){ return std::max(x, 0.f); }},
				{"elu", [](float x){ return x > 0.f ? x : std::exp(x) - 1.f; }},
				{"none", [](float x){ return x; }}
			};
			return activationMap;
		}
	}

  
	SparseLayer::SparseLayer(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
							 const std::vector<size_t>& matrixDims, const std::string activation="none") {
		activation_ = activation;
		initializeWeightsAndBiases(tripletList, bias, matrixDims);
	}

	void SparseLayer::initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
												 const std::vector<float>& bias,
												 const std::vector<size_t>& matrixDims) {
		sparseMat_.resize(matrixDims[1], matrixDims[0]); // transpose 
		sparseMat_.setFromTriplets(tripletList.begin(), tripletList.end());
		bias_.resize(bias.size());
		std::vector<float> biasCopy(bias);
		bias_ = Eigen::Map<Eigen::VectorXf>(biasCopy.data(), biasCopy.size());

		activationMap_ = defineActivationFunctions();
		
		initialized_ = true;
	}
	
	void SparseLayer::loadWeightsAndBiases(const std::string weightsFilename, const std::string biasFilename,
										   const std::vector<size_t>& matrixDims) {
		std::vector<Eigen::Triplet<float>> tripletList = loadWeightsFromCsv(weightsFilename);
		std::vector<float> bias = loadBiasesFromCsv(biasFilename);
		
		// check dimensions match
		assert(("Weights and bias do not have matching dimensions", matrixDims[0] == bias.size()));
		
		initializeWeightsAndBiases(tripletList, bias, matrixDims);
	}

	const Eigen::MatrixXf* SparseLayer::run(const Eigen::MatrixXf& inputMat) {
		assert(("Matrix dimension mismatch", inputMat.cols() == sparseMat_.rows()));
		allocateOutputMat(inputMat.rows());
 
		// activation(Ax + b) 
		outputMat_.noalias() = ((inputMat * sparseMat_).rowwise() + bias_.transpose()).unaryExpr(activationMap_[activation_]);
		return &outputMat_;
	}

	void SparseLayer::setActivationFunction(const std::string activation) {
		activation_ = activation;
	}
	
	void SparseLayer::print() const {
		Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
		std::cout << "sparseMat_:" << std::endl;
		std::cout << "  (" << sparseMat_.rows() << ", " << sparseMat_.cols() << ")" << std::endl;
		Eigen::MatrixXf dMat;
		dMat = Eigen::MatrixXf(sparseMat_);
		std::cout << dMat.format(CleanFmt) << std::endl;
		std::cout << std::endl;

		std::cout << "bias_" << std::endl;
		std::cout << "  (" << bias_.rows() << ", 1)" << std::endl;
		std::cout << bias_.format(CleanFmt) << std::endl;
	}

	std::vector<Eigen::Triplet<float>> SparseLayer::loadWeightsFromCsv(const std::string filename) const {
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
			j = std::stoi(item);
			
			std::getline(weightsFile, item, ',');
			i = std::stoi(item);
			
			std::getline(weightsFile, item);
			val = std::stof(item);

			tripletList.emplace_back(i, j, val);
		}
		weightsFile.close();
		
		return tripletList;
	}
	
	std::vector<float> SparseLayer::loadBiasesFromCsv(const std::string filename) const {
		std::fstream biasFile;
		std::vector<float> bias;
		std::string item;
		biasFile.open(filename, std::ios::in);
		while (std::getline(biasFile, item, ',')) {
			bias.push_back(std::stof(item));
		}
		return bias;
	}
	
	void SparseLayer::allocateOutputMat(const size_t batchSize) {
	    assert(("weights and biases must be initialized before calling allocateOutputMat", initialized_));
		
		if (batchSize <= reservedBatchSize_) {
			if (SPARSE_NN_DEBUG) {
				std::cout << "output mat is already the right size" << std::endl;
			}
			return;
		}

		if (SPARSE_NN_DEBUG) {
			std::cout << "Resizing output to (" << batchSize << ", " << bias_.rows() << ")" << std::endl;
		}
		outputMat_.resize(batchSize, bias_.rows());
		reservedBatchSize_ = batchSize;
	}
} // namespace sparse_nn
