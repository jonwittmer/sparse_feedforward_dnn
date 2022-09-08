#include "sparse/sparse_layer.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#define SPARSE_NN_DEBUG 1

namespace sparse_nn {  
	SparseLayer::SparseLayer(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
							 const std::vector<size_t>& matrixDims, const std::string activation="none") {
		activation_ = activation;
		initializeWeightsAndBiases(tripletList, bias, matrixDims);
	}

    void SparseLayer::initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
												 const std::vector<float>& bias,
												 const std::vector<size_t>& matrixDims) {
		sparseMat_.resize(matrixDims[0], matrixDims[1]); // transpose is embedded in TF weights matrix
		sparseMat_.setFromTriplets(tripletList.begin(), tripletList.end());
		bias_.resize(bias.size());
		std::vector<float> biasCopy(bias);
		bias_ = Eigen::Map<Eigen::VectorXf>(biasCopy.data(), biasCopy.size());

		activationMap_ = defineActivationFunctions();
		
		initialized_ = true;
	}

	const Eigen::MatrixXf* SparseLayer::run(const Eigen::MatrixXf& inputMat) {
		assert(("Matrix dimension mismatch", inputMat.cols() == sparseMat_.rows()));
		allocateOutputMat(inputMat.rows());
 
		// activation(Ax + b) 
		outputMat_.noalias() = ((inputMat * sparseMat_).rowwise() + bias_.transpose());
		outputMat_.noalias() = outputMat_.unaryExpr(activationMap_[activation_]);
		return &outputMat_;
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
} // namespace sparse_nn
