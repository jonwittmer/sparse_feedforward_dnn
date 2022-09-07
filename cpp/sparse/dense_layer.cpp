#include "dense_layer.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#define SPARSE_NN_DEBUG 1

namespace sparse_nn {  
    void DenseLayer::initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
												 const std::vector<float>& bias,
												 const std::vector<size_t>& matrixDims) {
		// since we have the matrix data in COO format, create sparse tensor first then
		// convert to dense for convenience
		Eigen::SparseMatrix<float> sparseMat;
		sparseMat.resize(matrixDims[0], matrixDims[1]);
		sparseMat.setFromTriplets(tripletList.begin(), tripletList.end());
		denseMat_ = Eigen::MatrixXf(sparseMat);
		bias_.resize(bias.size());
		std::vector<float> biasCopy(bias);
		bias_ = Eigen::Map<Eigen::VectorXf>(biasCopy.data(), biasCopy.size());

		activationMap_ = defineActivationFunctions();
		
		initialized_ = true;
	}

	const Eigen::MatrixXf* DenseLayer::run(const Eigen::MatrixXf& inputMat) {
		assert(("Matrix dimension mismatch", inputMat.cols() == denseMat_.rows()));
		allocateOutputMat(inputMat.rows());
 
		// activation(Ax + b) 
		outputMat_.noalias() = ((inputMat * denseMat_).rowwise() + bias_.transpose()).unaryExpr(activationMap_[activation_]);
		return &outputMat_;
	}
	
	void DenseLayer::print() const {
		Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
		std::cout << "denseMat_:" << std::endl;
		std::cout << "  (" << denseMat_.rows() << ", " << denseMat_.cols() << ")" << std::endl;
		std::cout << denseMat_.format(CleanFmt) << std::endl;
		std::cout << std::endl;

		std::cout << "bias_" << std::endl;
		std::cout << "  (" << bias_.rows() << ", 1)" << std::endl;
		std::cout << bias_.format(CleanFmt) << std::endl;
	}
} // namespace sparse_nn
