#include "sparse/dense_layer.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <mpi.h>

#define SPARSE_NN_DEBUG 1

namespace sparse_nn {  
  void DenseLayer::initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
                                              const std::vector<float>& bias,
                                              const std::vector<size_t>& matrixDims) {
		// since we have the matrix data in COO format, create sparse tensor first then
		// convert to dense for convenience
    Eigen::SparseMatrix<float> sparseMat;
    int biasSizeBytes = 0;
    int weightsSizeBytes = 0;
    Eigen::MatrixXf denseMatStorage;
    if (localRank_ == 0) {
      Eigen::SparseMatrix<float> sparseMatStorage;
      sparseMatStorage.resize(matrixDims[0], matrixDims[1]);
      sparseMatStorage.setFromTriplets(tripletList.begin(), tripletList.end());
      sparseMatStorage.makeCompressed();
      denseMatStorage = Eigen::MatrixXf(sparseMatStorage);
      
      weightsSizeBytes = denseMatStorage.rows() * denseMatStorage.cols() * sizeof(float);
      biasSizeBytes = bias.size() * sizeof(float);
    }

    // allocate shared memory - only rank 0 actually allocates anything
    MPI_Win_allocate_shared(weightsSizeBytes, sizeof(float), MPI_INFO_NULL, *nodalComm_, &weightsPtr_, &weightsWindow_);
    MPI_Win_allocate_shared(biasSizeBytes, sizeof(float), MPI_INFO_NULL, *nodalComm_, &biasPtr_, &biasWindow_);
    
    // get the pointer in this process' memory space to shared memory
    MPI_Aint weightsSize;
    MPI_Aint biasSize;
    int disp_unit;
    MPI_Win_shared_query(weightsWindow_, 0, &weightsSize, &disp_unit, &weightsPtr_);
    MPI_Win_shared_query(biasWindow_, 0, &biasSize, &disp_unit, &biasPtr_);

    // copy data to shared buffers
    if (localRank_ == 0) {
      int dataSize = denseMatStorage.rows() * denseMatStorage.cols();
      std::copy(denseMatStorage.data(), denseMatStorage.data() + dataSize, weightsPtr_);
      std::copy(bias.begin(), bias.end(), biasPtr_);
    }
    
    // synchronize before creating maps
    MPI_Barrier(*nodalComm_);

    // initialize maps
    new (&denseMat_) Eigen::Map<Eigen::MatrixXf, Eigen::Aligned32>(weightsPtr_, 
                                                                   matrixDims[0], 
                                                                   matrixDims[1]);
    new (&bias_) Eigen::Map<Eigen::VectorXf, Eigen::Aligned32>(biasPtr_, matrixDims[1]);

		activationMap_ = defineActivationFunctions();
		initialized_ = true;
	}

	const Eigen::MatrixXf* DenseLayer::run(const Eigen::MatrixXf& inputMat) {
		assert(("Matrix dimension mismatch", inputMat.cols() == denseMat_.rows()));
		allocateOutputMat(inputMat.rows());

		// activation(Ax + b) 
		outputMat_ = ((inputMat * denseMat_).rowwise() + bias_.transpose()).unaryExpr(activationMap_[activation_]);
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
