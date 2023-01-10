#include "sparse/layer.h"
#include "sparse/sparse_layer.h"

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
	//SparseLayer::SparseLayer(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
  //                         const std::vector<size_t>& matrixDims, const std::string activation="none") {
	//	activation_ = activation;
	//	initializeWeightsAndBiases(tripletList, bias, matrixDims);
	//}

  void SparseLayer::initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
                                               const std::vector<float>& bias,
                                               const std::vector<size_t>& matrixDims) {
    int biasSizeBytes = 0;
    int weightsSizeBytes = 0;
    int indexSizeBytes = 0;
    Eigen::SparseMatrix<float> sparseMatStorage;
    if (localRank_ == 0) {
      sparseMatStorage.resize(matrixDims[0], matrixDims[1]); // transpose is embedded in TF weights matrix
      sparseMatStorage.setFromTriplets(tripletList.begin(), tripletList.end());
      sparseMatStorage.makeCompressed();
      
      weightsSizeBytes = sparseMatStorage.nonZeros() * sizeof(float);
      indexSizeBytes = (sparseMatStorage.nonZeros() + sparseMatStorage.innerSize()) * sizeof(int);
      biasSizeBytes = bias.size() * sizeof(float);
    }

    // allocate shared memory - only rank 0 actually allocates anything
    MPI_Win_allocate_shared(weightsSizeBytes, sizeof(float), MPI_INFO_NULL, *nodalComm_, &weightsPtr_, &weightsWindow_);
    MPI_Win_allocate_shared(indexSizeBytes, sizeof(int), MPI_INFO_NULL, *nodalComm_, &innerIndexPtr_, &indexWindow_);
    MPI_Win_allocate_shared(biasSizeBytes, sizeof(float), MPI_INFO_NULL, *nodalComm_, &biasPtr_, &biasWindow_);

    std::cout << "windows allocated" << std::endl;
    
    // get the pointer in this process' memory space to shared memory
    MPI_Aint weightsSize;
    MPI_Aint indexSize;
    MPI_Aint biasSize;
    int disp_unit;
    MPI_Win_shared_query(weightsWindow_, 0, &weightsSize, &disp_unit, &weightsPtr_);
    MPI_Win_shared_query(indexWindow_, 0, &indexSize, &disp_unit, &innerIndexPtr_);
    MPI_Win_shared_query(biasWindow_, 0, &biasSize, &disp_unit, &biasPtr_);
    int nonZeros = weightsSize / disp_unit;
    outerIndexPtr_ = innerIndexPtr_ + nonZeros; // weightsSize / disp_unit should be nnz
    
    std::cout << "pointers communicated" << std::endl;

    // copy data to shared buffers
    if (localRank_ == 0) {
      std::copy(sparseMatStorage.valuePtr(), sparseMatStorage.valuePtr() + sparseMatStorage.nonZeros(), weightsPtr_);
      std::copy(sparseMatStorage.innerIndexPtr(), sparseMatStorage.innerIndexPtr() + sparseMatStorage.nonZeros(), innerIndexPtr_);
      std::copy(sparseMatStorage.outerIndexPtr(), sparseMatStorage.outerIndexPtr() + sparseMatStorage.innerSize(), outerIndexPtr_);
      std::copy(bias.begin(), bias.end(), biasPtr_);
    }
    
    // synchronize before creating maps
    MPI_Barrier(*nodalComm_);

    // associate map variable with data
    new (&bias_) Eigen::Map<Eigen::VectorXf, Eigen::Aligned32>(biasPtr_, matrixDims[1]);
    new (&sparseMat_) Eigen::Map<Eigen::SparseMatrix<float>>(matrixDims[0],
                                                             matrixDims[1],
                                                             nonZeros,
                                                             outerIndexPtr_,
                                                             innerIndexPtr_,
                                                             weightsPtr_);
   
		activationMap_ = defineActivationFunctions();

		initialized_ = true;
	}

	const Eigen::MatrixXf* SparseLayer::run(const Eigen::MatrixXf& inputMat) {
    assert(("Matrix dimension mismatch", inputMat.cols() == sparseMat_.rows()));
		allocateOutputMat(inputMat.rows());

		// activation(Ax + b) 
		outputMat_ = ((inputMat * sparseMat_).rowwise() + bias_.transpose());
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
