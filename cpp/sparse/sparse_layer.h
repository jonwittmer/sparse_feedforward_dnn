#pragma once

#include "sparse/layer.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>
#include <mpi.h>

namespace sparse_nn {	
	class SparseLayer : public Layer {
	public:
    SparseLayer(MPI_Comm *comm) : Layer(comm) { layerType_ = "sparse"; }

		// initialize from data that is already prepared
		//SparseLayer(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
		//			const std::vector<size_t>& matrixDims, const std::string activation);

		virtual void initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
												const std::vector<float>& bias,
												const std::vector<size_t>& matrixDims) override;
		
		// activation(Ax + b)
		// returns a pointer to const outputMat_. 
		virtual const Eigen::MatrixXf* run(const Eigen::MatrixXf& inputMat) override;

		virtual void print() const override;
		
	protected:
    int *innerIndexPtr_;
    int *outerIndexPtr_;
    MPI_Win indexWindow_;
    Eigen::SparseMatrix<float> sparseMatStorage_;
    Eigen::Map<Eigen::SparseMatrix<float>> sparseMat_{0, 0, 0, nullptr, nullptr, nullptr, nullptr};
	};
} // namespace sparse_nn
