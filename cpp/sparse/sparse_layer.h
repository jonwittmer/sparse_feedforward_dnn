#pragma once
#include "sparse/layer.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

namespace sparse_nn {	
	class SparseLayer : public Layer {
	public:
		SparseLayer() = default;

		// initialize from data that is already prepared
		SparseLayer(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
					const std::vector<size_t>& matrixDims, const std::string activation);

		virtual void initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
												const std::vector<float>& bias,
												const std::vector<size_t>& matrixDims) override;
		
		// activation(Ax + b)
		// returns a pointer to const outputMat_. 
		virtual const Eigen::MatrixXf* run(const Eigen::MatrixXf& inputMat) override;

		virtual void print() const override;
		
	protected:
		Eigen::SparseMatrix<float> sparseMat_;
	};
} // namespace sparse_nn
