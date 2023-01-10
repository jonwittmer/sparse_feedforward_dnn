#pragma once
#include "sparse/layer.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

namespace sparse_nn {	
	class DenseLayer : public Layer {
	public:
		DenseLayer() = default;

		virtual void initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList,
												const std::vector<float>& bias,
												const std::vector<size_t>& matrixDims) override;
		
		// activation(Ax + b)
		// returns a pointer to const outputMat_. 
		virtual const Eigen::MatrixXf* run(const Eigen::MatrixXf& inputMat) override;

		virtual void print() const override;
		
	protected:
    Eigen::MatrixXf denseMatStorage_;
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned32> denseMat_{nullptr, 0, 0};
	};
} // namespace sparse_nn
