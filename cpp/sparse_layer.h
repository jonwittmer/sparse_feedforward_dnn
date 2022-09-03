#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

namespace sparse_nn {	
	class SparseLayer {
	public:
		SparseLayer() = default;

		// initialize from data that is already prepared
		SparseLayer(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
					const std::vector<size_t>& matrixDims, const std::string activation);

		void initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
										const std::vector<size_t>& matrixDims);

		// loads weights and biases from csv files
		void loadWeightsAndBiases(const std::string weightsFilename, const std::string biasFilename,
								  const std::vector<size_t>& matrixDims);

		// public setter
		void setActivationFunction(const std::string activation);
		
		// activation(Ax + b)
		// returns a pointer to const outputMat_. 
		const Eigen::MatrixXf* run(const Eigen::MatrixXf& inputMat);

		// print weights and bias matrix
		void print() const;
		
	private:	
		Eigen::SparseMatrix<float> sparseMat_;
		Eigen::VectorXf bias_;
		std::string activation_ = "none";
		size_t reservedBatchSize_;
		Eigen::MatrixXf outputMat_;
		bool initialized_ = false;

		// activation functions so that we can easily swap activation in unaryExpr call
		std::map<std::string, std::function<float(float)>> activationMap_;

		// Weights are assumed to be in format
		//    i,j,val
		// note there cannot be whitespace in file
		std::vector<Eigen::Triplet<float>> loadWeightsFromCsv(const std::string filename) const;

		// bias file looks like
		// b0,b1,b2,b3
		std::vector<float> loadBiasesFromCsv(const std::string filename) const;
		
		void allocateOutputMat(const size_t batchSize);
	};
} // namespace sparse_nn
