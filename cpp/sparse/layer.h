#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>


namespace sparse_nn {
	inline std::map<std::string, std::function<float(float)>> defineActivationFunctions() {
		std::map<std::string, std::function<float(float)>> activationMap = {
			{"relu", [](float x){ return std::max(x, 0.f); }},
			{"elu", [](float x){ return x > 0.f ? x : std::exp(x) - 1.f; }},
			{"none", [](float x){ return x; }}
		};
		return activationMap;
	}
	
	class Layer {
	public:
    Layer() = default;

		virtual void initializeWeightsAndBiases(const std::vector<Eigen::Triplet<float>>& tripletList, const std::vector<float>& bias,
												const std::vector<size_t>& matrixDims) = 0;
		
		// loads weights and biases from csv files
		void loadWeightsAndBiases(const std::string weightsFilename, const std::string biasFilename,
								  const std::vector<size_t>& matrixDims);
		
		// public setter
		void setActivationFunction(const std::string activation);
		
		// activation(Ax + b)
		// returns a pointer to const outputMat_. 
		virtual const Eigen::MatrixXf* run(const Eigen::MatrixXf& inputMat) = 0;
		
		// print weights and bias matrix
		virtual void print() const = 0;
		
	protected:	
		Eigen::VectorXf bias_store_;
    Eigen::Map<Eigen::VectorXf, Eigen::Aligned32> bias_;
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
