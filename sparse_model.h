#pragma once
#include "sparse_layer.h"

#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace sparse_nn {
	struct ModelInfo {
		std::string weightsFilename;
		std::string biasesFilename;
		std::string activation;
	};
	
	inline void from_json(const json& j, ModelInfo& info) {
		j.at("weights").get_to(info.weightsFilename);
		j.at("biases").get_to(info.biasesFilename);
		j.at("activation").get_to(info.activation);
	}
	
	class SparseModel {
	public:
		SparseModel(const std::string configFilename);
	
		Eigen::MatrixXf run(const Eigen::MatrixXf& input);

	private:
		std::vector<SparseLayer> layers;
	};
} // namespace sparse_nn
