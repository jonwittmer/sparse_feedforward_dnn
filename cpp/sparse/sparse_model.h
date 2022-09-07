#pragma once
#include "sparse/sparse_layer.h"
#include "sparse/dense_layer.h"

#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace sparse_nn {
	struct ModelInfo {
		std::string weightsFilename;
		std::string biasesFilename;
		std::string activation;
		std::vector<size_t> dimension;
		std::string layerType;
	};
	
	inline void from_json(const json& j, ModelInfo& info) {
		j.at("weights").get_to(info.weightsFilename);
		j.at("biases").get_to(info.biasesFilename);
		j.at("activation").get_to(info.activation);
		j.at("dimension").get_to(info.dimension);
		j.at("type").get_to(info.layerType);
	}
	
	class SparseModel {
	public:
		SparseModel() = default;
		SparseModel(const std::string configFilename);
	
		Eigen::MatrixXf run(const Eigen::MatrixXf& input);

	private:
		std::vector<std::unique_ptr<Layer>> layers;
	};
} // namespace sparse_nn
