#include "sparse_model.h"
#include "sparse_layer.h"

#include <vector>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

namespace sparse_nn {
	SparseModel::SparseModel(const std::string configFilename) {
		// parse json file to get list of filenames
		std::ifstream f(configFilename);
		json allLayersInfo = json::parse(f);

		// get directory of model
		std::string basePath = "";
		const size_t lastSlash = configFilename.rfind('/');
		if (std::string::npos != lastSlash) {
			basePath += configFilename.substr(0, lastSlash) + "/";
		}
		std::cout << basePath << std::endl;
		
		for (const auto& currLayerInfo : allLayersInfo["layers"]) {
			const ModelInfo info = currLayerInfo.get<ModelInfo>();
			layers.emplace_back(SparseLayer());
			layers.back().loadWeightsAndBiases(basePath + info.weightsFilename, basePath + info.biasesFilename);
			layers.back().setActivationFunction(info.activation);
		}
	}

	Eigen::MatrixXf SparseModel::run(const Eigen::MatrixXf& input) {
		const Eigen::MatrixXf* output = &input;
		for (auto& layer : layers) {
			output = layer.run(*output);
		}
		return *output;
	}

}
