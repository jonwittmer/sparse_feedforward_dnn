#include "sparse/sparse_model.h"
#include "sparse/layer.h"
#include "sparse/sparse_layer.h"
#include "sparse/dense_layer.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

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
			if (info.layerType == "sparse") {
				layers.emplace_back(std::make_unique<SparseLayer>());
			}
			else if (info.layerType == "dense") {
				layers.emplace_back(std::make_unique<DenseLayer>());
			}
			else {
				std::cout << "layer type " << info.layerType <<  " not recognized. ";
				std::cout << "Choose from [dense, sparse]" << std::endl;
				assert(false);
			}
			layers.back()->loadWeightsAndBiases(basePath + info.weightsFilename,
												basePath + info.biasesFilename,
												info.dimension);
			layers.back()->setActivationFunction(info.activation);
		}
	}

	Eigen::MatrixXf SparseModel::run(const Eigen::MatrixXf& input) {
		const Eigen::MatrixXf* output = &input;
		for (auto& layer : layers) {
			output = layer->run(*output);
		}
		return *output;
	}

}
