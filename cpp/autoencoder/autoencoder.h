#pragma once

#include "autoencoder/compressed_batch.h"
#include "sparse/sparse_model.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	class Autoencoder {
	public:
		Autoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,
					int mpirank, bool debug);
		
		virtual void compressStates(const std::vector<std::vector<double>> &dataBuffer, int startingTimestep, int currBatchSize);
		virtual std::pair<int, int> prefetchDecompressedStates(std::vector<std::vector<double>> &dataBuffer,
													   const int latestTimestep);
		
	protected:
		SparseModel encoder_;
		SparseModel decoder_;
		Eigen::MatrixXd batchDataMatrix_;
		
		bool debugMode_ = false;
		bool shouldWrite_ = false;

		// specific to mangll seismic inversion problem
		int nStates_ = 36; // 9 variables, 4 states per variable
		int dataSize_;
		int mpirank_;

		void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<std::vector<double>>& dataBuffer);
		void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<std::vector<double>>& dataBuffer);

	private:
		std::vector<CompressedBatch<Eigen::MatrixXf>> compressedStates_;
		CompressedBatch<Eigen::MatrixXf>& getBatchStorage(const int startingTimestep, const int endingTimestep);
	};
}
