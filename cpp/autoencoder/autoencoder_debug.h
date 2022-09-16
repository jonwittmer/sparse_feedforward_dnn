#pragma once
#include "autoencoder/autoencoder.h"
#include "autoencoder/compressed_batch.h"
#include "sparse/sparse_model.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	class AutoencoderDebug : public Autoencoder {
	public:
		AutoencoderDebug(const std::string encoderPath, const std::string decoderPath, int dataSize,
					int mpirank, bool shouldWrite, bool debug);
		
		virtual void compressStates(const std::vector<std::vector<double>> &dataBuffer, int startingTimestep, int currBatchSize) override;
		virtual std::pair<int, int> prefetchDecompressedStates(std::vector<std::vector<double>> &dataBuffer,
													   const int latestTimestep) override;
		
	protected:
		void writeDataToFile(const std::vector<std::vector<double>>& data) const;
		bool shouldWrite_;

	private:
		std::vector<CompressedBatch<Eigen::MatrixXd>> compressedStates_;
		CompressedBatch<Eigen::MatrixXd>& getBatchStorage(const int startingTimestep, const int endingTimestep);
	};
}
