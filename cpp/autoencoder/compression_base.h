#pragma once

#include "batch_preparation/batch_preparer.h"
#include "batch_preparation/compressed_batch.h"
#include "sparse/sparse_model.h"

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	class CompressionBase {
	public:
		CompressionBase(const std::string encoderPath, const std::string decoderPath, int dataSize, int nStates,
                    int mpirank, bool debug);
		
		virtual void compressStates(const std::vector<Timestep> &dataBuffer, int startingTimestep, int currBatchSize) {};
		virtual std::pair<int, int> prefetchDecompressedStates(std::vector<Timestep> &dataBuffer,
                                                           const int latestTimestep) {};
    virtual void compressStates(const double* dataBuffer, int startingTimestep, int currBatchSize, int nLocalElements){};
		virtual std::pair<int, int> prefetchDecompressedStates(double *dataBuffer,
                                                           const int latestTimestep, int nLocalElements){};
		
	protected:
    std::string encoderPath_;
    std::string decoderPath_;
    Eigen::MatrixXd batchDataMatrix_;
		
		bool debugMode_ = false;
		bool shouldWrite_ = false;

		// specific to mangll seismic inversion problem
		int nStates_ = 36; // 9 variables, 4 states per variable
		int dataSize_;
		int mpirank_;

    std::unique_ptr<BatchPreparer> batchPreparer_;

		std::vector<CompressedBatch<Eigen::MatrixXf>> compressedStates_;
		CompressedBatch<Eigen::MatrixXf>& getBatchStorage(const int startingTimestep, const int endingTimestep);
	};
}
