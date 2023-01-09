#include "autoencoder/compression_base.h"
#include "batch_preparation/compressed_batch.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <Eigen/Core>

namespace sparse_nn {
	CompressionBase::CompressionBase(const std::string encoderPath, const std::string decoderPath, int dataSize,
                                   int nStates, int mpirank, bool debug) :
		encoderPath_(encoderPath),
    decoderPath_(decoderPath),
    dataSize_(dataSize),
    nStates_(nStates),
		mpirank_(mpirank),
		debugMode_(debug)	{}
	
	CompressedBatch<Eigen::MatrixXf>& CompressionBase::getBatchStorage(const int startingTimestep, const int endingTimestep) {
		// this function as written is pretty brittle as handling for batches that cross
		// muliple CompressedBatch objects is not detected or supported
		for (auto& batch : compressedStates_) {
			if (batch.isTimestepInBatch(startingTimestep)) {
				return batch;
			}
		}
		
		// create new CompressedBatch since startingTimestep is not already in compressedStates_
		compressedStates_.emplace_back(startingTimestep, endingTimestep);
		return compressedStates_.back();
	}
} // namespace sparse_nn
