#include "autoencoder/autoencoder_debug.h"
#include "autoencoder/compressed_batch.h"
#include "normalization/normalization.h"
#include "sparse/sparse_model.h"
#include "utils/timer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	AutoencoderDebug::AutoencoderDebug(const std::string encoderPath, const std::string decoderPath, int dataSize,
																		 int mpirank, bool shouldWrite, bool debug) :
		Autoencoder(encoderPath, decoderPath, dataSize,  mpirank, debug),
		shouldWrite_(shouldWrite)
	{}
	
	void AutoencoderDebug::compressStates(const std::vector<std::vector<double>> &dataBuffer,
																	 int startingTimestep, int currBatchSize) {
		if (debugMode_ && mpirank_ == 0) {
			std::cout << "[COMPRESS] Storing timesteps " << startingTimestep << "-" << startingTimestep + currBatchSize - 1 << std::endl;
		}

		if (shouldWrite_) {
			writeDataToFile(dataBuffer);
		}
		
		copyVectorToMatrix(batchDataMatrix_, dataBuffer);		
		CompressedBatch<Eigen::MatrixXd>& batchStorage = getBatchStorage(startingTimestep, startingTimestep + currBatchSize - 1);
		batchStorage.data = batchDataMatrix_;
		return;
	}
	
	std::pair<int, int> AutoencoderDebug::prefetchDecompressedStates(std::vector<std::vector<double>>& dataBuffer,
																const int latestTimestep) {
		// in decompression, we shouldn't be creating any new batches, so only one timestep is needed to
		// decompress batch
		if (debugMode_ && mpirank_ == 0) {
			std::cout << "[DECOMPRESS] Fetching timestep " << latestTimestep << std::endl;
		}

		CompressedBatch<Eigen::MatrixXd>& batchStorage = getBatchStorage(latestTimestep, latestTimestep);
		batchDataMatrix_ = batchStorage.data;
		copyMatrixToVector(batchDataMatrix_, dataBuffer);
		
		if (debugMode_ && mpirank_ == 0) {
			std::cout << "[DECOMPRESS] Batch has timesteps " << batchStorage.getStartingTimestep();
			std::cout << "-" << batchStorage.getEndingTimestep() << std::endl;
		}

		return {batchStorage.getStartingTimestep(), batchStorage.getEndingTimestep()};
	}

	CompressedBatch<Eigen::MatrixXd>& AutoencoderDebug::getBatchStorage(const int startingTimestep, const int endingTimestep) {
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
	
	void AutoencoderDebug::writeDataToFile(const std::vector<std::vector<double>>& data) const {
		// writing all ofthe data will take up an extremely large amount of storage
		// only write part of it
		double writeProbability = 0.0001;

		std::string filename = "data_storage_rank_" + std::to_string(mpirank_) + ".bin";
		std::ofstream myFile(filename, std::ios::binary | std::ios::app);
		if (!myFile.is_open()) {
			std::cout << "Failure to open file " << filename << std::endl;
			return;
		}

		for (const auto& singleTimestep : data) {
			// convert probability into integer for sampling
			// see https://www.cplusplus.com/refrence/cstdlib/rand/
			int integer_scaling = static_cast<int>(1. / writeProbability);
			int sample = rand() % integer_scaling;
			if (sample == 0) {
				// write all of the states for a single timestep if selected
				myFile.write(reinterpret_cast<const char*>(&(singleTimestep[0])),
							 std::streamsize(nStates_ * dataSize_ * sizeof(double)));
			}
		}
		myFile.close();
		return;
	}
} // namespace sparse_nn
