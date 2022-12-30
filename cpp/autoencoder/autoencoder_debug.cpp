#include "autoencoder/autoencoder_debug.h"
#include "autoencoder/batch_preparer.h"
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
																		 int nStates, int mpirank, bool shouldWrite, double writeProbability, bool debug) :
		Autoencoder(encoderPath, decoderPath, dataSize, nStates, mpirank, debug),
		shouldWrite_(shouldWrite), 
    writeProbability_(writeProbability)	{}
	
	void AutoencoderDebug::compressStates(const std::vector<Timestep> &dataBuffer,
																	 int startingTimestep, int currBatchSize) {
		if (debugMode_ && mpirank_ == 0) {
			std::cout << "[COMPRESS] Storing timesteps " << startingTimestep << "-" << startingTimestep + currBatchSize - 1 << std::endl;
		}
    
    batchPreparer_->copyVectorToMatrix(batchDataMatrix_, dataBuffer);

		if (shouldWrite_) {
			writeDataToFile();
		}
		
		CompressedBatch<Eigen::MatrixXd>& batchStorage = getBatchStorage(startingTimestep, startingTimestep + currBatchSize - 1);
		batchStorage.data = batchDataMatrix_;
		return;
	}
	
	std::pair<int, int> AutoencoderDebug::prefetchDecompressedStates(std::vector<Timestep>& dataBuffer,
																const int latestTimestep) {
		// in decompression, we shouldn't be creating any new batches, so only one timestep is needed to
		// decompress batch
		if (debugMode_ && mpirank_ == 0) {
			std::cout << "[DECOMPRESS] Fetching timestep " << latestTimestep << std::endl;
		}
    
		CompressedBatch<Eigen::MatrixXd>& batchStorage = getBatchStorage(latestTimestep, latestTimestep);
		batchDataMatrix_ = batchStorage.data;
    if (batchDataMatrix_.rows() == 0 || batchDataMatrix_.cols() == 0) {
      std::cout << "rank " << mpirank_ << " stored data has 0 shape at timestep " << latestTimestep << std::endl;
    }
		batchPreparer_->copyMatrixToVector(batchDataMatrix_, dataBuffer);
		
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
	
	void AutoencoderDebug::writeDataToFile() const {
    // open up file to write data to
		std::string filename = "data_storage_rank_" + std::to_string(mpirank_) + ".bin";
		std::ofstream myFile(filename, std::ios::binary | std::ios::app);
		if (!myFile.is_open()) {
			std::cout << "Failure to open file " << filename << std::endl;
			return;
		}

    int nRows = batchDataMatrix_.rows();
    auto singleVector = std::vector<double>(fullDimension_, 0);
		for (int i = 0; i < nRows; ++i) {
			// convert probability into integer for sampling
			// see https://www.cplusplus.com/refrence/cstdlib/rand/
			int integer_scaling = static_cast<int>(1. / writeProbability_);
			int sample = rand() % integer_scaling;
			if (sample == 0) {
        // not sure if this copy is needed, but interfacing with the raw storage is 
        // awkward in Eigen
        for (int j = 0; j < fullDimension_; ++j) {
          singleVector.at(j) = batchDataMatrix_(i, j);
        }

        // write out to file
				myFile.write(reinterpret_cast<const char*>(&(singleVector.at(0))),
							 std::streamsize(fullDimension_ * sizeof(double)));
			}
		}
		myFile.close();
		return;
	}
} // namespace sparse_nn
