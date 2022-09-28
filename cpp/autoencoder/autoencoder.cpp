#include "autoencoder/autoencoder.h"
#include "autoencoder/compressed_batch.h"
#include "normalization/normalization.h"
#include "sparse/sparse_model.h"
#include "utils/timer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <Eigen/Core>

namespace sparse_nn {
	Autoencoder::Autoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,
													 int nStates, int mpirank, bool debug) :
    CompressionBase(encoderPath, decoderPath, dataSize, nStates, mpirank, debug)
	{
		encoder_ = SparseModel(encoderPath_);
		decoder_ = SparseModel(decoderPath_);
		
		// give batchDataMatrix_ the correct number of columns
		batchDataMatrix_.resize(1, dataSize_);
	}
	
	void Autoencoder::compressStates(const std::vector<std::vector<double>> &dataBuffer,
																	 int startingTimestep, int currBatchSize) {		
		Timer copyTimer("[COMPRESS] copy to matrix");
		copyTimer.start();
		batchPreparer_->copyVectorToMatrix(batchDataMatrix_, dataBuffer);
		copyTimer.stop();
		
		CompressedBatch<Eigen::MatrixXf>& batchStorage = getBatchStorage(startingTimestep, startingTimestep + currBatchSize - 1);
		
		// normalize
		Timer normTimer("[COMPRESS] normalization");
		normTimer.start();
		batchStorage.mins = subtractAndReturnMins(batchDataMatrix_);		
		batchStorage.ranges = divideAndReturnRanges(batchDataMatrix_);
		normTimer.stop();

		// do compression
		Timer compressTimer("[COMPRESS] compression");
		compressTimer.start();
		batchStorage.data = encoder_.run(batchDataMatrix_.cast<float>());
		compressTimer.stop();
		
		if (debugMode_ && (mpirank_ == 0)) {
			copyTimer.print();
			normTimer.print();
			compressTimer.print();
		}
		
    // remove later once code has been verified
    if (debugMode_ && mpirank_ == 0) {
      Eigen::MatrixXf decodedMat = decoder_.run(batchStorage.data);
      //Eigen::MatrixXd unnormalizedMat = unnormalize(decodedMat, batchStorage.mins, batchStorage.ranges);
      double errorNorm = (decodedMat - batchDataMatrix_.cast<float>()).norm();
      double batchNorm = batchDataMatrix_.norm();
      if (batchNorm > 1e-13){
        double relError = errorNorm / batchNorm;
        std::cout << "[COMPRESS] Relative decompression error for batch: " << relError * 100;
        std::cout << "%" << std::endl;
      }
      std::cout << "[COMPRESS] True matrix norm: ";
      std::cout << batchNorm << std::endl;
    }
	}
	
	std::pair<int, int> Autoencoder::prefetchDecompressedStates(std::vector<std::vector<double>>& dataBuffer,
																															const int latestTimestep) {
		// in decompression, we shouldn't be creating any new batches, so only one timestep is needed to
		// decompress batch
		CompressedBatch<Eigen::MatrixXf>& batchStorage = getBatchStorage(latestTimestep, latestTimestep);

		Timer decompTimer("[DECOMPRESS] decompression");
		decompTimer.start();
		Eigen::MatrixXf decompressedBatch = decoder_.run(batchStorage.data);
		decompTimer.stop();
		
		Timer normTimer("[DECOMPRESS] unnormalize");
		normTimer.start();
		batchDataMatrix_ = unnormalize(decompressedBatch, batchStorage.mins, batchStorage.ranges);
		normTimer.stop();
	
		Timer copyTimer("[DECOMPRESS] copy to vector");
		copyTimer.start();
		batchPreparer_->copyMatrixToVector(batchDataMatrix_, dataBuffer);
		copyTimer.stop();

		if (debugMode_ && (mpirank_ == 0)) {
			decompTimer.print();
			normTimer.print();
			copyTimer.print();
		}

		return {batchStorage.getStartingTimestep(), batchStorage.getEndingTimestep()};
	}
} // namespace sparse_nn
