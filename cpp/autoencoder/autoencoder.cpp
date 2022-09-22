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
													 int mpirank, bool debug) :
		dataSize_(dataSize),
		mpirank_(mpirank),
		debugMode_(debug)
	{
		encoder_ = SparseModel(encoderPath);
		decoder_ = SparseModel(decoderPath);
		
		// give batchDataMatrix_ the correct number of columns
		batchDataMatrix_.resize(1, dataSize_);
	}
	
	void Autoencoder::compressStates(const std::vector<std::vector<double>> &dataBuffer,
																	 int startingTimestep, int currBatchSize) {		
		Timer copyTimer("[COMPRESS] copy to matrix");
		copyTimer.start();
		copyVectorToMatrix(batchDataMatrix_, dataBuffer);
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
      if (batchNorm > 1e-7){
        double relError = errorNorm / batchNorm;
        std::cout << "[COMPRESS] Relative decompression error for batch: " << relError * 100;
        std::cout << "%" << std::endl;
      }
      std::cout << "[COMPRESS] Decompressed matrix norm (still normalized): ";
      std::cout << decodedMat.norm() << std::endl;
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
		copyMatrixToVector(batchDataMatrix_, dataBuffer);
		copyTimer.stop();

		if (debugMode_ && (mpirank_ == 0)) {
			decompTimer.print();
			normTimer.print();
			copyTimer.print();
		}

		return {batchStorage.getStartingTimestep(), batchStorage.getEndingTimestep()};
	}

	void Autoencoder::copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<std::vector<double>>& dataBuffer) {		
		int totalStatesToStore = dataBuffer.size() * nStates_;
		batchDataMatrix_.resize(totalStatesToStore, dataSize_); // eigen resize is a no-op if not needed

		assert((mat.cols() * nStates_ == dataBuffer.at(0).size(), "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() / nStates_ == dataBuffer.size(), "allocated matrix does not have enough rows for all timesteps and states"));
		
		int baseRow = 0;
		for (const auto& fullState : dataBuffer) {
			for (int i = 0; i < fullState.size(); ++i) {
				int currRow = baseRow + i / dataSize_;
				int currCol = i % dataSize_;
				batchDataMatrix_(currRow, currCol) = fullState.at(i);
			}
			baseRow += nStates_;
		}
	}
	
	void Autoencoder::copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<std::vector<double>>& dataBuffer) {
		// assumes that dataBuffer already has enough storage space. This is because
		// dataBuffer is managed by other code.
		assert((mat.cols() * nStates_ == dataBuffer.at(0).size(), "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() / nStates_ <= dataBuffer.size(), "dataBuffer does not have enough timesteps allocated"));
		
		int vecIndex;
		for (int i = 0; i < mat.rows(); ++i) {
			vecIndex = i / nStates_;
			for (int j = 0; j < dataSize_; ++j) {
				dataBuffer.at(vecIndex).at(i % nStates_ * dataSize_ + j) = mat(i, j);
			}
		}
	}

	CompressedBatch<Eigen::MatrixXf>& Autoencoder::getBatchStorage(const int startingTimestep, const int endingTimestep) {
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
