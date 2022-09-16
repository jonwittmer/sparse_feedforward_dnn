#include "autoencoder/autoencoder.h"
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
	Autoencoder::Autoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,
							 int nStates, int mpirank, bool shouldWrite, bool debug) :
		dataSize_(dataSize),
		nStates_(nStates),
		mpirank_(mpirank),
		shouldWrite_(shouldWrite),
		debugMode_(debug)
	{
		encoder_ = SparseModel(encoderPath);
		decoder_ = SparseModel(decoderPath);

		// give batchDataMatrix_ the correct number of columns
		batchDataMatrix_.resize(1, dataSize_);
	}
		
	void Autoencoder::compressStates(const std::vector<std::vector<double>> &dataBuffer,
									 int startingTimestep, int currBatchSize) {
		if (shouldWrite_) {
			writeDataToFile(dataBuffer);
			return;
		}
		
		Timer copyTimer("[COMPRESS] copy to matrix");
		copyTimer.start();
		copyVectorToMatrix(batchDataMatrix_, dataBuffer);
		copyTimer.stop();
		
		CompressedBatch& batchStorage = getBatchStorage(startingTimestep, startingTimestep + currBatchSize - 1);

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
	}

	std::pair<int, int> Autoencoder::prefetchDecompressedStates(std::vector<std::vector<double>>& dataBuffer,
																const int latestTimestep) {
		if (shouldWrite_ && (mpirank_ == 0)) {
			std::cout << "Decompress called with shouldWrite_. No data was stored. ";
			std::cout << "Make sure checkpointing is being used." << std::endl;
			return {0, 0};
		}

		// in decompression, we shouldn't be creating any new batches, so only one timestep is needed to
		// decompress batch
		CompressedBatch& batchStorage = getBatchStorage(latestTimestep, latestTimestep);

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

	CompressedBatch& Autoencoder::getBatchStorage(const int startingTimestep, const int endingTimestep) {
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
	
	void Autoencoder::writeDataToFile(const std::vector<std::vector<double>>& data) const {
		// writing all ofthe data will take up an extremely large amount of storage
		// only write part of it
		double writeProbability = 0.001;

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
