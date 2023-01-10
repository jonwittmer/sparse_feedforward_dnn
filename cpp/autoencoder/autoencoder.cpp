#include "autoencoder/autoencoder.h"
#include "batch_preparation/batch_preparer.h"
#include "batch_preparation/compressed_batch.h"
#include "normalization/normalization.h"
#include "sparse/sparse_model.h"
#include "utils/timer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <Eigen/Core>

#define TIME_CODE(a, name)                             \
  do {                                                 \
    Timer my_timer(name);                              \
    my_timer.start();                                  \
    a;                                                 \
    my_timer.stop();                                   \
    if (debugMode_ && mpirank_ == 0) {                 \
      my_timer.print();                                \
    }                                                  \
  } while(0);
 
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

	void Autoencoder::compressStates(const double *dataBuffer,
																	 int startingTimestep, int currBatchSize, int nLocalElements) {
    // first batch sets the batch size
    if (batchSize_ == 0) {
      batchSize_ = currBatchSize;
    }

		TIME_CODE(batchPreparer_->copyVectorToMatrix(batchDataMatrix_, dataBuffer, nLocalElements);, "[COMPRESS] copy to matrix");
    
		CompressedBatch<Eigen::MatrixXf>& batchStorage = getBatchStorage(startingTimestep, startingTimestep + currBatchSize - 1);
		
    // directly store last batch if it is not the correct size since
    // compressing in time requires fixed batch size
    if (currBatchSize != batchSize_) {
      batchStorage.data = batchDataMatrix_.cast<float>();
      return;
    }

		// normalize
		TIME_CODE(
      batchStorage.mins = subtractAndReturnMins(batchDataMatrix_);		
      batchStorage.ranges = divideAndReturnRanges(batchDataMatrix_);,
             "[COMPRESS] normalization"
    );
    if (mpirank_ == 0) {
      std::cout << "[DEBUG] " << batchDataMatrix_.rows() << std::endl;
    }
    // Eigen::MatrixXf temp;
    // TIME_CODE(
    //   temp = batchDataMatrix_.cast<float>();, "[COMPRESS] cast to float"
    // );
		// do compression
		TIME_CODE(
      batchStorage.data = encoder_.run(batchDataMatrix_.cast<float>());, "[COMPRESS] compression");
    //verbosePrinting(batchStorage);
	}
	
	std::pair<int, int> Autoencoder::prefetchDecompressedStates(double *dataBuffer,
																															const int latestTimestep, int nLocalElements) {
		// in decompression, we shouldn't be creating any new batches, so only one timestep is needed to
		// decompress batch
		CompressedBatch<Eigen::MatrixXf>& batchStorage = getBatchStorage(latestTimestep, latestTimestep);
    //std::cout << "requested timestep " << latestTimestep << std::endl;
    if (batchStorage.getEndingTimestep() - batchStorage.getStartingTimestep() + 1 < batchSize_) {
      batchDataMatrix_ = batchStorage.data.cast<double>();
    } else {
      Eigen::MatrixXf decompressedBatch;
      TIME_CODE(decompressedBatch = decoder_.run(batchStorage.data);, "[DECOMPRESS] decompression");
		
      TIME_CODE(
        batchDataMatrix_ = unnormalize(decompressedBatch, batchStorage.mins, batchStorage.ranges);, 
        "[DECOMPRESS] unnormalize"
      );
    }
    
    if (batchDataMatrix_.rows() == 0 || batchDataMatrix_.cols() == 0) {
      if (true) {
        std::cout << "Something went wrong with retrieving timestep " << latestTimestep << std::endl;
      }
    }

		TIME_CODE(batchPreparer_->copyMatrixToVector(batchDataMatrix_, dataBuffer, nLocalElements);, "[DECOMPRESS] copy to vector");
		return {batchStorage.getStartingTimestep(), batchStorage.getEndingTimestep()};
	}

  void Autoencoder::verbosePrinting(const CompressedBatch<Eigen::MatrixXf> &batchStorage) {
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
} // namespace sparse_nn
