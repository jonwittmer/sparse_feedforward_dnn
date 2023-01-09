#pragma once

#include "batch_preparation/batch_preparer.h"
#include "autoencoder/compression_base.h"
#include "batch_preparation/compressed_batch.h"
#include "sparse/sparse_model.h"

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	class Autoencoder : public CompressionBase {
	public:
		Autoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,  
                int nStates, int mpirank, bool debug);
		
		void compressStates(const double* dataBuffer, int startingTimestep, int currBatchSize, int nLocalElements);
		std::pair<int, int> prefetchDecompressedStates(double *dataBuffer,
                                                   const int latestTimestep, int nLocalElements);
		
	protected:
    void verbosePrinting(const CompressedBatch<Eigen::MatrixXf> &batchStorage);

		SparseModel encoder_;
		SparseModel decoder_;
    int batchSize_ = 0;
  };
  
  class SpaceAutoencoder : public Autoencoder {
  public:
  SpaceAutoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,  
                   int nStates, int mpirank, bool debug) : Autoencoder(encoderPath, decoderPath, dataSize,  
                                                                       nStates, mpirank, debug) {
      batchPreparer_ = std::make_unique<SpaceBatchPreparer>();
    }
  };
  
  class TimeAutoencoder : public Autoencoder {
  public:
    // nDofsPerElement is dataSize in Autoencoder
  TimeAutoencoder(const std::string encoderPath, const std::string decoderPath, int nDofsPerElement,  
                  int nStates, int nTimestepsPerBatch, int mpirank, bool debug) : Autoencoder(encoderPath, decoderPath, nDofsPerElement,  
                                                                                              nStates, mpirank, debug) {
      // treat each Rk stage as an independent state
      batchPreparer_ = std::make_unique<TimeBatchPreparer>(nDofsPerElement, nTimestepsPerBatch, 36, 1);
    }
  };

  class TimeRkAutoencoder : public Autoencoder {
  public:
    // nDofsPerElement is dataSize in Autoencoder
  TimeRkAutoencoder(const std::string encoderPath, const std::string decoderPath, int nDofsPerElement,  
                  int nStates, int nTimestepsPerBatch, int mpirank, bool debug) : Autoencoder(encoderPath, decoderPath, nDofsPerElement,  
                                                                                              nStates, mpirank, debug) {
      // 9 actual states with 4 Rk stages each
      batchPreparer_ = std::make_unique<TimeBatchPreparer>(nDofsPerElement, nTimestepsPerBatch, 9, 4);
    }
  };
}
