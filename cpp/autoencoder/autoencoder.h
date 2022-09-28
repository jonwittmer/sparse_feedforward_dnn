#pragma once

#include "autoencoder/compression_base.h"
#include "autoencoder/compressed_batch.h"
#include "sparse/sparse_model.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	class Autoencoder : public CompressionBase {
	public:
		Autoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,  
                int nStates, int mpirank, bool debug);
		
		virtual void compressStates(const std::vector<Timestep> &dataBuffer, int startingTimestep, int currBatchSize) override;
		virtual std::pair<int, int> prefetchDecompressedStates(std::vector<Timestep> &dataBuffer,
													   const int latestTimestep) override;
		
	protected:
		SparseModel encoder_;
		SparseModel decoder_;
  };
  
  class SpaceAutoencoder : public Autoencoder {
  public:
  SpaceAutoencoder(const std::string encoderPath, const std::string decoderPath, int dataSize,  
                   int nStates, int mpirank, bool debug) : Autoencoder(encoderPath, decoderPath, dataSize,  
                                                                       nStates, mpirank, debug) {
      batchPreparer_ = std::make_unique<SpaceBatchPreparer>(dataSize, nStates);
    }
  };
  
  class TimeAutoencoder : public Autoencoder {
  public:
    // nDofsPerElement is dataSize in Autoencoder
  TimeAutoencoder(const std::string encoderPath, const std::string decoderPath, int nDofsPerElement,  
                  int nStates, int nTimestepsPerBatch, int mpirank, bool debug) : Autoencoder(encoderPath, decoderPath, nDofsPerElement,  
                                                                                              nStates, mpirank, debug) {
      batchPreparer_ = std::make_unique<TimeBatchPreparer>(nDofsPerElement, nStates, nTimestepsPerBatch);
    }
  };
}
