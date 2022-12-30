#pragma once 

#include "autoencoder/autoencoder.h"
#include "autoencoder/batch_preparer.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace sprase_nn {
  class SeparateVelocityCompressor : public CompressionBase {
  public:
    SeparateVelocityCompressor(const std::string encoderPath, const std::string decoderPath, 
                               int dataSize, int nStates, int mpirank, bool debug);
    
    virtual void compressStates(const std::vector<Timestep>& dataBuffer, int startingTimestep, int currBatchSize) override;
		virtual std::pair<int, int> prefetchDecompressedStates(std::vector<Timestep> &dataBuffer,
                                                           const int latestTimestep) override;
    
  protected:
    void splitBatch(const std::vector<Timestep>& dataBuffer);
    void consolidateBatch(std::vector<Timestep>& dataBuffer);
    
    std::unique_ptr<CompressionBase> velocityCompressor_;
    std::unique_ptr<CompressionBase> strainCompressor_;

    std::vector<int> velocityStateIds_;
    std::vector<Timestep> velocityBatch_;
    std::vector<Timestep> strainBatch_;
  };

  class SeparateVelocitySpaceAutoencoder : public SeparateVelocityCompressor {
  public:
    SeparateVelocitySpaceAutoencoder(const std::string encoderPath, const std::string decoderPath, 
                                     int dataSize, int nStates, int mpirank, bool debug) : 
    CompressionBase(encoderPath, decoderPath, dataSize, nStates, mpirank, debug) {
      veclocityStateIds_ = {0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29};
      velocityCompressor_ = std::make_unique<SpaceAutoencoder>(encoderPath, decoderPath, dataSize,
                                                               velocityStateIds_.size(), mpirank, debug);
      strainCompressor_ = std::make_unique<SpaceAutoencoder>(encoderPath, decoderPath, dataSize,
                                                             nStates - velocityStateIds_.size(), mpirank, debug);
    }
  }
}

