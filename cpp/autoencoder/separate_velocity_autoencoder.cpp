#include "autoencoder/separate_velocity_autoencoder.h"
#include "autoencoder/autoencoder.h"
#include "autoencoder/batch_preparer.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace sprase_nn {
  void SeparateVelocityCompressor::compressStates(const std::vector<Timestep> &dataBuffer, int startingTimestep, int currBatchSize) {

  }

  std::pair<int, int> SeparateVelocityCompressor::prefetchDecompressedStates(std::vector<Timestep> &dataBuffer,
                                                                             const int latestTimestep) {
    
  }


}
