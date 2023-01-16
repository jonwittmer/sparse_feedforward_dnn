#pragma once
#include "batch_preparation/batch_preparer.h"
#include "autoencoder/autoencoder.h"
#include "batch_preparation/compressed_batch.h"
#include "sparse/sparse_model.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
	class AutoencoderDebug : public Autoencoder {
	public:
		AutoencoderDebug(const std::string encoderPath, const std::string decoderPath, int dataSize,
                     int nStates, int mpirank, bool shouldWrite, double writeProbability, bool debug);
		
		virtual void compressStates(const std::vector<Timestep> &dataBuffer, int startingTimestep, int currBatchSize) override {};
		virtual std::pair<int, int> prefetchDecompressedStates(std::vector<Timestep> &dataBuffer,
                                                           const int latestTimestep) override {};
    virtual void compressStates(const double *dataBuffer, int startingTimestep, int currBatchSize, int nLocalElements) override;
		virtual std::pair<int, int> prefetchDecompressedStates(double *dataBuffer,
                                                           const int latestTimestep, int nLocalElements) override;
		
	protected:
		void writeDataToFile() const;
		bool shouldWrite_;
    double writeProbability_;
    int fullDimension_; // input dimension to autoencoder

		std::vector<CompressedBatch<Eigen::MatrixXf>> compressedStates_;
		CompressedBatch<Eigen::MatrixXf>& getBatchStorage(const int startingTimestep, const int endingTimestep);
	};


  class SpaceAutoencoderDebug : public AutoencoderDebug {
  public:
  SpaceAutoencoderDebug(const std::string encoderPath, const std::string decoderPath, int dataSize,  
                   int nStates, int mpirank, bool shouldWrite, 
                   double writeProbability, bool debug) : AutoencoderDebug(encoderPath, decoderPath, dataSize,  
                                                                           nStates, mpirank, shouldWrite, writeProbability, debug) {
      fullDimension_ = dataSize;
      batchPreparer_ = std::make_unique<SpaceBatchPreparer>();
      //std::cout << "using SpaceAutoencoderDebug" << std::endl;
    }
  };
  

  class TimeAutoencoderDebug : public AutoencoderDebug {
  public:
  // nDofsPerElement is dataSize in Autoencoder
  TimeAutoencoderDebug(const std::string encoderPath, const std::string decoderPath, int nDofsPerElement,  
                       int nStates, int nTimestepsPerBatch, 
                       int mpirank, bool shouldWrite, 
                       double writeProbability, bool debug) : AutoencoderDebug(encoderPath, decoderPath, nDofsPerElement,  
                                                                               nStates, mpirank, shouldWrite, 
                                                                               writeProbability, debug) {
      
      fullDimension_ = nDofsPerElement * nTimestepsPerBatch;
      batchPreparer_ = std::make_unique<TimeBatchPreparer>(nDofsPerElement, nTimestepsPerBatch, 36, 1);
      //std::cout << "using TimeAutoencoderDebug" << std::endl;
    }
  };


  class TimeRkAutoencoderDebug : public AutoencoderDebug {
  public:
  // nDofsPerElement is dataSize in Autoencoder
  TimeRkAutoencoderDebug(const std::string encoderPath, const std::string decoderPath, int nDofsPerElement,  
                         int nStates, int nTimestepsPerBatch, 
                         int mpirank, bool shouldWrite, 
                         double writeProbability, bool debug) : AutoencoderDebug(encoderPath, decoderPath, nDofsPerElement,  
                                                                                 nStates, mpirank, shouldWrite, 
                                                                                 writeProbability, debug) {
      // treat rk stages as part of same state
      fullDimension_ = nDofsPerElement * nTimestepsPerBatch * 4;
      batchPreparer_ = std::make_unique<TimeBatchPreparer>(nDofsPerElement, nTimestepsPerBatch, 9, 4);
    }
  };
}
