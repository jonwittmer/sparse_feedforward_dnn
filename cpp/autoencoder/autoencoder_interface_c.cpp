#include "autoencoder/autoencoder_interface_c.h"
#include "autoencoder/compression_base.h"
#include "autoencoder/autoencoder_debug.h"
#include "utils/timer.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define DEBUG_RANK -1
#define VERBOSE_DEBUG -1

/* shared data */
int globalDataSize;
int nStates = 36;
int mpirank;
int batchSize;
int batchIndex;
int currTimestep;
int compressionTimestep;
int decompressionTimestep;
int maxTimestep = 0;
double *currSharedDataBuffer;
int currBufferMinTimestep = -1;
int currBufferMaxTimestep = -1;
bool exitFlag;
//std::unique_ptr<sparse_nn::CompressionBase> autoencoder_global;
sparse_nn::CompressionBase *autoencoder_global;

void *run_autoencoder_manager(void *args);

void TestAutoencoder() {
	// needed for linking to mangll
	std::cout << "Autoencoder library loaded successfully!\n" << std::endl;
}

void create_autoencoder(ae_parameters_t *aeParams) {
  bool shouldWriteDataToFile = (aeParams->writeProbability > 0);

  std::string compressionStrategy = aeParams->compressionStrategy;
  // if (shouldWriteDataToFile) {
  //   if (compressionStrategy == "space" || compressionStrategy == "default") {
  //     autoencoder = std::make_unique<sparse_nn::SpaceAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
  //                                                                      aeParams->dataSize, aeParams->nStates,
  //                                                                      aeParams->mpirank, shouldWriteDataToFile, 
  //                                                                      aeParams->writeProbability, aeParams->debugMode);
  //   } else if (compressionStrategy == std::string("time")) {
  //     if (aeParams->mpirank == 0) {
  //       std::cout << "creating TimeAutoencoderDebug" << std::endl;
  //     }
  //     autoencoder = std::make_unique<sparse_nn::TimeAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
  //                                                                     aeParams->nDofsPerElement, aeParams->nStates, 
  //                                                                     aeParams->batchSize,
  //                                                                     aeParams->mpirank, shouldWriteDataToFile, 
  //                                                                     aeParams->writeProbability, aeParams->debugMode);
  //   } else if (compressionStrategy == std::string("time_rk")) {
  //     if (aeParams->mpirank == 0) {
  //       std::cout << "creating TimeRkAutoencoderDebug" << std::endl;
  //     }
  //     autoencoder = std::make_unique<sparse_nn::TimeRkAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
  //                                                                       aeParams->nDofsPerElement, aeParams->nStates, 
  //                                                                       aeParams->batchSize,
  //                                                                       aeParams->mpirank, shouldWriteDataToFile, 
  //                                                                       aeParams->writeProbability, aeParams->debugMode);
  //   } else {
  //     std::cout << "Attempting to use unsupported strategy " << compressionStrategy << " in debug mode.";
  //     std::cout << "Choose between 'space', 'time', 'time_rk', or 'default' which is 'space'." << std::endl;
  //   }
  // } else {
    //     if (compressionStrategy == "space" || compressionStrategy == "default") {
    //   autoencoder = std::make_unique<sparse_nn::SpaceAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
    //                                                               aeParams->dataSize, aeParams->nStates,
    //                                                               aeParams->mpirank, aeParams->debugMode);
    // } else if (compressionStrategy == std::string("time")) {
    //   autoencoder = std::make_unique<sparse_nn::TimeAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
    //                                                              aeParams->nDofsPerElement, aeParams->nStates, 
    //                                                              aeParams->batchSize,
    //                                                              aeParams->mpirank, aeParams->debugMode);
    // } else
          
  if (compressionStrategy == std::string("time_rk")) {
    autoencoder_global = new sparse_nn::TimeRkAutoencoder(aeParams->encoderDir, aeParams->decoderDir, 
                                                          aeParams->nDofsPerElement, aeParams->nStates, 
                                                          aeParams->batchSize,
                                                          aeParams->mpirank, aeParams->debugMode);
  } else {
    std::cout << "Attempting to use unsupported strategy " << compressionStrategy << " in pruduction mode.";
    std::cout << "Choose between 'space', 'time', 'time_rk', or 'default' which is 'space'." << std::endl;
  }
  //}
}

// std::unique_ptr<sparse_nn::CompressionBase> create_autoencoder(ae_parameters_t *aeParams) {
//   std::unique_ptr<sparse_nn::CompressionBase> autoencoder;
//   bool shouldWriteDataToFile = (aeParams->writeProbability > 0);

//   std::string compressionStrategy = aeParams->compressionStrategy;
//   if (shouldWriteDataToFile) {
//     if (compressionStrategy == "space" || compressionStrategy == "default") {
//       autoencoder = std::make_unique<sparse_nn::SpaceAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
//                                                                        aeParams->dataSize, aeParams->nStates,
//                                                                        aeParams->mpirank, shouldWriteDataToFile, 
//                                                                        aeParams->writeProbability, aeParams->debugMode);
//     } else if (compressionStrategy == std::string("time")) {
//       if (aeParams->mpirank == 0) {
//         std::cout << "creating TimeAutoencoderDebug" << std::endl;
//       }
//       autoencoder = std::make_unique<sparse_nn::TimeAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
//                                                                       aeParams->nDofsPerElement, aeParams->nStates, 
//                                                                       aeParams->batchSize,
//                                                                       aeParams->mpirank, shouldWriteDataToFile, 
//                                                                       aeParams->writeProbability, aeParams->debugMode);
//     } else if (compressionStrategy == std::string("time_rk")) {
//       if (aeParams->mpirank == 0) {
//         std::cout << "creating TimeRkAutoencoderDebug" << std::endl;
//       }
//       autoencoder = std::make_unique<sparse_nn::TimeRkAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
//                                                                         aeParams->nDofsPerElement, aeParams->nStates, 
//                                                                         aeParams->batchSize,
//                                                                         aeParams->mpirank, shouldWriteDataToFile, 
//                                                                         aeParams->writeProbability, aeParams->debugMode);
//     } else {
//       std::cout << "Attempting to use unsupported strategy " << compressionStrategy << " in debug mode.";
//       std::cout << "Choose between 'space', 'time', 'time_rk', or 'default' which is 'space'." << std::endl;
//     }
//   } else {
//     if (compressionStrategy == "space" || compressionStrategy == "default") {
//       autoencoder = std::make_unique<sparse_nn::SpaceAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
//                                                                   aeParams->dataSize, aeParams->nStates,
//                                                                   aeParams->mpirank, aeParams->debugMode);
//     } else if (compressionStrategy == std::string("time")) {
//       autoencoder = std::make_unique<sparse_nn::TimeAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
//                                                                  aeParams->nDofsPerElement, aeParams->nStates, 
//                                                                  aeParams->batchSize,
//                                                                  aeParams->mpirank, aeParams->debugMode);
//     } else if (compressionStrategy == std::string("time_rk")) {
//       autoencoder = std::make_unique<sparse_nn::TimeRkAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
//                                                                    aeParams->nDofsPerElement, aeParams->nStates, 
//                                                                    aeParams->batchSize,
//                                                                    aeParams->mpirank, aeParams->debugMode);
//     } else {
//       std::cout << "Attempting to use unsupported strategy " << compressionStrategy << " in pruduction mode.";
//       std::cout << "Choose between 'space', 'time', 'time_rk', or 'default' which is 'space'." << std::endl;
//     }
//   }

// 	return std::move(autoencoder);
// }

void print_data(double **dataLocations) {
	for (int i=0; i<20; i++) {
		std::cout << dataLocations[0][i] << std::endl;
	}
}

void copy_to_shared_buffer(double **dataLocations, int dataSize) {
	sparse_nn::Timer copyTimer("[INTERFACE] copy from mangll");
	copyTimer.start();

  // Something has gone wrong if this happens
  if (batchSize <= batchIndex) {
    std::cout << "ERROR: not enough space allocated for batches in autoencoder_interface" << std::endl;
    assert(false);
  }
  
  int timestepStart = batchIndex * nStates * dataSize;
  for (int i = 0; i < nStates; i++) {
    std::copy(dataLocations[i], dataLocations[i] + dataSize, &(currSharedDataBuffer[timestepStart + dataSize * i]));
  }
  copyTimer.stop();
	
	if (mpirank == VERBOSE_DEBUG) {
		copyTimer.print();
	}
}

void copy_from_shared_buffer(double **dataLocations) {
	sparse_nn::Timer copyTimer("[INTERFACE] copy to mangll");
	copyTimer.start();
	if (batchSize > batchIndex) {
    int timestepStart = batchIndex * nStates * globalDataSize;
    
		for (int i = 0; i < nStates; i++) {
			std::copy(&(currSharedDataBuffer[timestepStart + i * globalDataSize]),
                &(currSharedDataBuffer[timestepStart + (i+1) * globalDataSize]),
                dataLocations[i]);
		}
	} else {
		std::cout << "batch index too large - batch_index is outside shared_buffer size" << std::endl;
	}
	copyTimer.stop();
	if (mpirank == VERBOSE_DEBUG) {
		copyTimer.print();
	}
}

void initialize_storage() {
  // create empty vector of the right size
  //auto tempVector = Timestep(timestepSize, 0);
  currSharedDataBuffer = (double *)malloc(batchSize * nStates * globalDataSize * sizeof(double));
}

bool testAutoencoderDebug(const ae_parameters_t *aeParams) {
  // figure out how many timesteps are required to fill up pingpong buffer
  int totalDataSize = aeParams->nStates * aeParams->dataSize * aeParams->batchSize;
  
  // malloc C array with correct dimensions for this mpirank for batchSize timesteps
  double *inputArrayContinuous = (double *)malloc(totalDataSize * sizeof(double));
  double **inputArray = (double **)malloc(aeParams->nStates * aeParams->batchSize * sizeof(double*));
  double *outputArrayContinuous = (double *)malloc(totalDataSize * sizeof(double));
  double **outputArray = (double **)malloc(aeParams->nStates * aeParams->batchSize * sizeof(double*));
  for (int i = 0; i < aeParams->nStates * aeParams->batchSize; ++i) {
    inputArray[i] = &inputArrayContinuous[i * aeParams->dataSize];
    outputArray[i] = &outputArrayContinuous[i * aeParams->dataSize];
  }

  // fill C array with random numbers
  for (int i = 0; i < totalDataSize; ++i) {
    // pseudorandom doubles
    inputArrayContinuous[i] = static_cast<double>(rand()) / static_cast<double>(rand() + 1);
  }

  // invoke compress_from_array batchSize times
  for (int t = 0; t < aeParams->batchSize; ++t) {
    compress_from_array(&(inputArray[t * aeParams->nStates]), t, 0);
  }
  // hack to make sure timestep is read before it is changed
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // reset min/max timesteps to force loading data from autoencoder
  currBufferMinTimestep = -1;
  currBufferMaxTimestep = -1;
  
  // invoke decompress_from_array batchSize times
  for (int t = aeParams->batchSize - 1; t > -1; --t) {
    decompress_to_array(&outputArray[t * aeParams->nStates], t, 0);
  }

  // check that output data matches input data
  bool failed = false;
  for (int i = 0; i < totalDataSize; ++i) {
    if (outputArrayContinuous[i] != inputArrayContinuous[i]) {
      failed = true;
      std::cout << "rank " << mpirank << ": at index " << i << " ";
      std::cout << inputArrayContinuous[i] << " != " << outputArrayContinuous[i] << std::endl;
    }
  }

  // cleanup
  free(inputArray);
  free(inputArrayContinuous);
  free(outputArray);
  free(outputArrayContinuous);
  return failed;
}

void spawn_autoencoder_thread(ae_parameters_t *aeParams) {
	// initialize pthread synchronization variables
  globalDataSize = aeParams->dataSize;
	batchSize = aeParams->batchSize;
	nStates = aeParams->nStates;
	mpirank = aeParams->mpirank;
	
  initialize_storage();

  //autoencoder_global = create_autoencoder(aeParams);
  create_autoencoder(aeParams);
  std::cout << "Made it here " << std::endl;

  // test that copy-retrieve is working correctly in AutoencoderDebug class
  if (aeParams->writeProbability > 0) {
    bool failed = testAutoencoderDebug(aeParams);
    if (!failed) {
      std::cout << "testAutoencoderDebug passed" << std::endl;
    } else {
      std::cout << "testAutoencoderDebug failed to store and retrieve identically\n" << std::endl;
    }
    assert(!failed);
  }
}

void compress_from_array(double **localStateLocations, int timestep, int isLast) {
	// reset batch index
	if (timestep == 0) { batchIndex = 0; }
	
	copy_to_shared_buffer(localStateLocations, globalDataSize);
	
	batchIndex = (batchIndex + 1) % batchSize;
	if (batchIndex == 0 || isLast) {
    int startingTimestep = (batchIndex == 0) ? timestep - batchSize + 1 : timestep - batchIndex + 1;
    currBufferMinTimestep = startingTimestep;
    currBufferMaxTimestep = timestep;
    int currBatchSize = timestep - startingTimestep + 1;
    autoencoder_global->compressStates(currSharedDataBuffer, startingTimestep, currBatchSize, globalDataSize / 64);
	} 
}

void decompress_to_array(double **localStateLocations, int requestedTimestep, int isForwardMode) {
  if (requestedTimestep < currBufferMinTimestep || requestedTimestep > currBufferMaxTimestep) {
    auto minMaxTimesteps = autoencoder_global->prefetchDecompressedStates(currSharedDataBuffer, requestedTimestep, globalDataSize / 64);
    currBufferMinTimestep = minMaxTimesteps.first;
    currBufferMaxTimestep = minMaxTimesteps.second;
  }
  batchIndex = requestedTimestep - currBufferMinTimestep;
	copy_from_shared_buffer(localStateLocations);
}
