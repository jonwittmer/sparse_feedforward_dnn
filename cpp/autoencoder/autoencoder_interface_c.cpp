#include "autoencoder/autoencoder_interface_c.h"
#include "autoencoder/compression_base.h"
#include "autoencoder/autoencoder_debug.h"
#include "utils/timer.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define DEBUG_RANK 0
#define VERBOSE_DEBUG -1

/* pthreads variables for spinning off compression. */
pthread_t compressionThread;
pthread_cond_t sharedDataCond;
pthread_cond_t compressionCond;
pthread_mutex_t sharedDataMutex;
pthread_mutex_t compressionMutex;
bool bufferIsFull;
bool bufferReadyToCompress;

/* shared data */
bool sharedIsLast;
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
double *altSharedDataBuffer;
double *pingpongBufferPointers[2];
int currBufferIndex;
int altBufferIndex;
int currBufferMinTimestep = -1;
int currBufferMaxTimestep = -1;
int altBufferMinTimestep = -1;
int altBufferMaxTimestep = -1;
bool exitFlag;
bool decompressFlag;
bool forwardModeFlag; 

void *run_autoencoder_manager(void *args);

void TestAutoencoder() {
	// needed for linking to mangll
	std::cout << "Autoencoder library loaded successfully!\n" << std::endl;
}

std::unique_ptr<sparse_nn::CompressionBase> create_autoencoder(ae_parameters_t *aeParams) {
  std::unique_ptr<sparse_nn::CompressionBase> autoencoder;
  bool shouldWriteDataToFile = (aeParams->writeProbability > 0);

  std::string compressionStrategy = aeParams->compressionStrategy;
  if (shouldWriteDataToFile) {
    if (compressionStrategy == "space" || compressionStrategy == "default") {
      autoencoder = std::make_unique<sparse_nn::SpaceAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
                                                                       aeParams->dataSize, aeParams->nStates,
                                                                       aeParams->mpirank, shouldWriteDataToFile, 
                                                                       aeParams->writeProbability, aeParams->debugMode);
    } else if (compressionStrategy == std::string("time")) {
      if (aeParams->mpirank == 0) {
        std::cout << "creating TimeAutoencoderDebug" << std::endl;
      }
      autoencoder = std::make_unique<sparse_nn::TimeAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
                                                                      aeParams->nDofsPerElement, aeParams->nStates, 
                                                                      aeParams->batchSize,
                                                                      aeParams->mpirank, shouldWriteDataToFile, 
                                                                      aeParams->writeProbability, aeParams->debugMode);
    } else if (compressionStrategy == std::string("time_rk")) {
      if (aeParams->mpirank == 0) {
        std::cout << "creating TimeRkAutoencoderDebug" << std::endl;
      }
      autoencoder = std::make_unique<sparse_nn::TimeRkAutoencoderDebug>(aeParams->encoderDir, aeParams->decoderDir, 
                                                                        aeParams->nDofsPerElement, aeParams->nStates, 
                                                                        aeParams->batchSize,
                                                                        aeParams->mpirank, shouldWriteDataToFile, 
                                                                        aeParams->writeProbability, aeParams->debugMode);
    } else {
      std::cout << "Attempting to use unsupported strategy " << compressionStrategy << " in debug mode.";
      std::cout << "Choose between 'space', 'time', 'time_rk', or 'default' which is 'space'." << std::endl;
    }
  } else {
    if (compressionStrategy == "space" || compressionStrategy == "default") {
      autoencoder = std::make_unique<sparse_nn::SpaceAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
                                                                  aeParams->dataSize, aeParams->nStates,
                                                                  aeParams->mpirank, aeParams->debugMode);
    } else if (compressionStrategy == std::string("time")) {
      autoencoder = std::make_unique<sparse_nn::TimeAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
                                                                 aeParams->nDofsPerElement, aeParams->nStates, 
                                                                 aeParams->batchSize,
                                                                 aeParams->mpirank, aeParams->debugMode);
    } else if (compressionStrategy == std::string("time_rk")) {
      autoencoder = std::make_unique<sparse_nn::TimeRkAutoencoder>(aeParams->encoderDir, aeParams->decoderDir, 
                                                                   aeParams->nDofsPerElement, aeParams->nStates, 
                                                                   aeParams->batchSize,
                                                                   aeParams->mpirank, aeParams->debugMode);
    } else {
      std::cout << "Attempting to use unsupported strategy " << compressionStrategy << " in pruduction mode.";
      std::cout << "Choose between 'space', 'time', 'time_rk', or 'default' which is 'space'." << std::endl;
    }
  }
	return std::move(autoencoder);
}

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
    std::copy(dataLocations[i], dataLocations[i] + dataSize, &(altSharedDataBuffer[timestepStart + dataSize * i]));
  }
  copyTimer.stop();
	
	if (mpirank == VERBOSE_DEBUG - 1) {
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

void initialize_storage(int nStates, int dataSize) {
	// create empty vector of the right size
	//auto tempVector = Timestep(timestepSize, 0);
	for (int i = 0; i < 2; i++) {
		pingpongBufferPointers[i] = (double *)malloc(batchSize * nStates * dataSize * sizeof(double));
	}
	currBufferIndex = 0;
	altBufferIndex = 1;
	currSharedDataBuffer = pingpongBufferPointers[currBufferIndex];
	altSharedDataBuffer = pingpongBufferPointers[altBufferIndex];
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
  altBufferMinTimestep = -1;
  altBufferMaxTimestep = -1;

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
	bufferReadyToCompress = true;
	bufferIsFull = false;
  globalDataSize = aeParams->dataSize;
	batchSize = aeParams->batchSize;
	nStates = aeParams->nStates;
	mpirank = aeParams->mpirank;
	
	// pthread initialize 
	pthread_cond_init(&compressionCond, nullptr);
	pthread_mutex_init(&compressionMutex, nullptr);
	pthread_cond_init(&sharedDataCond, nullptr);
	pthread_mutex_init(&sharedDataMutex, nullptr);

	// create compression thread
	pthread_create(&compressionThread, nullptr, run_autoencoder_manager, static_cast<void *>(aeParams));

	// wait for compression thread to initialize before moving on
	pthread_mutex_lock(&compressionMutex);
	while (bufferReadyToCompress) {
		pthread_cond_wait(&compressionCond, &compressionMutex);
	}
	pthread_cond_signal(&compressionCond);
	pthread_mutex_unlock(&compressionMutex);

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

void clear_affinity_mask() {
	int nHardwareThreads = std::thread::hardware_concurrency();
	int nMpiTasksPerNode = 56;
	int localThreadNumber = mpirank % nMpiTasksPerNode; // this ensures NUMA consistency (assuming alternating socket CPU numbering)
	std::cout << nHardwareThreads << " hardware threads available" << std::endl;
	pthread_t thread = pthread_self();
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	//CPU_SET(nMpiTasksPerNode + (localThreadNumber % (nHardwareThreads - nMpiTasksPerNode)), &cpuset);
	// for timing testing, pin to same core as main thread.
	CPU_SET(localThreadNumber, &cpuset);
	pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}

void *run_autoencoder_manager(void *args) {
	// force thread to run on same core as process to keep NUMA access problems to a minimum
	clear_affinity_mask();

	// pthreads requires single void * argument 
	ae_parameters_t *aeParams = static_cast<ae_parameters_t *>(args);

	// create the pingpong buffers and initialize them
	initialize_storage(aeParams->nStates, aeParams->dataSize);
	
	// lock the mutex during initialization
	pthread_mutex_lock(&compressionMutex);
  std::unique_ptr<sparse_nn::CompressionBase> autoencoder = create_autoencoder(aeParams);
	
	// signal parent process and unlock mutex
	bufferReadyToCompress = false;
	pthread_cond_signal(&compressionCond);
	pthread_mutex_unlock(&compressionMutex);

	while (true) {
		// wait for work to do
		pthread_mutex_lock(&compressionMutex);
		while (!bufferReadyToCompress) {
			pthread_cond_wait(&compressionCond, &compressionMutex);
		}
		// acquire shared_data_mutex - no cond_wait needed since we release mutex in calling thread and wait on condition there
		pthread_mutex_lock(&sharedDataMutex);

		// exit if commanded
		if (exitFlag) {
			// free the shared locs memory
			free(pingpongBufferPointers[0]);
			free(pingpongBufferPointers[1]);
			
			// release lock so mangll can continue to solve pde
      bufferReadyToCompress = false;
			pthread_cond_signal(&compressionCond);
			pthread_mutex_unlock(&compressionMutex);
			std::cout << "Exiting from thread " << mpirank << std::endl;
			break;
		}

		if (!decompressFlag) {
      sparse_nn::Timer compressTimer("[INTERFACE] compress");
			compressTimer.start();

			// swap pointer to current shared buffer
			altBufferIndex = currBufferIndex;
			currBufferIndex = (currBufferIndex + 1) % 2;
			currSharedDataBuffer = pingpongBufferPointers[currBufferIndex];
			altSharedDataBuffer = pingpongBufferPointers[altBufferIndex];

			// make copy of timestep since main thread modifies this
			
			if (mpirank == 0 && sharedIsLast) {
				std::cout << "Shared is last, timestep " << compressionTimestep << std::endl;
			}
			
			// reset condition variable to enable mangll to continue solving
			bufferIsFull = false;
			pthread_cond_signal(&sharedDataCond);
			pthread_mutex_unlock(&sharedDataMutex);
			
			// go ahead with compressing
			// if (sharedIsLast) { maxTimestep = currTimestep; }
      if (compressionTimestep > maxTimestep) { maxTimestep = compressionTimestep; }
			int currBatchSize = sharedIsLast ? batchIndex : batchSize;
			// this handles the case where the batch is full (batchIndex set to 0)
			currBatchSize = currBatchSize == 0 ? batchSize : currBatchSize;
			int startingTimestep = compressionTimestep - currBatchSize + 1;
			if (mpirank == DEBUG_RANK) {
				std::cout << "startingTimestep: " << startingTimestep;
				std::cout << " currBatchSize: " <<  currBatchSize;
        std::cout << " nLocalElements: " << globalDataSize / 64 << std::endl;
			}
			autoencoder->compressStates(currSharedDataBuffer, startingTimestep, currBatchSize, globalDataSize / 64);

			compressTimer.stop();
			if (mpirank == DEBUG_RANK) {
				compressTimer.print();
			}
		} else {
			sparse_nn::Timer decompTimer("[INTERFACE] decompress");
			decompTimer.start();
			
			// check if compressionTimestep has been prefetched
			if (decompressionTimestep < altBufferMinTimestep || decompressionTimestep > altBufferMaxTimestep) {
				auto fetchedTimesteps = autoencoder->prefetchDecompressedStates((pingpongBufferPointers[altBufferIndex]),
                                                                        decompressionTimestep, globalDataSize / 64);
				altBufferMinTimestep = fetchedTimesteps.first;
				altBufferMaxTimestep = fetchedTimesteps.second;
			}
			
			// swap pointer to current shared buffer
			altBufferIndex = currBufferIndex;
			currBufferIndex = (currBufferIndex + 1) % 2;
			currSharedDataBuffer = pingpongBufferPointers[currBufferIndex];
			altSharedDataBuffer = pingpongBufferPointers[altBufferIndex];
			currBufferMinTimestep = altBufferMinTimestep;
			currBufferMaxTimestep = altBufferMaxTimestep;

			// reset condition variable to enable mangll to continue solving
			bufferIsFull = false;
			pthread_cond_signal(&sharedDataCond);
			pthread_mutex_unlock(&sharedDataMutex);

			// runs decoder to decompress batch of states if the last time step requested
			// was the minimum of the timesteps that are already prefetched.
			int nextBatchTimestep;
			if (!forwardModeFlag && currBufferMinTimestep > 0) {
				nextBatchTimestep = currBufferMinTimestep - 1;
			} else {
				nextBatchTimestep = currBufferMaxTimestep + 1;
			}
      
			if (nextBatchTimestep >= 0 && nextBatchTimestep <= maxTimestep) {
				auto fetchedTimesteps = autoencoder->prefetchDecompressedStates((pingpongBufferPointers[altBufferIndex]),
                                                                       nextBatchTimestep, globalDataSize / 64);
				altBufferMinTimestep = fetchedTimesteps.first;
				altBufferMaxTimestep = fetchedTimesteps.second;
			}

			decompTimer.stop();
			if (mpirank == DEBUG_RANK) {
				decompTimer.print();
			}
		}

		// reset condition variable
		bufferReadyToCompress = false;
		pthread_cond_signal(&compressionCond);
		pthread_mutex_unlock(&compressionMutex);
	}
}
 
void compress_from_array(double **localStateLocations, int timestep, int isLast) {
	// this function does not have access to autoencoder object
	exitFlag = false;
	decompressFlag = false;
	currTimestep = timestep;
	sharedIsLast = static_cast<bool>(isLast);
	
	pthread_mutex_lock(&sharedDataMutex);
  while (bufferIsFull) {
    pthread_cond_wait(&sharedDataCond, &sharedDataMutex);
  }

	// reset batch index
	if (timestep == 0) { batchIndex = 0; }
	
	copy_to_shared_buffer(localStateLocations, globalDataSize);
	
	batchIndex = (batchIndex + 1) % batchSize;
	if (batchIndex == 0 || sharedIsLast) {
		bufferIsFull = true;


		pthread_mutex_lock(&compressionMutex);
		while (bufferReadyToCompress) {
			pthread_cond_wait(&compressionCond, &compressionMutex);
		}
		
		bufferReadyToCompress = true;		
    
    // only change this when we have the lock so that the compression 
    // thread has the latest timestep in batch
    compressionTimestep = currTimestep;

		// we need to release sharedDataMutex so that compression can switch buffers
		pthread_mutex_unlock(&sharedDataMutex);
		pthread_cond_signal(&compressionCond);
		pthread_mutex_unlock(&compressionMutex);
    
    // sparse_nn::Timer waitingTimer("[INTERFACE] pde waiting for compression");
		// waitingTimer.start();
    // std::this_thread::sleep_for(std::chrono::milliseconds(5));
    // pthread_mutex_lock(&sharedDataMutex);
    // while (bufferIsFull) {
    //   pthread_cond_wait(&sharedDataCond, &sharedDataMutex);
    // }
    //waitingTimer.stop();
		//if (mpirank == DEBUG_RANK) {
		//	waitingTimer.print();
		//}
    // aids in making sure compression thread actually runs 
    // in compression mode before a decompress call is made
    if (sharedIsLast) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
	}
  else {
    pthread_mutex_unlock(&sharedDataMutex);	
	}
}

void decompress_to_array(double **localStateLocations, int requestedTimestep, int isForwardMode) {
	exitFlag = false;
	decompressFlag = true;
	forwardModeFlag = static_cast<bool>(isForwardMode);
	currTimestep = requestedTimestep;

	// make sure that the current buffer is available for read
	// - switching ping-pong buffer is not in progress
	sparse_nn::Timer waitingTimer2("[INTERFACE] pde waiting for currSharedBuffer");
	waitingTimer2.start();
	pthread_mutex_lock(&sharedDataMutex);
	while (bufferIsFull) {
		pthread_cond_wait(&sharedDataCond, &sharedDataMutex);
	}
	waitingTimer2.stop();
	if (mpirank == DEBUG_RANK) {
		waitingTimer2.print();
	}

	// make sure requested timestep is in current buffer
	if (requestedTimestep < currBufferMinTimestep || requestedTimestep > currBufferMaxTimestep) {
		bufferIsFull = true;

		sparse_nn::Timer waitingTimer("[INTERFACE] pde waiting for decompression");
		waitingTimer.start();
		
		pthread_mutex_lock(&compressionMutex);
		while (bufferReadyToCompress) {
			pthread_cond_wait(&compressionCond, &compressionMutex);
		}
		
		bufferReadyToCompress = true;
	
    decompressionTimestep = requestedTimestep;
	
		// we need to release sharedDataMutex so that compression can switch buffers
		// signals compression thread to prefetch the next batch
		pthread_mutex_unlock(&sharedDataMutex);
		pthread_cond_signal(&compressionCond);
		pthread_mutex_unlock(&compressionMutex);

		// wait until compression thread releases shared buffer to move on
		pthread_mutex_lock(&sharedDataMutex);
		while (bufferIsFull) {
			pthread_cond_wait(&sharedDataCond, &sharedDataMutex);
		}
		waitingTimer.stop();
		if (mpirank == DEBUG_RANK) {
			waitingTimer.print();
		}
	}

	// now that we know the current shared buffer has the requested timestep, copy data
	// Data is aligned to the end of buffer since we are going backward, so subtract from batchSize to get index
	batchIndex = requestedTimestep - currBufferMinTimestep;
	copy_from_shared_buffer(localStateLocations);
	pthread_mutex_unlock(&sharedDataMutex);
}
