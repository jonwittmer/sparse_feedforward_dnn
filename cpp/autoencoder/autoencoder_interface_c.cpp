#include "autoencoder/autoencoder_interface_c.h"
#include "autoencoder/autoencoder.h"
#include "utils/timer.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#define DEBUG_RANK -1

bool debugMode = true;
bool shouldWriteDataToFile = true;
bool useTimeCompression = true;

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
int dataSize;
int dofsPerElement = 64;
int nElements;
int nStates = 36;
int mpirank;
int batchSize;
int batchIndex;
int currTimestep;
int maxTimestep = 0;
std::vector<std::vector<double>> *currSharedDataBuffer;
std::vector<std::vector<double>> *altDataBuffer;
std::vector<std::vector<double>> *pingpongBufferPointers[2];
int currBufferIndex;
int altBufferIndex;
int currBufferMinTimestep = -1;
int currBufferMaxTimestep = -1;
int altBufferMinTimestep = -1;
int altBufferMaxTimestep = -1;
bool exitFlag;
bool decompressFlag;
bool forwardModeFlag; 

void TestAutoencoder() {
	// needed for linking to mangll
	std::cout << "Autoencoder library loaded successfully!\n" << std::endl;
}

sparse_nn::Autoencoder *create_autoencoder(ae_parameters_t *aeParams) {
	sparse_nn::Autoencoder* autoencoder;
	if (useTimeCompression) {
		// abuse some of the arguments to get dimensions to work out correctly when compressing
		// over time instead of over space
		autoencoder = new sparse_nn::Autoencoder(aeParams->encoderDir, aeParams->decoderDir,
												 aeParams->dofsPerElement * batchSize,
												 aeParams->nStates * aeParams->nElements,
												 aeParams->mpirank,
												 shouldWriteDataToFile, debugMode);
	}
	else {
		autoencoder = new sparse_nn::Autoencoder(aeParams->encoderDir, aeParams->decoderDir,
												 aeParams->dataSize, aeParams->nStates,
												 aeParams->mpirank,
												 shouldWriteDataToFile, debugMode);
	}
	
	if (debugMode) {
		std::cout << "Autoencoder in debug mode\n" << std::endl;
	}
	
	return autoencoder;
}

void destroy_autoencoder(void *autoencoder) {
	delete static_cast<sparse_nn::Autoencoder*>(autoencoder);
	return;
}

void print_data(double **dataLocations) {
	for (int i=0; i<20; i++) {
		std::cout << dataLocations[0][i] << std::endl;
	}
}

void copy_to_shared_buffer(double **dataLocations) {
	sparse_nn::Timer copyTimer("[INTERFACE] copy from mangll");
	copyTimer.start();
	if (altDataBuffer->size() < batchIndex) {
		std::cout << "batch index too large - alt_data_buffer is not big enough" << std::endl;
		assert(false);
	}

	// expand altDataBuffer if needed
	if (altDataBuffer->size() == batchIndex) {
		std::vector<double> tempVector(dataSize * nStates);
		(*altDataBuffer).push_back(std::move(tempVector));
	}
	
	for (int i = 0; i < nStates; i++) {
		std::copy(dataLocations[i], dataLocations[i] + dataSize, &((*altDataBuffer)[batchIndex][0]) + i * dataSize);
	}
	copyTimer.stop();
	
	if (mpirank == DEBUG_RANK) {
		copyTimer.print();
	}
}

void copy_from_shared_buffer(double **dataLocations) {
	sparse_nn::Timer copyTimer("[INTERFACE] copy to mangll");
	copyTimer.start();
	if (currSharedDataBuffer->size() > batchIndex) {
		for (int i = 0; i < nStates; i++) {
			std::copy((*currSharedDataBuffer)[batchIndex].begin() + i * dataSize,
					  (*currSharedDataBuffer)[batchIndex].begin() + (i + 1) * dataSize,
					  dataLocations[i]);
		}
	} else {
		std::cout << "batch index too large - batch_index is outside shared_buffer size" << std::endl;
	}
	copyTimer.stop();
	if (mpirank == DEBUG_RANK) {
		copyTimer.print();
	}
}

void copy_to_shared_buffer_time(double **dataLocations) {
	sparse_nn::Timer copyTimer("[INTERFACE] copy from mangll");
	copyTimer.start();
    
	// expand altDataBuffer if needed
	if (altDataBuffer->size() < nStates * nElements) {
		std::cout << "altDataBuffer too small. Adding ";
		std::cout << " vectors. " << std::endl;
		for (int i = 0; i < nStates * nElements - altDataBuffer->size(); ++i) {
			(*altDataBuffer).emplace_back(dofsPerElement * batchSize, 0);
		}
	}

	for (int i = 0; i < nStates; ++i) {
		int currStateIndex = i * nElements;
		for (int j = 0; j < nElements; ++j) {
			double* startLoc = &(dataLocations[i][j * dofsPerElement]);
			std::copy(startLoc, startLoc + dofsPerElement, &((*altDataBuffer)[currStateIndex + j][0]) + dofsPerElement * batchIndex);
		}
	}
	copyTimer.stop();
	
	if (mpirank == DEBUG_RANK) {
		copyTimer.print();
	}
}

void copy_from_shared_buffer_time(double **dataLocations) {
	sparse_nn::Timer copyTimer("[INTERFACE] copy to mangll");
	copyTimer.start();
	if (batchIndex >= currSharedDataBuffer->size()) {
		std::cout << "batch index too large - batch_index is outside shared_buffer size" << std::endl;
		assert(false);
	}

	for (int i = 0; i < nStates; ++i) {
		int currStateIndex = i * nElements;
		for (int j = 0; j < nElements; ++j) {
			double* startLoc = &((*currSharedDataBuffer)[currStateIndex + j][0]) + dofsPerElement * batchIndex;
			std::copy(startLoc, startLoc + dofsPerElement, &(dataLocations[i][j * dofsPerElement]));
		}
	}
	
	copyTimer.stop();
	if (mpirank == DEBUG_RANK) {
		copyTimer.print();
	}
}

void initialize_storage(int bufferSize, int nBuffers) {
	// create empty vector of the right size
	auto tempVector = std::vector<double>(bufferSize, 0);
	for (int i = 0; i < 2; i++) {
		pingpongBufferPointers[i] = new std::vector<std::vector<double>>(nBuffers);

		// copy temp vector into pingpong_buffer elements so each element has the right size
		for (auto &item : (*pingpongBufferPointers[i])) {
			item = tempVector;
		}
	}
	currBufferIndex = 0;
	altBufferIndex = 1;
	currSharedDataBuffer = pingpongBufferPointers[currBufferIndex];
	altDataBuffer = pingpongBufferPointers[altBufferIndex];
}

void spawn_autoencoder_thread(ae_parameters_t *aeParams) {
	// initialize pthread synchronization variables
	bufferReadyToCompress = true;
	bufferIsFull = false;
	batchSize = aeParams->batchSize;
	batchSize = 64;
	dataSize = aeParams->dataSize;
	dofsPerElement = aeParams->dofsPerElement;
	nElements = aeParams->nElements;
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
	std::cout << "autoencoder spawned\n" << std::endl;
	pthread_cond_signal(&compressionCond);
	pthread_mutex_unlock(&compressionMutex);
}

void clear_affinity_mask() {
	int nHardwareThreads = std::thread::hardware_concurrency();
	int nMpiTasksPerNode = 32;
	int localThreadNumber = mpirank % nMpiTasksPerNode; // this ensures NUMA consistency (assuming alternating socket CPU numbering)
	std::cout << nHardwareThreads << " hardware threads available" << std::endl;
	pthread_t thread = pthread_self();
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	// CPU_SET(n_mpi_tasks_per_node + (local_thread_number % (n_hardware_threads - n_mpi_tasks_per_node)), &cpuset);
	// for timing testing, pin to same core as main thread.
	 CPU_SET(mpirank, &cpuset);
	pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}

void* run_autoencoder_manager(void *args) {
	// allow thread to run on any processor
	//clear_affinity_mask();

	// pthreads requires single void * argument 
	ae_parameters_t *aeParams = static_cast<ae_parameters_t *>(args);

	// create the pingpong buffers and initialize them
	if (useTimeCompression) {
		initialize_storage(aeParams->nStates * aeParams->nElements, aeParams->dofsPerElement * batchSize);
	}
	else {
		initialize_storage(aeParams->nStates * aeParams->dataSize, aeParams->batchSize);
	}
	
	// lock the mutex during initialization
	pthread_mutex_lock(&compressionMutex);
	sparse_nn::Autoencoder *autoencoder = create_autoencoder(aeParams);
	
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
			delete(pingpongBufferPointers[0]);
			delete(pingpongBufferPointers[1]);
			
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
			altDataBuffer = pingpongBufferPointers[altBufferIndex];

			// make copy of timestep since main thread modifies this
			int localCurrTimestep = currTimestep;
			
			if (mpirank == 0 && sharedIsLast) {
				std::cout << "Shared is last, timestep " << localCurrTimestep << std::endl;
			}
			
			// reset condition variable to enable mangll to continue solving
			bufferIsFull = false;
			pthread_cond_signal(&sharedDataCond);
			pthread_mutex_unlock(&sharedDataMutex);
			
			// go ahead with compressing
			if (sharedIsLast) { maxTimestep = currTimestep; }
			int currBatchSize = sharedIsLast ? batchIndex : batchSize;
			// this handles the case where the batch is full (batchIndex set to 0)
			currBatchSize = currBatchSize == 0 ? batchSize : currBatchSize;
			int startingTimestep = localCurrTimestep - currBatchSize + 1;
			if (mpirank == DEBUG_RANK) {
				std::cout << "startingTimestep: " << startingTimestep;
				std::cout << " currBatchSize: " <<  currBatchSize << std::endl;
			}
			autoencoder->compressStates(*currSharedDataBuffer, startingTimestep, currBatchSize);
			
			compressTimer.stop();
			if (mpirank == DEBUG_RANK) {
				compressTimer.print();
			}
		} else {
			sparse_nn::Timer decompTimer("[INTERFACE] decompress");
			decompTimer.start();
			
			int localCurrTimestep = currTimestep;
			
			// check if localCurrTimestep has been prefetched
			if (localCurrTimestep < altBufferMinTimestep || localCurrTimestep > altBufferMaxTimestep) {
				auto fetchedTimesteps = autoencoder->prefetchDecompressedStates(*(pingpongBufferPointers[altBufferIndex]),
																				localCurrTimestep);
				altBufferMinTimestep = fetchedTimesteps.first;
				altBufferMaxTimestep = fetchedTimesteps.second;
			}
			
			// swap pointer to current shared buffer
			altBufferIndex = currBufferIndex;
			currBufferIndex = (currBufferIndex + 1) % 2;
			currSharedDataBuffer = pingpongBufferPointers[currBufferIndex];
			altDataBuffer = pingpongBufferPointers[altBufferIndex];
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
				auto fetchedTimesteps = autoencoder->prefetchDecompressedStates(*(pingpongBufferPointers[altBufferIndex]),
																				nextBatchTimestep);
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

	if (useTimeCompression) {
		copy_to_shared_buffer_time(localStateLocations);
	}
	else {
		copy_to_shared_buffer(localStateLocations);
	}
	
	batchIndex = (batchIndex + 1) % batchSize;
	if (batchIndex == 0 || sharedIsLast) {
		bufferIsFull = true;

		sparse_nn::Timer waitingTimer("[INTERFACE] pde waiting for compression");
		waitingTimer.start();

		pthread_mutex_lock(&compressionMutex);
		while (bufferReadyToCompress) {
			pthread_cond_wait(&compressionCond, &compressionMutex);
		}
		waitingTimer.stop();
		if (mpirank == DEBUG_RANK) {
			waitingTimer.print();
		}
		
		bufferReadyToCompress = true;		
		
		// we need to release sharedDataMutex so that compression can switch buffers
		pthread_mutex_unlock(&sharedDataMutex);
		pthread_cond_signal(&compressionCond);
		pthread_mutex_unlock(&compressionMutex);
	} else {
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
	batchIndex = batchSize - (currBufferMaxTimestep - requestedTimestep) - 1;
	if (useTimeCompression) {
		copy_from_shared_buffer_time(localStateLocations);
	}
	else {
		copy_from_shared_buffer(localStateLocations);
	}
	pthread_mutex_unlock(&sharedDataMutex);
}
