#pragma once

#include <stdlib.h>  // rand()
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif
	
typedef struct {
	int nStates;
	int nElements;
	int dofsPerElement;
	int dataSize;
	int batchSize;
	char *encoderDir;
	char *decoderDir;
	int mpirank;
} ae_parameters_t;
	
void TestAutoencoder();
void spawn_autoencoder_thread(ae_parameters_t *ae_params);
void *run_autoencoder_manager(void *args);
void destroy_autoencoder(void *autoencoder);
void compress_from_array(double **localStates, int timestep, int isLast);
void decompress_to_array(double **localStates, int requestedTimestep, int isForwardMode);
	
#ifdef __cplusplus
}
#endif
