#pragma once

#include <stdlib.h>  // rand()
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif
	
typedef struct {
	int nStates;
	int dataSize;
  int nDofsPerElement;
	int batchSize;
	char *encoderDir;
	char *decoderDir;
	int mpirank;
  char *compressionStrategy;
  double writeProbability;
  int debugMode;
} ae_parameters_t;
	
void TestAutoencoder();
void spawn_autoencoder_thread(ae_parameters_t *ae_params);
void compress_from_array(double **localStates, int timestep, int isLast);
void decompress_to_array(double **localStates, int requestedTimestep, int isForwardMode);
	
#ifdef __cplusplus
}
#endif
