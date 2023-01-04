#include "autoencoder/compressed_batch.h"

#include <Eigen/Core>

namespace sparse_nn {
	template <typename T>
	CompressedBatch<T>::CompressedBatch(const int startingTimestep, const int endingTimestep) {
		startingTimestep_ = startingTimestep;
		endingTimestep_ = endingTimestep;
	}
	
	template <typename T>
	bool CompressedBatch<T>::isTimestepInBatch(const int timestep) const {
		if (timestep < startingTimestep_) { return false; }
		if (timestep > endingTimestep_) { return false; }
		return true;		
	}

	template <typename T>
	int CompressedBatch<T>::getStartingTimestep() const {
		return startingTimestep_;
	}

	template <typename T>
	int CompressedBatch<T>::getEndingTimestep() const {
		return endingTimestep_;
	}

	// instantiate the templates we need
	template class CompressedBatch<Eigen::MatrixXf>;
	template class CompressedBatch<Eigen::MatrixXd>;
}
