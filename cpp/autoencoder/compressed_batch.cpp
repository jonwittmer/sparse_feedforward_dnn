#include "compressed_batch.h"

#include <Eigen/Core>

namespace sparse_nn {
	CompressedBatch::CompressedBatch(const int startingTimestep, const int endingTimestep) {
		startingTimestep_ = startingTimestep;
		endingTimestep_ = endingTimestep;
	}
	
	bool CompressedBatch::isTimestepInBatch(const int timestep) const {
		if (timestep < startingTimestep_) { return false; }
		if (timestep > endingTimestep_) { return false; }
		return true;		
	}

	int CompressedBatch::getStartingTimestep() const {
		return startingTimestep_;
	}

	int CompressedBatch::getEndingTimestep() const {
		return endingTimestep_;
	}
}
