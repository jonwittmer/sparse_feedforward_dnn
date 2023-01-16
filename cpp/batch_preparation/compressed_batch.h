#pragma once

#include <Eigen/Core>

namespace sparse_nn {
	template <typename T>
	class CompressedBatch {
	public:
		CompressedBatch(const int startingTimestep, const int endingTimestep);

		bool isTimestepInBatch(const int timestep) const;

		int getStartingTimestep() const;
		int getEndingTimestep() const;
		
		Eigen::VectorXf mins;
		Eigen::VectorXf ranges;
		T data;
		
	private:
		int startingTimestep_;
		int endingTimestep_;
		int batchSize_;
	};
}
