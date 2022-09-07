#pragma once

#include <Eigen/Core>

namespace sparse_nn {
	class CompressedBatch {
	public:
		CompressedBatch(const int startingTimestep, const int endingTimestep);

		bool isTimestepInBatch(const int timestep) const;

		int getStartingTimestep() const;
		int getEndingTimestep() const;
		
		Eigen::VectorXd mins;
		Eigen::VectorXd ranges;
		Eigen::MatrixXf data;
		
	private:
		int startingTimestep_;
		int endingTimestep_;
		int batchSize_;
	};
}
