#include "autoencoder/batch_preparer.h"

#include <vector>
#include <Eigen/Core>

namespace sparse_nn {
  SpaceBatchPreparer::SpaceBatchPreparer(int dataSize, int nStates) :
    dataSize_(dataSize), nStates_(nStates) {}
  
	void SpaceBatchPreparer::copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) {		
		int totalStatesToStore = dataBuffer.size() * nStates_;
		mat.resize(totalStatesToStore, dataSize_); // eigen resize is a no-op if not needed

		assert((mat.cols() * nStates_ == dataBuffer.at(0).size(), "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() / nStates_ == dataBuffer.size(), "allocated matrix does not have enough rows for all timesteps and states"));
		
		int baseRow = 0;
		for (const auto& fullState : dataBuffer) {
			for (int i = 0; i < fullState.size(); ++i) {
				int currRow = baseRow + i / dataSize_;
				int currCol = i % dataSize_;
				mat(currRow, currCol) = fullState.at(i);
			}
			baseRow += nStates_;
		}
	}
	
	void SpaceBatchPreparer::copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) {
		// assumes that dataBuffer already has enough storage space. This is because
		// dataBuffer is managed by other code.
		assert((mat.cols() * nStates_ == dataBuffer.at(0).size(), "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() / nStates_ <= dataBuffer.size(), "dataBuffer does not have enough timesteps allocated"));
		
		int vecIndex;
		for (int i = 0; i < mat.rows(); ++i) {
			vecIndex = i / nStates_;
			for (int j = 0; j < dataSize_; ++j) {
				dataBuffer.at(vecIndex).at(i % nStates_ * dataSize_ + j) = mat(i, j);
			}
		}
	}



  TimeBatchPreparer::TimeBatchPreparer(int nDofsPerElement, int nStates, int nTimestepsPerBatch) :
    dataSize_(nDofsPerElement), nStates_(nStates), nTimestepsPerBatch_(nTimestepsPerBatch) {}
  
	void TimeBatchPreparer::copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) {		
    assert((nTimestepsPerBatch_ == dataBuffer.size(), "dataBuffer does not match nTimestepsPerBatch"));
    int nElements = dataBuffer.at(0).size()  / nStates_ / dataSize_;

    // Each element is treated as it's own state, so the total number of states == nStates_ * nElements 
    int totalStatesToStore = nElements * nStates_;
		mat.resize(totalStatesToStore, dataSize_ * nTimestepsPerBatch_); // dataSize_ is nDofsPerElement

		assert((mat.cols() == dataSize_ * nTimestepsPerBatch_, "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() * dataSize_  == dataBuffer.at(0).size(), "allocated matrix does not have enough rows for all timesteps and states"));
    
    int t = 0;
    for (const auto& timestep : dataBuffer) {
      int colStart = dataSize_ * t;
      for (int state = 0; state < nStates_; ++state) {
        int stateOffset = state * nElements;
        for (int element = 0; element < nElements; ++element){
          int row = stateOffset + element;
          for (int i = 0; i < dataSize_; ++i) {
            int vectorIndex = row * dataSize_ + i;
            mat(row, colStart + i) = timestep[vectorIndex];
          }
        }
      }
      ++t;
    }
	}
	
	void TimeBatchPreparer::copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) {
    int nElements = dataBuffer.at(0).size()  / nStates_ / dataSize_;
    int t = 0;
    for (auto& timestep : dataBuffer) {
      int colStart = dataSize_ * t;
      for (int state = 0; state < nStates_; ++state) {
        int stateOffset = state * nElements;
        for (int element = 0; element < nElements; ++element){
          int row = stateOffset + element;
          for (int i = 0; i < dataSize_; ++i) {
            int vectorIndex = row * dataSize_ + i;
            timestep[vectorIndex] = mat(row, colStart + i);
          }
        }
      }
      ++t;
    }
	}
} // namespace sparse_nn
