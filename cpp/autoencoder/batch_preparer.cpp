#include "autoencoder/batch_preparer.h"

#include <vector>
#include <Eigen/Core>
#include <iostream>

namespace sparse_nn {
  SpaceBatchPreparer::SpaceBatchPreparer() {} 
  
	void SpaceBatchPreparer::copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) {		
    int nStates = dataBuffer.at(0).size();
		int totalStatesToStore = dataBuffer.size() * nStates;
    int dataSize = dataBuffer.at(0).at(0).size();
		mat.resize(totalStatesToStore, dataSize); // eigen resize is a no-op if not needed

		assert((mat.rows() / nStates == dataBuffer.size(), "allocated matrix does not have enough rows for all timesteps and states"));
		
		int currRow = 0;
		for (const auto& fullState : dataBuffer) {
      for(const auto& currState : fullState) {
        for (int i = 0; i < currState.size(); ++i) {
          mat(currRow, i) = currState.at(i);
        }
        ++currRow;
			}
		}
	}
	
	void SpaceBatchPreparer::copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) {
    int nStates = dataBuffer.at(0).size();
    assert((nStates > 0, "dataBuffer does not have all states allocated"));
    assert((mat.rows() / nStates <= dataBuffer.size(), "dataBuffer does not have enough timesteps allocated"));
		
		for (int i = 0; i < mat.rows(); ++i) {
			int timestepIndex = i / nStates;
      int stateIndex = i % nStates;
			for (int j = 0; j < mat.cols(); ++j) {
				dataBuffer.at(timestepIndex).at(stateIndex).at(j) = mat(i, j);
			}
		}
	}



  TimeBatchPreparer::TimeBatchPreparer(int nDofsPerElement, int nTimestepsPerBatch) :
    nDofsPerElement_(nDofsPerElement), nTimestepsPerBatch_(nTimestepsPerBatch) {}
  
	void TimeBatchPreparer::copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) {
    assert((nTimestepsPerBatch_ == dataBuffer.size(), "dataBuffer does not match nTimestepsPerBatch"));

    int nLocalElements = dataBuffer.at(0).at(0).size() / nDofsPerElement_;
    int nStates = dataBuffer.at(0).size();
    assert((nLocalElements * nDofsPerElement_ == dataBuffer.at(0).at(0).size(), 
            "dataBuffer.at(0).at(0).size() must be a multiple of nDofsPerElement"));

    // Each element is treated as it's own state, so the total number of states == nStates * nLocalElements 
    int totalStatesToStore = nLocalElements * nStates;
		mat.resize(totalStatesToStore, nDofsPerElement_ * nTimestepsPerBatch_);

		assert((mat.cols() == nDofsPerElement_ * nTimestepsPerBatch_, "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() == nLocalElements * dataBuffer.at(0).size(), "allocated matrix does not have enough rows for all timesteps and states"));
    
    int t = 0;
    for (const auto& timestep : dataBuffer) {
      int colStart = nDofsPerElement_ * t;
      for (int state = 0; state < nStates; ++state) {
        int stateOffset = state * nLocalElements;
        for (int element = 0; element < nLocalElements; ++element){
          int row = stateOffset + element;
          int elementOffset = element * nDofsPerElement_;
          for (int i = 0; i < nDofsPerElement_; ++i) {
            if (row >= mat.rows() || colStart >= mat.cols() || row < 0 || colStart < 0) {
              std::cout << "attempting to write (" << row  << ", " << colStart << ") from array of size (" << mat.rows() << ", " << mat.cols() << ")" << std::endl;
            }
            mat(row, colStart + i) = timestep.at(state).at(elementOffset + i);
          }
        }
      }
      ++t;
    }
	}
	
	void TimeBatchPreparer::copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) {
    int nLocalElements = dataBuffer.at(0).at(0).size() / nDofsPerElement_;
    int nStates = dataBuffer.at(0).size();
    int t = 0;
    for (auto& timestep : dataBuffer) {
      int colStart = nDofsPerElement_ * t;
      for (int state = 0; state < nStates; ++state) {
        int stateOffset = state * nLocalElements;
        for (int element = 0; element < nLocalElements; ++element){
          int row = stateOffset + element;
          if (row >= mat.rows() || colStart >= mat.cols() || row < 0 || colStart < 0) {
            std::cout << "attempting to read (" << row  << ", " << colStart << ") from array of size (" << mat.rows() << ", " << mat.cols() << ")" << std::endl;
          }
          int elementOffset = element * nDofsPerElement_;
          for (int i = 0; i < nDofsPerElement_; ++i) {
            timestep.at(state).at(elementOffset + i) = mat(row, colStart + i);
          }
        }
      }
      ++t;
    }
  }
} // namespace sparse_nn
