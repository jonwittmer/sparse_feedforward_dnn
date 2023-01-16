#include "batch_preparation/batch_preparer.h"

#include <iostream>
#include <math.h>
#include <vector>

#include <Eigen/Core>

namespace sparse_nn {
  SpaceBatchPreparer::SpaceBatchPreparer() {} 
  
	void SpaceBatchPreparer::copyVectorToMatrix(Eigen::MatrixXf& mat, const std::vector<Timestep>& dataBuffer) {		
    int nStates = dataBuffer.at(0).size();
		int totalStatesToStore = dataBuffer.size() * nStates;
    int dataSize = dataBuffer.at(0).at(0).size();
		mat.resize(totalStatesToStore, dataSize); // eigen resize is a no-op if not needed

		assert((mat.rows() / nStates == dataBuffer.size(), "allocated matrix does not have enough rows for all timesteps and states"));
		
		int currRow = 0;
		for (const auto& fullState : dataBuffer) {
      for(const auto& currState : fullState) {
        for (int i = 0; i < currState.size(); ++i) {
          mat(currRow, i) = static_cast<float>(currState.at(i));
        }
        ++currRow;
			}
		}
	}
	
	void SpaceBatchPreparer::copyMatrixToVector(const Eigen::MatrixXf& mat, std::vector<Timestep>& dataBuffer) {
    int nStates = dataBuffer.at(0).size();
    assert((nStates > 0, "dataBuffer does not have all states allocated"));
    assert((mat.rows() / nStates <= dataBuffer.size(), "dataBuffer does not have enough timesteps allocated"));
		
		for (int i = 0; i < mat.rows(); ++i) {
			int timestepIndex = i / nStates;
      int stateIndex = i % nStates;
			for (int j = 0; j < mat.cols(); ++j) {
				dataBuffer.at(timestepIndex).at(stateIndex).at(j) = static_cast<double>(mat(i, j));
			}
		}
	}



  TimeBatchPreparer::TimeBatchPreparer(int nDofsPerElement, int nTimestepsPerBatch, int nStates, int nRkStages) :
    nDofsPerElement_(nDofsPerElement), nTimestepsPerBatch_(nTimestepsPerBatch),
    nStates_(nStates), nRkStages_(nRkStages) {}
  
	void TimeBatchPreparer::copyVectorToMatrix(Eigen::MatrixXf& mat, const std::vector<Timestep>& dataBuffer) {
    assert((nTimestepsPerBatch_ == dataBuffer.size(), "dataBuffer does not match nTimestepsPerBatch"));

    int nLocalElements = dataBuffer.at(0).at(0).size() / nDofsPerElement_;
    assert((nLocalElements * nDofsPerElement_ == dataBuffer.at(0).at(0).size(), 
            "dataBuffer.at(0).at(0).size() must be a multiple of nDofsPerElement"));

    // Each element is treated as it's own state, so the total number of states == nStates * nLocalElements 
    int totalStatesToStore = nLocalElements * nStates_;
		mat.resize(totalStatesToStore, nDofsPerElement_ * nTimestepsPerBatch_ * nRkStages_);

		assert((mat.cols() == nDofsPerElement_ * nTimestepsPerBatch_ * nRkStages_, "dataBuffer timestep size does not match matrix"));
		assert((mat.rows() == nLocalElements * dataBuffer.at(0).size(), "allocated matrix does not have enough rows for all timesteps and states"));
    
    int t = 0;
    for (const auto& timestep : dataBuffer) {
      for (int state = 0; state < nStates_; ++state) {
        int colStart = nDofsPerElement_ * t * nRkStages_;
        int stateOffset = state * nLocalElements;
        for (int rk = 0; rk < nRkStages_; ++rk) {
          for (int element = 0; element < nLocalElements; ++element) {
            int row = stateOffset + element;
            int elementOffset = element * nDofsPerElement_;
            for (int i = 0; i < nDofsPerElement_; ++i) {
              if (row >= mat.rows() || colStart >= mat.cols() || row < 0 || colStart < 0) {
                std::cout << "attempting to write (" << row  << ", " << colStart << ") from array of size (" << mat.rows() << ", " << mat.cols() << ")" << std::endl;
              }
              mat(row, colStart + i) = static_cast<float>(timestep.at(state + nStates_ * rk).at(elementOffset + i));
            }
          }
          colStart += nDofsPerElement_;
        }
      }
      ++t;
    }
	}
	
	void TimeBatchPreparer::copyMatrixToVector(const Eigen::MatrixXf& mat, std::vector<Timestep>& dataBuffer) {
    int nLocalElements = dataBuffer.at(0).at(0).size() / nDofsPerElement_;
    int t = 0;
    for (auto& timestep : dataBuffer) {
      for (int state = 0; state < nStates_; ++state) {
        int colStart = nDofsPerElement_ * t * nRkStages_;
        int stateOffset = state * nLocalElements;
        for (int rk = 0; rk < nRkStages_; ++rk) {
          for (int element = 0; element < nLocalElements; ++element) {
            int row = stateOffset + element;
            if (row >= mat.rows() || colStart >= mat.cols() || row < 0 || colStart < 0) {
              std::cout << "attempting to read (" << row  << ", " << colStart << ") from array of size (" << mat.rows() << ", " << mat.cols() << ")" << std::endl;
            }
            int elementOffset = element * nDofsPerElement_;
            for (int i = 0; i < nDofsPerElement_; ++i) {
              timestep.at(state + nStates_ * rk).at(elementOffset + i) = static_cast<double>(mat(row, colStart + i));
            }
          }
          colStart += nDofsPerElement_;
        }
      }
      ++t;
    }
  }



  void TimeBatchPreparer::createMapping(int nLocalElements) {
    if (mapCompressionToPde_ == nullptr){
      mapCompressionToPde_ = (int *)malloc(nLocalElements * nTimestepsPerBatch_ * nStates_ * nRkStages_ * sizeof(int));
    } else {
      std::cout << "mapCompressionToPde_ is not nullptr. Only call this function once" << std::endl;
    }
    
    int nRows = nStates_ * nLocalElements;
    for (int r = 0; r < nRows; ++r) {
      int state = r / nLocalElements;
      int element = r % nLocalElements;
      for (int i = 0; i < nTimestepsPerBatch_ * nRkStages_; ++i) {
        int timestep = i / nRkStages_;
        int rk = i % nRkStages_;
        
        int timestep_offset = timestep * nRkStages_ * nStates_ * nLocalElements * nDofsPerElement_;
        int rk_offset = rk * nStates_ * nLocalElements * nDofsPerElement_;
        int state_offset = state * nLocalElements * nDofsPerElement_;
        int element_offset = element * nDofsPerElement_;
        mapCompressionToPde_[r * nTimestepsPerBatch_ * nRkStages_ + i] = timestep_offset + rk_offset + state_offset + element_offset;
      }
    }
  }



  void TimeBatchPreparer::copyVectorToMatrix(Eigen::MatrixXf& mat, const double *dataBuffer, int nLocalElements) {
    if (mapCompressionToPde_ == nullptr) {
      createMapping(nLocalElements);
    }

    mat.resize(nStates_ * nLocalElements, nTimestepsPerBatch_ * nRkStages_ * nDofsPerElement_);
    for (int row = 0; row < mat.rows(); ++row) {
      for (int i = 0; i < nTimestepsPerBatch_ * nRkStages_; ++i) {
        int storageOffset = mapCompressionToPde_[row * nTimestepsPerBatch_ * nRkStages_ + i];
        int colOffset = nDofsPerElement_ * i;
        for (int j = 0; j < nDofsPerElement_; ++j) {
          mat(row, colOffset + j) = static_cast<float>(dataBuffer[storageOffset + j]);
        }
      }
    }


	}
	
  void TimeBatchPreparer::copyMatrixToVector(const Eigen::MatrixXf& mat, double *dataBuffer, int nLocalElements) {
    for (int row = 0; row < mat.rows(); ++row) {
      for (int i = 0; i < nTimestepsPerBatch_ * nRkStages_; ++i) {
        int storageOffset = mapCompressionToPde_[row * nTimestepsPerBatch_ * nRkStages_ + i];
        int colOffset = nDofsPerElement_ * i;
        for (int j = 0; j < nDofsPerElement_; ++j) {
          dataBuffer[storageOffset + j] = static_cast<double>(mat(row, colOffset + j));
        }
      }
    }
  }


  // combined copy / min / max computation
  void TimeBatchPreparer::copyVectorToMatrixWithNormalization(Eigen::MatrixXf& mat, const double *dataBuffer, int nLocalElements, 
                                                              Eigen::VectorXf& mins, Eigen::VectorXf& ranges, int currBatchSize) {
    if (mapCompressionToPde_ == nullptr) {
      createMapping(nLocalElements);
    }

    // make sure eigen data structures are the correct size
    mat.resize(nStates_ * nLocalElements, nTimestepsPerBatch_ * nRkStages_ * nDofsPerElement_);
    mins.resize(mat.rows());
    ranges.resize(mat.rows());

    // loop through each batch element
    for (int row = 0; row < mat.rows(); ++row) {
      // compute min and range
      //float *currMin = mins.data() + row;
      //float *currMax = ranges.data() + row;
      float currMin; 
      float currMax;

      if (currBatchSize == nTimestepsPerBatch_) {
        //*currMin = static_cast<float>(1e30); // really big number
        //*currMax = static_cast<float>(-1e30); // really negative number
        currMin = 1e30;
        currMax = -1e30;
        for (int i = 0; i < nTimestepsPerBatch_ * nRkStages_; ++i) {
          int storageOffset = mapCompressionToPde_[row * nTimestepsPerBatch_ * nRkStages_ + i];
          int colOffset = nDofsPerElement_ * i;
          for (int j = 0; j < nDofsPerElement_; ++j) {
            float currVal = static_cast<float>(dataBuffer[storageOffset + j]);
            currMin = (currVal < currMin) ? currVal : currMin;
            currMax = (currVal > currMax) ? currVal : currMax;
          }
        }
        // convert max to range of data
        currMax = currMax - currMin < 1e-7 ? 1e-7 : currMax - currMin;
    
      } else {
        //std::cout << "batch / matrix mismatch: " << currBatchSize << " vs " << nTimestepsPerBatch_ << std::endl; 
        // direct storage with no normalization
        currMin = 0;
        currMax = 1;
      }
      mins(row) = currMin;
      ranges(row) = currMax;
      
      // do copy with normalization
      for (int i = 0; i < nTimestepsPerBatch_ * nRkStages_; ++i) {
        int storageOffset = mapCompressionToPde_[row * nTimestepsPerBatch_ * nRkStages_ + i];
        int colOffset = nDofsPerElement_ * i;
        for (int j = 0; j < nDofsPerElement_; ++j) {
          float currVal = static_cast<float>(dataBuffer[storageOffset + j]);
          mat(row, colOffset + j) = (currVal - currMin) / currMax; 
        }
      }
    }
	}
	
  void TimeBatchPreparer::copyMatrixToVectorWithUnnormalization(const Eigen::MatrixXf& mat, double *dataBuffer, int nLocalElements, 
                                                                const Eigen::VectorXf& mins, const Eigen::VectorXf& ranges) {
    for (int row = 0; row < mat.rows(); ++row) {
      // lets cast these to double to keep more digits of multiplication
      double currMin = static_cast<double>(mins(row));
      double currRange = static_cast<double>(ranges(row));
      for (int i = 0; i < nTimestepsPerBatch_ * nRkStages_; ++i) {
        int storageOffset = mapCompressionToPde_[row * nTimestepsPerBatch_ * nRkStages_ + i];
        int colOffset = nDofsPerElement_ * i;
        for (int j = 0; j < nDofsPerElement_; ++j) {
          dataBuffer[storageOffset + j] = static_cast<double>(mat(row, colOffset + j)) * currRange + currMin ;
        }
      }
    }
  }
} // namespace sparse_nn
