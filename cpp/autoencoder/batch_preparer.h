#pragma once

#include <vector>
#include <Eigen/Core>

namespace sparse_nn {
  using Timestep = std::vector<double>;

  class BatchPreparer {
  public:
    BatchPreparer() = default;
    virtual void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) = 0;
    virtual void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) = 0;
  };

  class SpaceBatchPreparer : public BatchPreparer {
  public:
    SpaceBatchPreparer(int dataSize, int nStates);
    virtual void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) override;
    
  private:
    int nStates_;
    int dataSize_;
  };

  class TimeBatchPreparer : public BatchPreparer {
  public:
    TimeBatchPreparer(int dataSize, int nStates, int nTimestepsPerBatch);
    virtual void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) override;
    
  private:
    int nStates_;
    int dataSize_;
    int nTimestepsPerBatch_;
  };
} // namespace sparse_nn
