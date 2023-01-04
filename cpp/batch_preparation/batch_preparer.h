#pragma once

#include <vector>
#include <Eigen/Core>

namespace sparse_nn {
  using FullState = std::vector<double>;
  using Timestep = std::vector<FullState>;

  class BatchPreparer {
  public:
    BatchPreparer() = default;
    virtual void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) = 0;
    virtual void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) = 0;
  };

  class SpaceBatchPreparer : public BatchPreparer {
  public:
    SpaceBatchPreparer();
    virtual void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) override;    
  };

  class TimeBatchPreparer : public BatchPreparer {
  public:
    TimeBatchPreparer(int nDofsPerElement,  int nTimestepsPerBatch);
    virtual void copyVectorToMatrix(Eigen::MatrixXd& mat, const std::vector<Timestep>& dataBuffer) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXd& mat, std::vector<Timestep>& dataBuffer) override;
    
  private:
    int nDofsPerElement_;
    int nTimestepsPerBatch_;
  };
} // namespace sparse_nn
