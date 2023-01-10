#pragma once

#include <vector>
#include <Eigen/Core>

namespace sparse_nn {
  using FullState = std::vector<double>;
  using Timestep = std::vector<FullState>;

  class BatchPreparer {
  public:
    BatchPreparer() = default;
    virtual void copyVectorToMatrix(Eigen::MatrixXf& mat, const std::vector<Timestep>& dataBuffer) = 0;
    virtual void copyMatrixToVector(const Eigen::MatrixXf& mat, std::vector<Timestep>& dataBuffer) = 0;
    virtual void copyVectorToMatrix(Eigen::MatrixXf& mat, const double *dataBuffer, int nLocalElements) = 0;
    virtual void copyMatrixToVector(const Eigen::MatrixXf& mat, double *dataBuffer, int nLocalElements) = 0;
  };

  class SpaceBatchPreparer : public BatchPreparer {
  public:
    SpaceBatchPreparer();
    virtual void copyVectorToMatrix(Eigen::MatrixXf& mat, const std::vector<Timestep>& dataBuffer) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXf& mat, std::vector<Timestep>& dataBuffer) override;    
    virtual void copyVectorToMatrix(Eigen::MatrixXf& mat, const double *dataBuffer, int nLocalElements) override {};
    virtual void copyMatrixToVector(const Eigen::MatrixXf& mat, double *dataBuffer, int nLocalElements) override {};
  };

  class TimeBatchPreparer : public BatchPreparer {
  public:
    TimeBatchPreparer(int nDofsPerElement,  int nTimestepsPerBatch, int nStates, int nRkStages);
    virtual void copyVectorToMatrix(Eigen::MatrixXf& mat, const std::vector<Timestep>& dataBuffer) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXf& mat, std::vector<Timestep>& dataBuffer) override;
    virtual void copyVectorToMatrix(Eigen::MatrixXf& mat, const double *dataBuffer, int nLocalElements) override;
    virtual void copyMatrixToVector(const Eigen::MatrixXf& mat, double *dataBuffer, int nLocalElements) override;

  private:
    int nDofsPerElement_;
    int nTimestepsPerBatch_;
    int nStates_;
    int nRkStages_;
    int *mapCompressionToPde_ = nullptr;

    void createMapping(int nLocalElements);
  };
} // namespace sparse_nn
