#pragma once
#include "utils/timer.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace sparse_nn {
	// not constants because we are modifying mat within function
	inline Eigen::VectorXd subtractAndReturnMins(Eigen::MatrixXd& mat) {
		Eigen::VectorXd mins = mat.rowwise().minCoeff();
		mat.colwise() -= mins;

		return mins;
	}

	inline Eigen::VectorXd divideAndReturnRanges(Eigen::MatrixXd& mat) {
		// keep max and unary separate operations
		// it turns out to be much faster for some reason
		Eigen::VectorXd maxs = mat.rowwise().maxCoeff();
		maxs.noalias() = maxs.unaryExpr([](double x){ return x < 1e-13 ? 1e-13 : x; });
		mat.array().colwise() /= maxs.array();
		
		return maxs;
	}
	
	inline Eigen::MatrixXd unnormalize(const Eigen::MatrixXf& mat, const Eigen::VectorXd& mins,
									   const Eigen::VectorXd& ranges) {
		Eigen::MatrixXd matD = (mat.cast<double>().array().colwise() * ranges.array()).colwise() + mins.array();
		return matD;
	}
} // namespace sparse_nn
