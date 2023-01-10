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
	inline Eigen::VectorXf subtractAndReturnMins(Eigen::MatrixXf& mat) {
		Eigen::VectorXf mins = mat.rowwise().minCoeff();
		mat.colwise() -= mins;

		return mins;
	}

	inline Eigen::VectorXf divideAndReturnRanges(Eigen::MatrixXf& mat) {
		// keep max and unary separate operations
		// it turns out to be much faster for some reason
		Eigen::VectorXf maxs = mat.rowwise().maxCoeff();
    maxs.noalias() = maxs.unaryExpr([](float x){ return x < 1e-7 ? (float)1e-7 : x; });
		mat.array().colwise() /= maxs.array();
		
		return maxs;
	}
	
	inline Eigen::MatrixXf unnormalize(const Eigen::MatrixXf& mat, const Eigen::VectorXf& mins,
									   const Eigen::VectorXf& ranges) {
		Eigen::MatrixXf matD = (mat.array().colwise() * ranges.array()).colwise() + mins.array();
		return matD;
	}
} // namespace sparse_nn
