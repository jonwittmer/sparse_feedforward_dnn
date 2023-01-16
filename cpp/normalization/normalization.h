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
    Timer computeMinsTimer = Timer("Compute mins");
    computeMinsTimer.start();
		Eigen::VectorXf mins = mat.rowwise().minCoeff();
    computeMinsTimer.stop();
    
    Timer subtractMinsTimer = Timer("Subtract mins");
    subtractMinsTimer.start();
		mat.colwise() -= mins;
    subtractMinsTimer.stop();

    computeMinsTimer.print();
    subtractMinsTimer.print();
		return mins;
	}

	inline Eigen::VectorXf divideAndReturnRanges(Eigen::MatrixXf& mat) {
		// keep max and unary separate operations
		// it turns out to be much faster for some reason
    Timer computeMaxTimer = Timer("Compute max");
    Timer truncateTimer = Timer("Truncate to 1e-7");
    Timer divideTimer = Timer("Divide by range");

    computeMaxTimer.start();
		Eigen::VectorXf maxs = mat.rowwise().maxCoeff();
    computeMaxTimer.stop();

    truncateTimer.start();
    maxs.noalias() = maxs.unaryExpr([](float x){ return x < 1e-7 ? (float)1e-7 : x; });
    truncateTimer.stop();

    divideTimer.start();
		mat.array().colwise() /= maxs.array();
    divideTimer.stop();

    computeMaxTimer.print();
    truncateTimer.print();
    divideTimer.print();
		
		return maxs;
	}
	
	inline Eigen::MatrixXf unnormalize(const Eigen::MatrixXf& mat, const Eigen::VectorXf& mins,
									   const Eigen::VectorXf& ranges) {
		Eigen::MatrixXf matD = (mat.array().colwise() * ranges.array()).colwise() + mins.array();
		return matD;
	}
} // namespace sparse_nn
