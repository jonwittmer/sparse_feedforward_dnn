#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace sparse_nn {
	using MsT = std::chrono::microseconds;
	using TimestampT = decltype(std::chrono::system_clock::now());		

	class Timer {
	public:
	    Timer(const std::string name) {
			name_ = name;
		}
		
		inline void start() {
			startTime_ = std::chrono::system_clock::now();
		}

		inline void stop() {
			endTime_ = std::chrono::system_clock::now();
			duration_ = std::chrono::duration_cast<MsT>(
			    endTime_ - startTime_);
			if (duration_.count() < 0) {
				duration_ = std::chrono::duration_cast<MsT>(
			    endTime_ - endTime_);
			}
		}

		inline int getDuration() {
			return duration_.count();
		}

		inline void print() {
			std::cout << name_ << ": " << getDuration() / 1000.0 << " ms" << std::endl;
		}
		
	private:
		std::string name_;
		TimestampT startTime_;
		TimestampT endTime_;
		MsT duration_;
	};
}
