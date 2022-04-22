#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

project(Knapsack
	LANGUAGES CUDA CXX)

set(MERCATOR_FLAGS -q 2)

add_mercator_app(TARGET BranchAndBoundKnapsack
		 SPECFILE NtrackKnapsackPipe.mtr
		 SOURCES NtrackKnapsackPipe.cu)

add_mercator_executable(TARGET Knapsack
			APPS BranchAndBoundKnapsack
			SOURCES NtrackKnapsackDriver.cu)

