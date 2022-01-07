#ifndef __SUB_PROBLEM
#define __SUB_PROBLEM

#include <cuda_runtime.h>

struct SubProblem {
  unsigned int currentItem;
  double upperBound;
  double currentTotalProfit;
  double currentTotalWeight;
};

#endif
