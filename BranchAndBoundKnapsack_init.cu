#include "BranchAndBoundKnapsack_dev.cuh"

#include "hostCode/AppDriver.cuh"

BranchAndBoundKnapsack::BranchAndBoundKnapsack(cudaStream_t stream, int deviceId)
 :
   SinkNode(&allParams.n__MTR_SINK_12877600[0])
{
   if (deviceId == -1) cudaGetDevice(&deviceId);
   driver = new Mercator::AppDriver<Params, BranchAndBoundKnapsack_dev>(stream, deviceId);
}
