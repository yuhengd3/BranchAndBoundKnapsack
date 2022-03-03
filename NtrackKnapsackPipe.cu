#include "BranchAndBoundKnapsack_dev.cuh"

// #include <math.h>

__device__
void findRealGlobalLowerBound(double * arr, size_t sz) {
	// size_t iter = log2(sz - 1) + 1;
	size_t half_sz;

	while (true) {
		__syncthreads();
		half_sz = (sz+1) / 2;
		if (threadIdx.x < sz / 2) {
			if (arr[threadIdx.x] < arr[threadIdx.x + half_sz]) {
				arr[threadIdx.x] = arr[threadIdx.x + half_sz];
			}
		}
		sz = half_sz;
		if (sz == 1) {
			break;
		}
	}
}

__device__
double calculateUpperBound(const SubProblem * curr, unsigned int* weights, unsigned int* profits, int maxCapacity, int maxItems) {
	double upperBoundProfit = curr->currentTotalProfit;
	double upperBoundWeight = curr->currentTotalWeight;

	unsigned int i = curr->currentItem + 1;
	
	while(upperBoundWeight + weights[i] < maxCapacity && i < maxItems) {
		upperBoundProfit += profits[i];
		upperBoundWeight += weights[i];
		++i;
	}

	if(upperBoundWeight < maxCapacity && i < maxItems) {
		double partialCapacity = maxCapacity - upperBoundWeight;
		double percentage = double(partialCapacity) / double(weights[i]);
		upperBoundProfit += percentage * profits[i];
	}

	return upperBoundProfit;
}

__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::run(SubProblem const & inputItem, unsigned int nInputs)
{
	
	unsigned int tid = threadIdx.x;
	auto appParams = getAppParams();

	__shared__ double lowerBounds[THREADS_PER_BLOCK];

	__syncthreads();
	if (tid == 0) {
		for (int i = 0; i != nInputs; i++) {
			lowerBounds[i] = appParams->blockLowerBounds[blockIdx.x];
		}
	}
	__syncthreads();

	int toPush = 1;
	if (toPush && tid >= nInputs) {
		toPush = 0;
	}

	SubProblem leftSub, rightSub;
	int pushLeft = 0, pushRight = 0;
	// __shared__ double block_bounds[appParams->numBlocks];
	__shared__ double block_bounds[400];
	memcpy(block_bounds, appParams->blockLowerBounds, sizeof(double) * appParams->numBlocks);

	findRealGlobalLowerBound(block_bounds, appParams->numBlocks);
	__syncthreads();


	double realGLB = block_bounds[0];

	if (toPush && inputItem.upperBound > realGLB) {
		leftSub = inputItem;
		rightSub = inputItem;
		
		leftSub.currentItem += 1;
		rightSub.currentItem += 1;

		leftSub.currentTotalProfit += appParams->profits[leftSub.currentItem];
		leftSub.currentTotalWeight += appParams->weights[leftSub.currentItem];

		leftSub.upperBound = calculateUpperBound(&leftSub, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
		if (leftSub.currentTotalWeight <= appParams->maxCapacity && leftSub.upperBound > realGLB) {
			if (leftSub.currentItem == appParams->maxItems && leftSub.currentTotalWeight > lowerBounds[tid]) {
				lowerBounds[tid] = leftSub.currentTotalWeight;
			} else if (leftSub.currentItem != appParams->maxItems) {
				pushLeft = 1;
			}
		}

		rightSub.upperBound = calculateUpperBound(&rightSub, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
		if (rightSub.upperBound > realGLB) {
			if (rightSub.currentItem == appParams->maxItems && rightSub.currentTotalWeight > lowerBounds[tid]) {
				lowerBounds[tid] = rightSub.currentTotalWeight;
			} else if (rightSub.currentItem != appParams->maxItems) {
				pushRight = 1;
			}
		}
	}

	__syncthreads();
	
	if (tid == 0) {
		double maximum = appParams->blockLowerBounds[blockIdx.x];
		for (int i = 0; i != nInputs; i++) {
			if (lowerBounds[i] > maximum) {
				maximum = lowerBounds[i];
			}
		}
		appParams->blockLowerBounds[blockIdx.x] = maximum;
	}
	__syncthreads();

	push(leftSub, pushLeft);
	push(rightSub, pushRight);
}

/*
__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::init()
{
        if (threadIdx.x == 0) {
                // getState()->nodeLowerBound = getAppParams()->globalLowerBound;
                // printf("blockLowerBound: %f\n", getAppParams()->blockLowerBounds[0]);
                // printf("blockid:%d\n", blockIdx.x);
        }
}
*/

/*
__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::cleanup()
{

}
*/
