#include "BranchAndBoundKnapsack_dev.cuh"

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
A<InputView>::init()
{
        if (threadIdx.x == 0) {
                getState()->nodeLowerBound = getAppParams()->globalLowerBound;
        }


}


__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::run(SubProblem const & inputItem, unsigned int nInputs)
{
	
	unsigned int tid = threadIdx.x;

	__shared__ double upperBounds[THREADS_PER_BLOCK];

	auto appParams = getAppParams();
	int toPush = 1;
	if (toPush && tid >= nInputs) {
		toPush = 0;
	}

	if (toPush && inputItem.currentTotalWeight > appParams->maxCapacity) {
		toPush = 0;
	}

	if (toPush && inputItem.upperBound <= getState()->nodeLowerBound) {
		toPush = 0;
	}

  	double inputUpperBound;
	SubProblem leftSub, rightSub;
	int pushLeft = 0, pushRight = 0;

	if (toPush) {
		inputUpperBound = calculateUpperBound(&inputItem, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
	}
	if (toPush && inputUpperBound > getState()->nodeLowerBound) {
		leftSub = inputItem;
		rightSub = inputItem;
		
		leftSub.currentItem += 1;
		rightSub.currentItem += 1;

		leftSub.currentTotalProfit += appParams->profits[leftSub.currentItem];
		leftSub.currentTotalWeight += appParams->weights[leftSub.currentItem];

		leftSub.upperBound = calculateUpperBound(&leftSub, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
		if (leftSub.upperBound > getState()->nodeLowerBound) {
			pushLeft = 1;
		}
		rightSub.upperBound = calculateUpperBound(&rightSub, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
		if (rightSub.upperBound > getState()->nodeLowerBound) {
			pushRight = 1;
		}
	}

	__syncthreads();

	push(leftSub, pushLeft);
	push(rightSub, pushRight);
}

__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::cleanup()
{

}
