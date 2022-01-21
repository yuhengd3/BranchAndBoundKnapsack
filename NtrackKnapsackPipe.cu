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

}


__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::run(SubProblem const & inputItem, unsigned int nInputs)
{
	
	unsigned int tid = threadIdx.x;
	if (tid == 0) {
		getState()->nodeUpperBound = 0.0;
	}

	__shared__ double upperBounds[blockIdx.x] = {0};

	auto appParams = getAppParams();
	int toPush = 1;
	if (toPush && tid >= nInputs) {
		toPush = 0;
	}
	/*
	SubProblem newInput = inputItem;
	newInput.currentItem += 1;
	push(newInput, toPush);
	push(newInput, toPush);
	push(newInput, toPush);
	return;
	*/

	if (toPush && inputItem.currentTotalWeight > appParams->maxCapacity) {
		toPush = 0;
	}

	if (toPush && inputItem.upperBound < getState()->nodeUpperBound) {
		toPush = 0;
	}


	if (toPush && inputItem.currentItem == appParams->maxItems) {
		double finalBranchCost = inputItem.currentTotalProfit;
		upperBounds[tid] = finalBranchCost;
		//if(finalBranchCost > appParams->globalUpperBound) {
			// TODO: needs to be modifiable?
			// appParams->globalUpperBound = finalBranchCost;

		//}
		toPush = 0;
	}
	__syncthreads();

	if (tid == 0) {
		double maximum = 0;
		for (unsigned i = 0; i != blockDim.x; i++) {
			if (upperBounds[i] > maximum) {
				maximum = upperbounds[i];
			}
		}
		getState()->nodeUpperBound = maximum;
	}

	__syncthreads();

  	double inputUpperBound = 10000000;
	SubProblem nextLeft, nextRight;

	if (toPush) {
		inputUpperBound = calculateUpperBound(&inputItem, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
	}
	
	if (toPush && inputUpperBound > appParams->globalUpperBound) {
		nextLeft = inputItem;
		nextRight = inputItem;
		
		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;

		nextLeft.currentTotalProfit += appParams->profits[nextLeft.currentItem];
		nextLeft.currentTotalWeight += appParams->weights[nextLeft.currentItem];
	} else {
		toPush = 0;
	}

	push(nextLeft, toPush);
	push(nextRight, toPush);
}

__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::cleanup()
{

}
