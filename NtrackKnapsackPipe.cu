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
                getState()->nodeUpperBound = 0.0;
        }


}


__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::run(SubProblem const & inputItem, unsigned int nInputs)
{
	
	unsigned int tid = threadIdx.x;
	if (tid < nInputs) {
	printf("entered, currentItem: %d\n", inputItem.currentItem);
	printf("weights[0], weights[1], profits[0], profits[1]: %f, %f, %f, %f\n", getAppParams()->weights[0], getAppParams()->weights[1], getAppParams()->profits[0], getAppParams()->profits[1]);
	}


	__shared__ double upperBounds[THREADS_PER_BLOCK];

	if (tid == 0) {
		for (unsigned i = 0; i != nInputs; i++) {
			upperBounds[i] = 0;
		}
	}

	__syncthreads();

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
		printf("currentTotalWeight, currentitem is %d \n", inputItem.currentItem);
	}

	if (toPush && inputItem.upperBound < getState()->nodeUpperBound) {
		toPush = 0;
		printf("hi my upperBound is %lf, currentItem is %d \n", inputItem.upperBound, inputItem.currentItem);
	}


	if (toPush && inputItem.currentItem == appParams->maxItems) {
		printf("maxItems reached\n");
		double finalBranchCost = inputItem.currentTotalProfit;
		upperBounds[tid] = finalBranchCost;
		//if(finalBranchCost > appParams->globalUpperBound) {
			// TODO: needs to be modifiable?
			// appParams->globalUpperBound = finalBranchCost;

		//}
		toPush = 0;
	}
	/*
	__syncthreads();

	if (tid == 0) {
		double maximum = 0;
		for (unsigned i = 0; i != nInputs; i++) {
			if (upperBounds[i] > maximum) {
				maximum = upperBounds[i];
			}
		}
		// getState()->nodeUpperBound = maximum;
		getState()->nodeUpperBound = 0;
	}*/

	// __syncthreads();

  	double inputUpperBound;
	SubProblem nextLeft, nextRight;

	if (toPush) {
		printf("calculating upperBound\n");
		inputUpperBound = calculateUpperBound(&inputItem, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
	}
	if (toPush) printf("inputUpperBound: %f, nodeUpperBound: %f\n", inputUpperBound, getState()->nodeUpperBound);	
	if (toPush && inputUpperBound > getState()->nodeUpperBound) {
		nextLeft = inputItem;
		nextRight = inputItem;
		
		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;

		nextLeft.upperBound = inputUpperBound;
		nextRight.upperBound = inputUpperBound;

		nextLeft.currentTotalProfit += appParams->profits[nextLeft.currentItem];
		nextLeft.currentTotalWeight += appParams->weights[nextLeft.currentItem];
	} else {
		toPush = 0;
	}

	__syncthreads();

	push(nextLeft, toPush);
	push(nextRight, toPush);
}

__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::cleanup()
{

}
