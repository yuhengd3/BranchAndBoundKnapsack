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
A<InputView>::run(SubProblem const & inputItem, unsigned int nInputs)
{
	unsigned int tid = threadIdx.x;
	auto appParams = getAppParams();
	int toPush = 1;
	if (toPush && tid >= nInputs) {
		toPush = 0;
	}

	if (toPush && inputItem.currentTotalWeight > appParams->maxCapacity) {
		toPush = 0;
	}

	if (toPush && inputItem.upperBound < appParams->globalUpperBound) {
		toPush = 0;
	}

	if (toPush && inputItem.currentItem == appParams->maxItems) {
		double finalBranchCost = inputItem.currentTotalProfit;
		if(finalBranchCost > appParams->globalUpperBound) {
			// TODO: needs to be modifiable?
			// appParams->globalUpperBound = finalBranchCost;
		}
		toPush = 0;
	}
  	double inputUpperBound;
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
	}

	push(nextLeft, toPush);
	push(nextRight, toPush);
}

