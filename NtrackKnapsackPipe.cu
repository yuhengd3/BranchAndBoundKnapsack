#include "BranchAndBoundKnapsack_dev.cuh"
/*
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
*/
__device__
double calculateUpperBound(const SubProblem * curr, unsigned int* accu_weights, unsigned int* accu_profits, int maxCapacity, int maxItems) {
	int maxWeightLeft = maxCapacity - curr->currentTotalWeight;
	int prevSumWeights = accu_weights[curr->currentItem];
	int weightToFind = maxWeightLeft + prevSumWeights;
	unsigned left = 0, right = maxItems;
	while (left < right - 1) {
		unsigned middle = (left + right) / 2;
		if (accu_weights[middle] > weightToFind) {
			right = middle;
		} else {
			left = middle;
		}
	}
	double upperBoundProfit = curr->currentTotalProfit + accu_profits[left] - accu_profits[curr->currentItem];
	if (weightToFind > accu_weights[left] && left != maxItems - 1) {
		upperBoundProfit += double(weightToFind - accu_weights[left]) / (accu_weights[left+1] - accu_weights[left]) * (accu_profits[left+1] - accu_profits[left]);
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
	} else {
		getState()->num_subs[tid] += 1;
	}

	SubProblem leftSub, rightSub;
	int pushLeft = 0, pushRight = 0;

	double lowerBound = appParams->globalLowerBound;

	if (toPush && inputItem.currentItem < appParams->maxItems && inputItem.upperBound > lowerBound) {
		leftSub = inputItem;
		rightSub = inputItem;
		
		leftSub.currentItem += 1;
		rightSub.currentItem += 1;

		leftSub.currentTotalProfit += appParams->profits[leftSub.currentItem] - appParams->profits[leftSub.currentItem - 1];
		leftSub.currentTotalWeight += appParams->weights[leftSub.currentItem] - appParams->weights[leftSub.currentItem - 1];

		leftSub.upperBound = calculateUpperBound(&leftSub, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
		// printf("leftSub.upperBound: %f\n", leftSub.upperBound);
		if (leftSub.currentTotalWeight <= appParams->maxCapacity && leftSub.upperBound > lowerBound) {
			pushLeft = 1;
		}

		rightSub.upperBound = calculateUpperBound(&rightSub, appParams->weights, appParams->profits, appParams->maxCapacity, appParams->maxItems);
		if (rightSub.upperBound > lowerBound) {
			pushRight = 1;
		}
	}

	__syncthreads();

	push(leftSub, pushLeft);
	push(rightSub, pushRight);
}

__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::init()
{
	if (threadIdx.x == 0) {
		cudaMalloc((void**) &(getState()->num_subs), THREADS_PER_BLOCK * sizeof(unsigned));
		for (unsigned i = 0; i != THREADS_PER_BLOCK; i++) {
			getState()->num_subs[i] = 0;
		}
	}
}


__MDECL__
void BranchAndBoundKnapsack_dev::
A<InputView>::cleanup()
{
	if (threadIdx.x == 0) {
		long total = 0;
		for (unsigned i = 0; i != THREADS_PER_BLOCK; i++) {
			total += getState()->num_subs[i];
		}
		// cudaFree(getState()->num_subs);
		atomicAdd(getAppParams()->global_num_subs, total);
	}
}
