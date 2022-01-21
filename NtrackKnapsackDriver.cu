#include <iostream>
#include <cstdlib>
#include <cmath>

#include "SubProblem.cuh"
#include "BranchAndBoundKnapsack.cuh"

#define MAX_ITEMS 100000
#define MAX_CAPACITY 10000
#define HOST_MAX_ITEM 16

double globalUpperBound = 0.0;

double calculateUpperBound(unsigned int currentItem, double currentWeight, double currentProfit, unsigned int* weights, unsigned int* profits) {
	double upperBoundProfit = currentProfit;
	double upperBoundWeight = currentWeight;

	unsigned int i = currentItem + 1;
	while (upperBoundWeight + weights[i] < MAX_CAPACITY && i < MAX_ITEMS) {
		upperBoundProfit += profits[i];
		upperBoundWeight += weights[i];
		++i;
	}

	if (upperBoundWeight < MAX_CAPACITY && i < MAX_ITEMS) {
		double partialCapacity = MAX_CAPACITY - upperBoundWeight;
		double percentage = double(partialCapacity) / double(weights[i]);
		upperBoundProfit += percentage * profits[i];
	}

	return upperBoundProfit;
}

void randomItems(unsigned int * weights, unsigned int * profits) {
	unsigned int baseWeights[MAX_ITEMS];
	unsigned int baseProfits[MAX_ITEMS];

	double profitPerWeight[MAX_ITEMS];

	unsigned int minWeight = 1;
	unsigned int maxWeight = 100;
	unsigned int minProfit = 1;
	unsigned int maxProfit = 100;

	srand(0);

	baseWeights[0] = 0;
	baseProfits[0] = 0;
	profitPerWeight[0] = 0.0;

	for (unsigned int i = 1; i < MAX_ITEMS; ++i) {
		baseWeights[i] = rand() % (maxWeight - minWeight) + minWeight;
		baseProfits[i] = rand() % (maxProfit - minProfit) + minProfit;

		profitPerWeight[i] = double(baseProfits[i]) / double(baseWeights[i]);
	}

	// sort
	unsigned int j = 1;
	while (true) {
		double highestProfitPerWeight = 0.0;
		unsigned int index = 0;
		for (unsigned int i = 1; i < MAX_ITEMS; ++i) {
			if (highestProfitPerWeight < profitPerWeight[i]) {
				highestProfitPerWeight = profitPerWeight[i];
				index = i;
			}
		}

		weights[j] = baseWeights[index];
		profits[j] = baseProfits[index];

		profitPerWeight[index] = 0.0;

		++j;

		if (j == MAX_ITEMS) {
			break;
		}
	}

	weights[0] = 0;
	profits[0] = 0;
}

void branch(SubProblem s, unsigned int* weights, unsigned int* profits) {
	if(s.currentTotalWeight > MAX_CAPACITY) {
		return;
	}
	if(s.upperBound < globalUpperBound) {
		return;
	}

	//If we've reached a leaf node . . .
	if(s.currentItem == MAX_ITEMS) {
		double finalBranchCost = s.currentTotalProfit;
		if(finalBranchCost > globalUpperBound) {
			globalUpperBound = finalBranchCost;
			// globalBestSubProblem = s;
		}
		return;
	}

	//if(s.storedItems[s.currentItem] == false) {
		s.upperBound = calculateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);
	//}

	if (s.upperBound > globalUpperBound) {
		
		SubProblem nextLeft = s;	//Include next item
		SubProblem nextRight = s;	//Exclude next item

		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;

		nextLeft.currentTotalProfit += profits[nextLeft.currentItem];
		nextLeft.currentTotalWeight += weights[nextLeft.currentItem];

		branch(nextLeft, weights, profits);
		branch(nextRight, weights, profits);
	}
}

int main() {
	unsigned int weights[MAX_ITEMS];
	unsigned int profits[MAX_ITEMS];

	randomItems(weights, profits);
        
	SubProblem * input = new SubProblem();
	input->currentItem = 0;
	input->currentTotalProfit = 0;
	input->currentTotalWeight = 0;
	input->upperBound = calculateUpperBound(0, 0, 0, weights, profits);
        
	unsigned int MAX_OUTPUTS = 100000;
	SubProblem * outputs = new SubProblem[MAX_OUTPUTS];

	Mercator::Buffer<SubProblem> inBuffer(1);
	Mercator::Buffer<SubProblem> outBuffer(MAX_OUTPUTS);

	BranchAndBoundKnapsack app;
	
	inBuffer.set(input, 1);

	app.getParams()->globalUpperBound = 0.0;

	unsigned * d_weights, * d_profits;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**) &d_weights, MAX_ITEMS * sizeof(unsigned));
	cudaMemcpy(d_weights, weights, MAX_ITEMS * sizeof(unsigned), cudaMemcpyHostToDevice);
       	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc error" << std::endl;
		cudaFree(d_weights);
		return -1;
	}

	cudaStatus = cudaMalloc((void**) &d_profits, MAX_ITEMS * sizeof(unsigned));
	cudaMemcpy(d_profits, profits, MAX_ITEMS * sizeof(unsigned), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc error" << std::endl;
		cudaFree((void*)d_profits);
		return -1;
	}

	app.getParams()->weights = d_weights;
	app.getParams()->profits = d_profits;
	app.getParams()->maxCapacity = MAX_CAPACITY;
	app.getParams()->maxItems = MAX_ITEMS;
	

	app.setSource(inBuffer);
	app.SinkNode.setSink(outBuffer);

        app.run();

        double updatedUpperBound = app.getParams()->globalUpperBound; 

	unsigned int outsize = outBuffer.size();
	std::cout << "got " << outsize << " " << std::endl;
	std::cout << "upperBound: " << updatedUpperBound << std::endl;
	std::cout << "currentitem ";
        outBuffer.get(outputs, outsize);
	std::cout << outputs->currentItem << std::endl;

	cudaFree((void*)d_weights);
	cudaFree((void*)d_profits);
        return 0;
}

