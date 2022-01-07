#include <iostream>
#include <cstdlib>
#include <cmath>

#include "SubProblem.cuh"
#include "BranchAndBoundKnapsack.cuh"

#define MAX_ITEMS 100000
#define MAX_CAPACITY 10000
#define HOST_MAX_ITEM 16

double globalUpperBound = 0.0;

double calcuateUpperBound(unsigned int currentItem, double currentWeight, double currentProfit, unsigned int* weights, unsigned int* profits) {
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
		s.upperBound = calcuateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);
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
        
	unsigned int MAX_OUTPUTS = 1 << 8;
	SubProblem * outputs = new SubProblem[MAX_OUTPUTS];

	Mercator::Buffer<SubProblem> inBuffer(1);
	Mercator::Buffer<SubProblem> outBuffer(MAX_OUTPUTS);

	BranchAndBoundKnapsack app;
	
	inBuffer.set(input, 1);

	app.getParams()->globalUpperBound = 0.0;
	app.getParams()->weights = weights;
	app.getParams()->profits = profits;
	

	app.setSource(inBuffer);
	app.SinkNode.setSink(outBuffer);

        app.run();

	unsigned int outsize = outBuffer.size();
	std::cout << "got " << outsize << " " << std::endl;
        outBuffer.get(outputs, outsize);




        return 0;
}
