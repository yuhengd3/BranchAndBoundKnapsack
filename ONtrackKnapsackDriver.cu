#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stack>
#include <vector>
#include <algorithm>
#include <iterator>

#include "SubProblem.cuh"
#include "BranchAndBoundKnapsack.cuh"

#define MAX_ITEMS 1000
#define MAX_CAPACITY 100
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

int findFirstRemaining(std::vector<SubProblem> repo[]) {
	int index = MAX_ITEMS / 8;
	index -= 1;
	while (index >= 0) {
		if (!repo[index].empty()) {
			break;
		}
		index -= 1;
	}
	return index;
}

int main() {
	unsigned int weights[MAX_ITEMS];
	unsigned int profits[MAX_ITEMS];

	randomItems(weights, profits);

	// std::stack<std::vector<SubProblem>> st;
	std::vector<SubProblem> repo[MAX_ITEMS / 8] = {std::vector<SubProblem>()};	
	SubProblem input;
	input.currentItem = 0;
	input.currentTotalProfit = 0;
	input.currentTotalWeight = 0;
	input.upperBound = calculateUpperBound(0, 0, 0, weights, profits);
	std::cout << "upperBound " << input.upperBound << std::endl;

	std::vector<SubProblem> & vec = repo[0];
	vec.push_back(input);
	// st.push_back(vec);
       
	unsigned int OUTPUTS_MULTIPLIER = 128; // (1 << 8);
	unsigned int MAX_INPUT_ = 100000;

	unsigned int input_size;
	unsigned int output_size;
	SubProblem * input_ptr = NULL;
	SubProblem * output_ptr = NULL;
	std::vector<SubProblem> leafSubProblems;

	// copy weights and profits to gpu memory
	/*
	unsigned * d_weights, * d_profits;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**) &d_weights, MAX_ITEMS * sizeof(unsigned));
       	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc error" << std::endl;
		cudaFree(d_weights);
		return -1;
	}

	cudaStatus = cudaMalloc((void**) &d_profits, MAX_ITEMS * sizeof(unsigned));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc error" << std::endl;
		cudaFree((void*)d_profits);
		return -1;
	}
	*/
	int index = -1;

	while ((index = findFirstRemaining(repo)) > -1) {
		std::cout << "index: " << index << std::endl;
		std::vector<SubProblem> & nextVec = repo[index];
		if (nextVec.size() <= MAX_INPUT_) {
			input_size = nextVec.size();
			input_ptr = new SubProblem[input_size];
			std::copy(nextVec.begin(), nextVec.end(), input_ptr);
			// st.pop_back();
			nextVec.clear();
		} else {
			input_size = MAX_INPUT_;
			input_ptr = new SubProblem[input_size];
			std::copy(nextVec.begin(), nextVec.begin()+MAX_INPUT_, input_ptr);
			// delete from the start of the vector
			nextVec.erase(nextVec.begin(), nextVec.begin()+MAX_INPUT_);
		}

		output_size = input_size * OUTPUTS_MULTIPLIER;
		output_ptr = new SubProblem[output_size];

		Mercator::Buffer<SubProblem> inBuffer(input_size);
		Mercator::Buffer<SubProblem> outBuffer(output_size);

		BranchAndBoundKnapsack app;
	
		inBuffer.set(input_ptr, input_size);

		app.getParams()->globalUpperBound = 0.0;
		// app.getParams()->weights = d_weights;
		// app.getParams()->profits = d_profits;
		app.getParams()->maxCapacity = MAX_CAPACITY;
		app.getParams()->maxItems = MAX_ITEMS;

		unsigned * d_weights, * d_profits;
		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**) &d_weights, MAX_ITEMS * sizeof(unsigned));
       		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMalloc error" << std::endl;
			cudaFree(d_weights);
			return -1;
		}

		cudaStatus = cudaMalloc((void**) &d_profits, MAX_ITEMS * sizeof(unsigned));
		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMalloc error" << std::endl;
			cudaFree((void*)d_profits);
			return -1;
		}

		app.getParams()->weights = d_weights;
		app.getParams()->profits = d_profits;

		app.setSource(inBuffer);
		app.SinkNode.setSink(outBuffer);

        	app.run();

        	// double updatedUpperBound = app.getParams()->globalUpperBound; 
		unsigned int outsize = outBuffer.size();
		std::cout << "got " << outsize << " outputs " << std::endl;
		// std::cout << "upperBound: " << updatedUpperBound << std::endl;
        	if (outsize != 0) {
			outBuffer.get(output_ptr, outsize);
			if (index == MAX_ITEMS / 8 - 1) {
				// leaf
				std::copy(output_ptr, output_ptr + outsize, std::back_inserter(leafSubProblems));
			} else {
				std::copy(output_ptr, output_ptr + outsize, std::back_inserter(repo[index]));
			}
		}

		std::cout << "currentItem  " << output_ptr[0].currentItem << std::endl;
		delete [] output_ptr;
		delete [] input_ptr;
			
		cudaFree((void*)d_weights);
		cudaFree((void*)d_profits);
		
		std::cout << "pipeline done" << std::endl;
		return 0;

	}

	std::cout << "number of total outputs " << leafSubProblems.size() << std::endl;

	/*
	cudaFree((void*)d_weights);
	cudaFree((void*)d_profits);
	*/
        return 0;
}

