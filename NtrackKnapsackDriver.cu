#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stack>
#include <vector>
#include <algorithm>
#include <iterator>

#include "SubProblem.cuh"
#include "BranchAndBoundKnapsack.cuh"

#define MAX_ITEMS 960
#define MAX_CAPACITY 100
#define HOST_MAX_ITEM 16
unsigned int OUTPUTS_MULTIPLIER = 256; // (1 << 9);
unsigned int MAX_INPUT_ = 200000;
unsigned int HOST_MAX_LEVEL = 16;
double globalLowerBound = 0;

double calculateInitialLowerBound(unsigned int * weights, unsigned int * profits) {
	double currProfit = 0;
	double currWeight = 0;

	for (unsigned i = 1; i != MAX_ITEMS; i++) {
		if (currWeight + weights[i] <= MAX_CAPACITY) {
			currWeight += weights[i];
			currProfit += profits[i];
		}
	}
	return currProfit;
}

double calculateUpperBound(unsigned int currentItem, double currentWeight, double currentProfit, unsigned int* weights, unsigned int* profits) {
	double upperBoundProfit = currentProfit;
	double upperBoundWeight = currentWeight;

	unsigned int i = currentItem + 1;
	while (i < MAX_ITEMS && upperBoundWeight + weights[i] < MAX_CAPACITY) {
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

	srand(2);

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

void branch(SubProblem s, unsigned int* weights, unsigned int* profits, std::vector<SubProblem> repo[]) {
	//printSubProblem(s, weights, profits);

	//Sub-problem is overweight, terminate
	if(s.currentTotalWeight > MAX_CAPACITY) {
		//cout << "Over Capacity  . . ." << endl;
		return;
	}

	//Sub-problem does not do better than current globalLowerBound
	if(s.upperBound < globalLowerBound) {
		//cout << "Upper bound lower than current best . . ." << endl;
		return;
	}

	//NOTE: WE DON'T STORE WHICH BRANCH THIS IS, SO WE JUST GO AHEAD AND RECALCULATE
	//THE UPPER BOUND EVERY TIME.  IF WE TRACKED IT, THEN WE COULD RECALCUATE FOR EVERY
	//RIGHT BRANCH ONLY INSTEAD. . .
	//if(s.storedItems[s.currentItem] == false) {
		s.upperBound = calculateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);
	//}

	if(s.upperBound > globalLowerBound) {
		SubProblem nextLeft = s;	//Include next item
		SubProblem nextRight = s;	//Exclude next item

		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;

		nextLeft.currentTotalProfit += profits[nextLeft.currentItem];
		nextLeft.currentTotalWeight += weights[nextLeft.currentItem];

		nextLeft.upperBound = calculateUpperBound(nextLeft.currentItem, nextLeft.currentTotalWeight, nextLeft.currentTotalProfit, weights, profits);
		nextRight.upperBound = calculateUpperBound(nextRight.currentItem, nextRight.currentTotalWeight, nextRight.currentTotalProfit, weights, profits);

	 	if (nextLeft.upperBound > globalLowerBound) {	
			if (nextLeft.currentItem < HOST_MAX_LEVEL) {
				branch(nextLeft, weights, profits, repo);
			} else {
				repo[HOST_MAX_LEVEL / 8].push_back(nextLeft);
			}
		}	
		if (nextRight.upperBound > globalLowerBound) {
		       	if (nextRight.currentItem < HOST_MAX_LEVEL) {
		       		branch(nextRight, weights, profits, repo);
		 	} else {
				repo[HOST_MAX_LEVEL / 8].push_back(nextRight);
			}	
		}
	}
}


int main() {
	unsigned int weights[MAX_ITEMS];
	unsigned int profits[MAX_ITEMS];

	randomItems(weights, profits);

	globalLowerBound = calculateInitialLowerBound(weights, profits);
	std::cout << "initial global lower bound: " << globalLowerBound << std::endl;

	std::vector<SubProblem> repo[MAX_ITEMS / 8] = {std::vector<SubProblem>()};	
	SubProblem input;
	input.currentItem = 0;
	input.currentTotalProfit = 0;
	input.currentTotalWeight = 0;
	input.upperBound = calculateUpperBound(0, 0, 0, weights, profits);
	// std::cout << "upperBound " << input.upperBound << std::endl;

	branch(input, weights, profits, repo);
       
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
			nextVec.clear();
		} else {
			input_size = MAX_INPUT_;
			input_ptr = new SubProblem[input_size];
			std::copy(nextVec.begin(), nextVec.begin()+MAX_INPUT_, input_ptr);
			// delete from the start of the vector
			nextVec.erase(nextVec.begin(), nextVec.begin()+MAX_INPUT_);
		}

		output_size = input_size * OUTPUTS_MULTIPLIER;
		// std::cout << "output size " << output_size << std::endl;
		output_ptr = new SubProblem[output_size];

		Mercator::Buffer<SubProblem> inBuffer(input_size);
		Mercator::Buffer<SubProblem> outBuffer(output_size);

		BranchAndBoundKnapsack app;
	
		inBuffer.set(input_ptr, input_size);

		app.getParams()->globalLowerBound = globalLowerBound;
		// app.getParams()->weights = d_weights;
		// app.getParams()->profits = d_profits;
		app.getParams()->maxCapacity = MAX_CAPACITY;
		app.getParams()->maxItems = MAX_ITEMS;

		unsigned * d_weights, * d_profits;
		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**) &d_weights, MAX_ITEMS * sizeof(unsigned));
		cudaMemcpy(d_weights, weights, MAX_ITEMS * sizeof(unsigned),cudaMemcpyHostToDevice);
       		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMalloc error" << std::endl;
			cudaFree(d_weights);
			return -1;
		}

		cudaStatus = cudaMalloc((void**) &d_profits, MAX_ITEMS * sizeof(unsigned));
		cudaMemcpy(d_profits, profits, MAX_ITEMS * sizeof(unsigned),cudaMemcpyHostToDevice);
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

		unsigned int outsize = outBuffer.size();
		std::cout << "got " << outsize << " outputs " << std::endl;
		// std::cout << "upperBound: " << updatedUpperBound << std::endl;
        	if (outsize != 0) {
			outBuffer.get(output_ptr, outsize);
			if (index == MAX_ITEMS / 8 - 1) {
				// leaf
				// std::copy(output_ptr, output_ptr + outsize, std::back_inserter(leafSubProblems));
				
				// update global lower boud;
				for (size_t a = 0; a != outsize; a++) {
					if (output_ptr[a].upperBound > globalLowerBound) {
						globalLowerBound = output_ptr[a].upperBound;
					}
				}

			} else {
				std::copy(output_ptr, output_ptr + outsize, std::back_inserter(repo[index + 1]));
			}
		}

		// std::cout << "currentItem  " << output_ptr[0].currentItem << std::endl;
		delete [] output_ptr;
		delete [] input_ptr;
			
		cudaFree((void*)d_weights);
		cudaFree((void*)d_profits);
		

	}
	
	/*
	std::cout << "number of total outputs " << leafSubProblems.size() << std::endl;
	double maximum_value = 0;
	for (size_t i = 0; i != leafSubProblems.size(); i++) {
		SubProblem & s = leafSubProblems[i];
		if (maximum_value < s.currentTotalProfit) {
			maximum_value = s.currentTotalProfit;
		}
	}
	*/

	std::cout << "max profit: " << globalLowerBound << std::endl;

	/*
	cudaFree((void*)d_weights);
	cudaFree((void*)d_profits);
	*/
        return 0;
}

