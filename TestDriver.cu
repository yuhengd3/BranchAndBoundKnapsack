#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stack>
#include <vector>
#include <algorithm>
#include <iterator>

#include "SubProblem.cuh"
#include "BranchAndBoundKnapsack.cuh"

//#define MAX_ITEMS 100000
//#define MAX_CAPACITY 10000
#define HOST_MAX_ITEM 16
unsigned int OUTPUTS_MULTIPLIER = 256; // (1 << 9);
unsigned int MAX_INPUT_ = 200000;
unsigned int HOST_MAX_LEVEL = 16;
double globalLowerBound = 0;

double cpuGlobalLowerBound = 0;

unsigned int srand_seed = 0;
unsigned int MAX_ITEMS = 100000;
double MAX_CAPACITY = 10000;

unsigned counter = 0;

double calculateUpperBound(unsigned int currentItem, double currentWeight, double currentProfit, unsigned int* weights, unsigned int* profits);

void branchCPU(SubProblem s, unsigned int* weights, unsigned int* profits) {
	//Sub-problem is overweight, terminate
	if(s.currentTotalWeight > MAX_CAPACITY) {
		//cout << "Over Capacity  . . ." << endl;
		return;
	}

	counter ++;

	//Sub-problem does not do better than current globalUpperBound
	if(s.upperBound < cpuGlobalLowerBound) {
		//cout << "Upper bound lower than current best . . ." << endl;
		return;
	}

	//If we've reached a leaf node . . .
	if(s.currentItem == MAX_ITEMS) {
		double finalBranchCost = s.currentTotalProfit;
		if(finalBranchCost > cpuGlobalLowerBound) {
			cpuGlobalLowerBound = finalBranchCost;
			// globalBestSubProblem = s;

			//cout << "New Best Branch Profit: " << finalBranchCost << endl;
		}
		return;
	}

	//NOTE: WE DON'T STORE WHICH BRANCH THIS IS, SO WE JUST GO AHEAD AND RECALCULATE
	//THE UPPER BOUND EVERY TIME.  IF WE TRACKED IT, THEN WE COULD RECALCUATE FOR EVERY
	//RIGHT BRANCH ONLY INSTEAD. . .
	//if(s.storedItems[s.currentItem] == false) {
		s.upperBound = calculateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);
	//}

	if(s.upperBound > cpuGlobalLowerBound) {
		SubProblem nextLeft = s;	//Include next item
		SubProblem nextRight = s;	//Exclude next item

		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;

		nextLeft.currentTotalProfit += profits[nextLeft.currentItem];
		nextLeft.currentTotalWeight += weights[nextLeft.currentItem];

		//cout << endl << "Left Branch. . . " << endl;
		branchCPU(nextLeft, weights, profits);
		//cout << endl << "Right Branch. . . " << endl;
		branchCPU(nextRight, weights, profits);
	}
}

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
	unsigned int maxWeight = 1000;
	// unsigned int minProfit = 1;
	// unsigned int maxProfit = 100;

	srand(srand_seed);

	baseWeights[0] = 0;
	baseProfits[0] = 0;
	profitPerWeight[0] = 0.0;

	for (unsigned int i = 1; i < MAX_ITEMS; ++i) {
		baseWeights[i] = rand() % (maxWeight - minWeight) + minWeight;
		// baseProfits[i] = rand() % (maxProfit - minProfit) + minProfit;
		baseProfits[i] = baseWeights[i] + 50;

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
		//
		// baseWeights[index] = baseWeights[j];
		// baseProfits[index] = baseProfits[j];

		// profitPerWeight[index] = baseProfits[index] / baseWeights[index];
		profitPerWeight[index] = 0;
		//
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


int main(int argc, char * argv[]) {
	if (argc != 3) {
		printf("usage: ./Knapsack srand_seed max_items\n");
		return -1;
	}

	srand_seed = atoi(argv[1]);
	MAX_ITEMS = atoi(argv[2]);
	// MAX_CAPACITY = atoi(argv[3]);

	unsigned int weights[MAX_ITEMS];
	unsigned int profits[MAX_ITEMS];

	randomItems(weights, profits);
	MAX_CAPACITY = 0;
	for (unsigned i = 1; i != MAX_ITEMS; i++) {
		MAX_CAPACITY += weights[i];
		// std::cout << (double) profits[i] / weights[i] << std::endl;
	}
	MAX_CAPACITY /= 2;

	std::cout << "finished randomItems" << std::endl;

	// CPU version
	SubProblem s;
	s.currentItem = 0;
	s.currentTotalProfit = 0;
	s.currentTotalWeight = 0;

	s.upperBound = calculateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);

	branchCPU(s, weights, profits);

	std::cout << "finished cpu version: " << cpuGlobalLowerBound << std::endl;	
	std::cout << "counter: " << counter << std::endl;

	globalLowerBound = calculateInitialLowerBound(weights, profits);
	std::cout << "initial global lower bound: " << globalLowerBound << std::endl;


	std::vector<SubProblem> repo[MAX_ITEMS / 8] = {std::vector<SubProblem>()};	
	SubProblem input;
	input.currentItem = 0;
	input.currentTotalProfit = 0;
	input.currentTotalWeight = 0;
	input.upperBound = calculateUpperBound(0, 0, 0, weights, profits);

	branch(input, weights, profits, repo);
       
	unsigned int input_size;
	unsigned int output_size;
	SubProblem * input_ptr = NULL;
	SubProblem * output_ptr = NULL;
	std::vector<SubProblem> leafSubProblems;
	unsigned int num_blocks = 0;
	double * block_bounds = NULL;

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

		// app.getParams()->globalLowerBound = globalLowerBound;
		// app.getParams()->weights = d_weights;
		// app.getParams()->profits = d_profits;
		app.getParams()->maxCapacity = MAX_CAPACITY;
		app.getParams()->maxItems = MAX_ITEMS;
		
		if (num_blocks == 0) {
			num_blocks = app.getNBlocks(); // 184
			block_bounds = (double*) calloc(num_blocks, sizeof(double));
		}
		app.getParams()->numBlocks = num_blocks;
		for (unsigned i = 0; i != num_blocks; i++) {
			block_bounds[i] = globalLowerBound;
		}
		double * d_blockLowerBounds;
		cudaMalloc((void**) &d_blockLowerBounds, num_blocks * sizeof(double));
		cudaMemcpy(d_blockLowerBounds, block_bounds, num_blocks * sizeof(double), cudaMemcpyHostToDevice); 
		app.getParams()->blockLowerBounds = d_blockLowerBounds;

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
        	if (outsize != 0) {
			outBuffer.get(output_ptr, outsize);
			if ((unsigned) index == MAX_ITEMS / 8 - 1) {
				// leaf
				// update global lower boud;
				for (size_t a = 0; a != outsize; a++) {
					if (output_ptr[a].upperBound > globalLowerBound) {
						globalLowerBound = output_ptr[a].upperBound;
					}
				}

				/*
				cudaMemcpy(block_bounds, d_blockLowerBounds, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
				for (size_t a = 0; a != num_blocks; a++) {
					if (block_bounds[a] > globalLowerBound) {
						globalLowerBound = block_bounds[a];
					}
				}
				*/
				std::cout << "current GPU max: " << globalLowerBound << std::endl;

			} else {
				std::copy(output_ptr, output_ptr + outsize, std::back_inserter(repo[index + 1]));
			}
		}

		// std::cout << "currentItem  " << output_ptr[0].currentItem << std::endl;
		delete [] output_ptr;
		delete [] input_ptr;
			
		cudaFree((void*)d_weights);
		cudaFree((void*)d_profits);
		cudaFree((void*)d_blockLowerBounds);

	}
	

	std::cout << "max profit from gpu: " << globalLowerBound << std::endl;

	if (fabs(globalLowerBound - cpuGlobalLowerBound) < 0.0001) {
		std::cout << "both versions got the same result" << std::endl;
	} else {
		std::cout << "cpu got " << cpuGlobalLowerBound << " gpu got " << globalLowerBound << std::endl;
	}

	free(block_bounds);

	/*
	cudaFree((void*)d_weights);
	cudaFree((void*)d_profits);
	*/
        return 0;
}

