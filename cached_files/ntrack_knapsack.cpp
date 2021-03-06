#include <iostream>
#include <cstdlib>

// #define WINDOWS

#ifdef WINDOWS
#include <conio.h>
#endif

#include <sys/resource.h>

using namespace std;

// To test these values, need to increase stack size to ~100MB.
// #define MAX_ITEMS 100000
// #define MAX_CAPACITY 10000

unsigned int srand_seed = 0;
int MAX_ITEMS = 100;
int MAX_CAPACITY = 100;



struct subProblem {
	unsigned int currentItem;
	double upperBound;
	double currentTotalProfit;
	double currentTotalWeight;
};


double globalUpperBound = 0.0;
subProblem globalBestSubProblem;


//Calculates the partial upper bound of a given sub-problem greedily.
double calcuateUpperBound(unsigned int currentItem, double currentWeight, double currentProfit, unsigned int* weights, unsigned int* profits) {
	double upperBoundProfit = currentProfit;
	double upperBoundWeight = currentWeight;

	unsigned int i = currentItem + 1;
	while(upperBoundWeight + weights[i] < MAX_CAPACITY && i < MAX_ITEMS) {
		upperBoundProfit += profits[i];
		upperBoundWeight += weights[i];
		++i;
	}

	if(upperBoundWeight < MAX_CAPACITY && i < MAX_ITEMS) {
		double partialCapacity = MAX_CAPACITY - upperBoundWeight;
		double percentage = double(partialCapacity) / double(weights[i]);
		upperBoundProfit += percentage * profits[i];
	}

	return upperBoundProfit;
}

//Create a set of random items, weight and profit ranges can be changed within
//this function.
void randomItems(unsigned int* weights, unsigned int* profits) {
	unsigned int baseWeights[MAX_ITEMS];
	unsigned int baseProfits[MAX_ITEMS];

	double profitPerWeight[MAX_ITEMS];

	unsigned int minWeight = 1;
	unsigned int maxWeight = 1000;
	// unsigned int minProfit = 1;
	// unsigned int maxProfit = 100;

	srand(srand_seed);

	//Set first dummy item
	baseWeights[0] = 0;
	baseProfits[0] = 0;
	profitPerWeight[0] = 0.0;

	double total_weights = 0;

	for(unsigned int i = 1; i < MAX_ITEMS; ++i) {
		baseWeights[i] = rand() % (maxWeight - minWeight) + minWeight;
		baseProfits[i] = baseWeights[i] + 50;

		total_weights += baseWeights[i];

		profitPerWeight[i] = double(baseProfits[i]) / double(baseWeights[i]);
	}

	MAX_CAPACITY = total_weights / 2;

	cout << "MAX_CAPACITY " << MAX_CAPACITY << endl;

	//Sort random items by highest profit per weight to lowest
	unsigned int j = 1;
	while(true) {
		double highestProfitPerWeight = 0.0;
		unsigned int index = 0;
		for(unsigned int i = 1; i < MAX_ITEMS; ++i) {
			if(highestProfitPerWeight < profitPerWeight[i]) {
				highestProfitPerWeight = profitPerWeight[i];
				index = i;
			}
		}

		weights[j] = baseWeights[index];
		profits[j] = baseProfits[index];

		profitPerWeight[index] = 0.0;	//Exclude the found item from further searches

		++j;

		if(j == MAX_ITEMS) {
			break;
		}
	}

	//Set first dummy item
	weights[0] = 0;
	profits[0] = 0;
}

void printItems(unsigned int* weights, unsigned int* profits) {
	for(unsigned int i = 0; i < MAX_ITEMS; ++i) {
		cout << "[" << profits[i] << ", " << weights[i] << "], ";
	}
	cout << endl << endl;
}

void printItemProfitPerWeight(unsigned int* weights, unsigned int* profits) {
	cout << "[" << double(profits[0]) << "/" << double(weights[0]) << "], ";
	for(unsigned int i = 1; i < MAX_ITEMS; ++i) {
		cout << "[" << double(profits[i]) / double(weights[i]) << "], ";
	}
	cout << endl << endl;
}

void printSubProblem(subProblem s, unsigned int* weights, unsigned int* profits) {
	// cout << "currentItem = " << s.currentItem << endl;
	// cout << "upperBound = " << s.upperBound << endl;
	

	cout << "Final Profit = " << s.currentTotalProfit << endl;
	cout << "Final Weight = " << s.currentTotalWeight << endl << endl;
}

//Performs the bounding on a new sub-problem, called every time a new item is being
//added to (left branch) or not added to (right branch) the knapsack.  Prune sub-problem
//if over capacity or profit is worse than the current global upper bound.  Finish
//execution when a leaf node is reached, update global upper bound if neccessary.
void branch(subProblem s, unsigned int* weights, unsigned int* profits) {
	//printSubProblem(s, weights, profits);

	//Sub-problem is overweight, terminate
	if(s.currentTotalWeight > MAX_CAPACITY) {
		//cout << "Over Capacity  . . ." << endl;
		return;
	}

	//Sub-problem does not do better than current globalUpperBound
	if(s.upperBound < globalUpperBound) {
		//cout << "Upper bound lower than current best . . ." << endl;
		return;
	}

	//If we've reached a leaf node . . .
	if(s.currentItem == MAX_ITEMS) {
		double finalBranchCost = s.currentTotalProfit;
		if(finalBranchCost > globalUpperBound) {
			globalUpperBound = finalBranchCost;
			globalBestSubProblem = s;

			//cout << "New Best Branch Profit: " << finalBranchCost << endl;
		}
		return;
	}

	//NOTE: WE DON'T STORE WHICH BRANCH THIS IS, SO WE JUST GO AHEAD AND RECALCULATE
	//THE UPPER BOUND EVERY TIME.  IF WE TRACKED IT, THEN WE COULD RECALCUATE FOR EVERY
	//RIGHT BRANCH ONLY INSTEAD. . .
	//if(s.storedItems[s.currentItem] == false) {
		s.upperBound = calcuateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);
	//}

	if(s.upperBound > globalUpperBound) {
		subProblem nextLeft = s;	//Include next item
		subProblem nextRight = s;	//Exclude next item

		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;

		nextLeft.currentTotalProfit += profits[nextLeft.currentItem];
		nextLeft.currentTotalWeight += weights[nextLeft.currentItem];

		//cout << endl << "Left Branch. . . " << endl;
		branch(nextLeft, weights, profits);
		//cout << endl << "Right Branch. . . " << endl;
		branch(nextRight, weights, profits);
	}
}

int main(int argc, char * argv[]) {
	/*
	const rlim_t kStackSize = 64L * 1024L * 1024L;
	struct rlimit rl;
	if (setrlimit(kStackSize, NULL)) {
		printf("can't set stack size\n");
		return -1;
	}
	*/

	if (argc != 3) {
		cout << "correct usage: ./ntrack <srand_seed> <max_items>" << endl;
		return -1;
	}

	srand_seed = atoi(argv[1]);
	MAX_ITEMS = atoi(argv[2]);

	unsigned int weights[MAX_ITEMS];
	unsigned int profits[MAX_ITEMS];

	randomItems(weights, profits);
	//printItems(weights, profits);
	//printItemProfitPerWeight(weights, profits);

	cout << "Done generating items. . . " << endl;

	subProblem s;
	s.currentItem = 0;
	s.currentTotalProfit = 0;
	s.currentTotalWeight = 0;

	s.upperBound = calcuateUpperBound(s.currentItem, s.currentTotalWeight, s.currentTotalProfit, weights, profits);

	cout << "Initial upper bound: " << s.upperBound << endl;

	
	branch(s, weights, profits);


	printSubProblem(globalBestSubProblem, weights, profits);

#ifdef WINDOWS
	_getch();
#endif
	return 0;
}
