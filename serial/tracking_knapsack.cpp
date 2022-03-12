#include<iostream>
#include<cstdlib>

#define WINDOWS

#ifdef WINDOWS
#include<conio.h>
#endif

using namespace std;


#define MAX_ITEMS 1000
#define MAX_CAPACITY 10000



struct subProblem {
	unsigned int currentItem;
	double upperBound;
	bool* storedItems;
};


double globalUpperBound = 0.0;
subProblem globalBestSubProblem;


//Calculates the profit up to and INCLUDING the currentItem index.
unsigned int calcuateCurrentProfit(bool* storedItems, unsigned int currentItem, unsigned int* profits) {
	double totalProfit = 0.0;
	
	for(unsigned int i = 0; i < currentItem; ++i) {
		totalProfit += (storedItems[i] ? profits[i] : 0);
	}

	return totalProfit;
}

//Calculates the weight up to and INCLUDING the currentItem index.
unsigned int calcuateCurrentWeight(bool* storedItems, unsigned int currentItem, unsigned int* weights) {
	double totalWeight = 0.0;
	
	for(unsigned int i = 0; i < currentItem; ++i) {
		totalWeight += (storedItems[i] ? weights[i] : 0);
	}

	return totalWeight;
}

//Calculates the partial upper bound of a given sub-problem greedily.
double calcuateUpperBound(bool* storedItems, unsigned int currentItem, unsigned int* weights, unsigned int* profits) {
	double upperBoundProfit = calcuateCurrentProfit(storedItems, currentItem, profits);
	double upperBoundWeight = calcuateCurrentWeight(storedItems, currentItem, weights);

	unsigned int i = currentItem + 1;
	while(upperBoundWeight + weights[i] < MAX_CAPACITY && i < MAX_ITEMS) {
		upperBoundProfit += profits[i];
		upperBoundWeight += weights[i];
		++i;
	}

	if(upperBoundWeight < MAX_CAPACITY && i < MAX_ITEMS) {
		unsigned int partialCapacity = MAX_CAPACITY - upperBoundWeight;
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
	unsigned int maxWeight = 100;
	unsigned int minProfit = 1;
	unsigned int maxProfit = 100;

	srand(0);

	//Set first dummy item
	baseWeights[0] = 0;
	baseProfits[0] = 0;
	profitPerWeight[0] = 0.0;

	for(unsigned int i = 1; i < MAX_ITEMS; ++i) {
		baseWeights[i] = rand() % (maxWeight - minWeight) + minWeight;
		baseProfits[i] = rand() % (maxProfit - minProfit) + minProfit;

		profitPerWeight[i] = double(baseProfits[i]) / double(baseWeights[i]);
	}

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
	cout << "currentItem = " << s.currentItem << endl;
	cout << "upperBound = " << s.upperBound << endl;
	cout << "Included Items: " << endl << "[";
	for(unsigned int i = 1; i < MAX_ITEMS; ++i) {
		cout << (s.storedItems[i] ? "1" : "0") << ", ";
	}
	cout << "]" << endl;

	cout << "Final Profit = " << calcuateCurrentProfit(s.storedItems, s.currentItem, profits) << endl;
	cout << "Final Weight = " << calcuateCurrentWeight(s.storedItems, s.currentItem, weights);
}

//Performs the bounding on a new sub-problem, called every time a new item is being
//added to (left branch) or not added to (right branch) the knapsack.  Prune sub-problem
//if over capacity or profit is worse than the current global upper bound.  Finish
//execution when a leaf node is reached, update global upper bound if neccessary.
void branch(subProblem s, unsigned int* weights, unsigned int* profits) {
	//printSubProblem(s, weights, profits);

	//Sub-problem is overweight, terminate
	if(calcuateCurrentWeight(s.storedItems, s.currentItem, weights) > MAX_CAPACITY) {
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
		unsigned int finalBranchCost = calcuateCurrentProfit(s.storedItems, MAX_ITEMS - 1, profits);
		if(finalBranchCost > globalUpperBound) {
			globalUpperBound = finalBranchCost;
			globalBestSubProblem = s;

			//cout << "New Best Branch Profit: " << finalBranchCost << endl;
		}
		return;
	}

	if(s.storedItems[s.currentItem] == false) {
		s.upperBound = calcuateUpperBound(s.storedItems, s.currentItem, weights, profits);
	}

	if(s.upperBound > globalUpperBound) {
		subProblem nextLeft = s;	//Include next item
		subProblem nextRight = s;	//Exclude next item
		nextLeft.storedItems = new bool [MAX_ITEMS];
		nextRight.storedItems = new bool [MAX_ITEMS];
		for(unsigned int i = 0; i < MAX_ITEMS; ++i) {
			nextLeft.storedItems[i] = s.storedItems[i];
			nextRight.storedItems[i] = s.storedItems[i];
		}
		nextLeft.currentItem += 1;
		nextRight.currentItem += 1;
		nextLeft.storedItems[nextLeft.currentItem] = true;
		nextRight.storedItems[nextRight.currentItem] = false;
		//cout << endl << "Left Branch. . . " << endl;
		branch(nextLeft, weights, profits);
		//cout << endl << "Right Branch. . . " << endl;
		branch(nextRight, weights, profits);
	}
}

int main() {

	unsigned int weights[MAX_ITEMS];
	unsigned int profits[MAX_ITEMS];

	randomItems(weights, profits);
	//printItems(weights, profits);
	//printItemProfitPerWeight(weights, profits);

	subProblem s;
	s.currentItem = 0;
	s.storedItems = new bool [MAX_ITEMS];
	for(unsigned int i = 0; i < MAX_ITEMS; ++i) {
		s.storedItems[i] = false;
	}
	s.upperBound = calcuateUpperBound(s.storedItems, s.currentItem, weights, profits);

	
	branch(s, weights, profits);


	printSubProblem(globalBestSubProblem, weights, profits);


	_getch();
	return 0;
}