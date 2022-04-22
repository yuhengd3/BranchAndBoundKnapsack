

echo 4pipe

cp ./configs/4pipe.mtr NtrackKnapsackPipe.mtr
cp ./configs/4pipeDriver.cu NtrackKnapsackDriver.cu
make
echo "" > ./out_4pipe.txt

for ((i=96; i < 800; i+=96))
do
	for ((j = 5; j < 50; j+= 10))
	do
		echo ./Knapsack 43 ${i} ${j} >> ./out_4pipe.txt
		./Knapsack 43 ${i} ${j} >> ./out_4pipe.txt
	done
done

echo 8pipe
