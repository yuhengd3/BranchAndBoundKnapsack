

#echo 4pipe

#cp ./configs/4pipe.mtr NtrackKnapsackPipe.mtr
#cp ./configs/4pipeDriver.cu NtrackKnapsackDriver.cu
#make
#echo "" > ./out_4pipe.txt

#for ((i=96; i < 800; i+=96))
#do
#	for ((j = 5; j < 50; j+= 10))
#	do
#		echo ./Knapsack 43 ${i} ${j} >> ./out_4pipe.txt
#		./Knapsack 43 ${i} ${j} >> ./out_4pipe.txt
#	done
#done
#
#echo 8pipe

#cp ./configs/8pipe.mtr NtrackKnapsackPipe.mtr
#cp ./configs/8pipeDriver.cu NtrackKnapsackDriver.cu
#make
#echo "" > ./out_8pipe.txt

#for ((i=96; i < 800; i+=96))
#do
#        for ((j = 5; j < 50; j+= 10))
#        do
#                echo ./Knapsack 43 ${i} ${j} >> ./out_8pipe.txt
#                ./Knapsack 43 ${i} ${j} >> ./out_8pipe.txt
#        done
#done

echo 4pipe

cp ./configs/4pipe.mtr NtrackKnapsackPipe.mtr
cp ./configs/4pipeDriver.cu NtrackKnapsackDriver.cu
make
echo "" > ./out_4pipe5.txt


for ((i=96; i < 800; i+=96))
do
        for ((j = 5; j < 50; j+= 10))
        do
		for ((k = 1; k< 6; k += 1))
		do
                	echo ./Knapsack 43 ${i} ${j} >> ./out_4pipe5.txt
                	./Knapsack 43 ${i} ${j} >> ./out_4pipe5.txt
		done
        done
done
