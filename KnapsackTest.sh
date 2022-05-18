

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

echo 8pipe

cp ./configs/8pipe.mtr NtrackKnapsackPipe.mtr
cp ./configs/8pipeDriver.cu NtrackKnapsackDriver.cu
make
echo "" > ./out_8pipe5.txt
#echo "" > ./out_seq.txt


for ((i=96; i < 800; i+=96))
do
        for ((j = 5; j < 50; j+= 10))
        do
		for ((k = 1; k< 6; k += 1))
		do
                	echo ./Knapsack 53 ${i} ${j} >> ./out_8pipe5.txt
                	./Knapsack 53 ${i} ${j} >> ./out_8pipe5.txt
#			echo ./Knapsack 53 ${i} ${j} >> ./out_seq.txt
#			./knapsack_sequential 53 ${i} ${j} >> ./out_seq.txt
		done
        done
done
