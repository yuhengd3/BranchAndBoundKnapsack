#!/bin/sh

#cp ./testDrivers/SGDriver_long2.cu SGDriver.cu
cd ./testSizes
#sed -i -e 's/__MTR_SOURCE_[0-9]\+/__MTR_SOURCE_76290192/g' *.cuh
sed -i -e 's/__MTR_SINK_[0-9]\+/__MTR_SINK_6389264/g' *.cuh
cd ..

echo 600MB
#########
#600MB EVEN
cp ./testSizes/NQueensApp_dev_600MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out600MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out600MB_even.txt
done

#600MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_600MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out600MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out600MB_rdis.txt
done

echo 700MB
#########
#700MB EVEN
cp ./testSizes/NQueensApp_dev_700MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out700MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out700MB_even.txt
done

#700MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_700MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out700MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out700MB_rdis.txt
done

echo 800MB
#########
#800MB EVEN
cp ./testSizes/NQueensApp_dev_800MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out800MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out800MB_even.txt
done

#800MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_800MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out800MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out800MB_rdis.txt
done

echo 900MB
#########
#900MB EVEN
cp ./testSizes/NQueensApp_dev_900MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out900MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out900MB_even.txt
done

#900MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_900MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out900MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out900MB_rdis.txt
done

echo 1000MB
#########
#1000MB EVEN
cp ./testSizes/NQueensApp_dev_1000MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out1000MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1000MB_even.txt
done

#1000MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_1000MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out1000MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1000MB_rdis.txt
done

echo 1100MB
#########
#1100MB EVEN
cp ./testSizes/NQueensApp_dev_1100MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out1100MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1100MB_even.txt
done

#1100MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_1100MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE   > ./serviceOut/long2/out1100MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1100MB_rdis.txt
done

echo 1200MB
##########
#1200MB EVEN
cp ./testSizes/NQueensApp_dev_1200MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE    > ./serviceOut/long2/out1200MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1200MB_even.txt
done

#1200MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_1200MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE    > ./serviceOut/long2/out1200MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1200MB_rdis.txt
done

echo 1300MB
##########
#1300MB EVEN
cp ./testSizes/NQueensApp_dev_1300MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE    > ./serviceOut/long2/out1300MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1300MB_even.txt
done

#1300MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_1300MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE    > ./serviceOut/long2/out1300MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1300MB_rdis.txt
done

echo 1400MB
##########
#1400MB EVEN
cp ./testSizes/NQueensApp_dev_1400MB_even.cuh NQueensApp_dev.cuh
make
./NQueensEXE    > ./serviceOut/long2/out1400MB_even.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1400MB_even.txt
done

#1400MB REDISTRIBUTE
cp ./testSizes/NQueensApp_dev_1400MB_rdis.cuh NQueensApp_dev.cuh
make
./NQueensEXE    > ./serviceOut/long2/out1400MB_rdis.txt
for (( i=1; i<$1; ++i ))
do
	./NQueensEXE    > ./serviceOut/long2/tmp.txt
	cat ./serviceOut/long2/tmp.txt >> ./serviceOut/long2/out1400MB_rdis.txt
done


./cleanupServiceTest.sh

