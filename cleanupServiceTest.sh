#!/bin/sh

cd ./serviceOut/long
sed -i -e 's/[0-9]\+:\sSched\sLoop\sCount:\s//g' *.txt
sed -i -e 's/RUNNING\sAPP\.\.\.//g' *.txt
sed -i -e 's/APP\sFINISHED!//g' *.txt
sed -i -e 's/Outputs\smatch!//g' *.txt
sed -i -e 's/RUNNING\sAPP\sWITH\s[0-9]\+\sBLOCKS//g' *.txt
sed -i -e 's/\*\*\*Application\sruntimes\s[(]ms[)]://g' *.txt
sed -i -e 's/\sinit:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/\smain:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/\scleanup:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/\s[(][0-9a-zA-Z\.%]\+[)]//g' *.txt

sed -i -e 's/[0-9a-zA-Z\]\+:\smain\skernel\sruntime:\s[0-9]\+//g' *.txt

sed -i -e 's/[0-9]\+:\sSched\sLoop\sCount:\s//g' *.txt
sed -i -e 's/Cumulative\sGain\s=\s[0-9]\+//g' *.txt

sed -i -e 's/Board\ssize\s=\s[0-9]\+;\sGPU\sstarts\sat\srow\s[0-9]\+//g' *.txt
sed -i -e 's/#\sseeds\s=\s[0-9]\+//g' *.txt

sed -i -e 's/Total\sOutput\sSpace\s=\s[0-9]\+//g' *.txt
sed -i -e 's/Total\sNumber\sOutputs\s=\s[0-9]\+//g' *.txt
sed -i -e 's/Expected\sNumber\sOutputs\s=\s[0-9]\+//g' *.txt
sed -i -e 's/\*\*\*Application\sruntimes\s[(]ms[)]://g' *.txt
sed -i -e 's/\sinit:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/\smain:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/\scleanup:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/\s[(][0-9a-zA-Z\.%]\+[)]//g' *.txt
sed -i -e 's/APP\sFINISHED!//g' *.txt
sed -i -e 's/RUNNING\sAPP\sWITH\s[0-9]\+\sBLOCKS//g' *.txt

sed -i -e 's/Running\sNQueens\swith\s[0-9]\+\sblocks//g' *.txt
sed -i -e 's/Done!//g' *.txt

#Output timings to separate file before removal
grep "Total time: " * &>> time.tx
grep "GPU clock rate" * &>> cycle.xt
sed -i -e 's/\sTotal\stime:\s//g' *.tx
sed -i -e 's/ms//g' *.tx
sed -i -e 's/:/,/g' *.tx
sed -i -e 's/\.txt//g' *.tx

sed -i -e 's/\sGPU\sclock\srate\s//g' *.xt
sed -i -e 's/ms//g' *.xt
sed -i -e 's/cycles//g' *.xt
sed -i -e 's/://g' *.xt
sed -i -e 's/[/]//g' *.xt
sed -i -e 's/\.txt//g' *.xt

sed -i -e 's/\sTotal\stime:\s[0-9a-zA-Z\.\s]\+//g' *.txt
sed -i -e 's/GPU\sclock\srate\s[(]cycles[/]ms[)]:\s[0-9]\+//g' *.txt
sed -i -e 's/,/ /g' *.txt
sed -i -e 's/blockIdx\snodeID\suser\soverhead//g' *.txt
#sed -i -e 's/GPU\sblast\sapp\sfinished\.\sNum\sresults:\s[0-9]\+//g' *.txt
#sed -i -e 'H;1h;$!d;x; s/\(.*\)/,/g' *.txt
#tr '\n' ','
#sed -i -e :a -e '$!N;s/ *\n */,/;ta' -e 'P;D' *.txt
cd ..
cd ..

