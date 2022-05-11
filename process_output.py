
f = open("out_16pipe.txt", "r")
fw = open("16_processed.txt", "w")

line = f.readline()
GPU_time = 0
while line:
    if line.startswith("./Knapsack"):
        #print(line)
        
        fw.write(line)
        GPU_time = 0
    elif line.strip().startswith("Total time"):
        time_line = line.strip()[12:]
        ms_idx = time_line.find('ms')
        GPU_time += float(time_line[:ms_idx])
    elif line.startswith("execu"):
        fw.write("total GPU time:" + str(GPU_time))
        fw.write(line)
    elif line.startswith("max profit"):
        fw.write(line)
        

    line = f.readline()
