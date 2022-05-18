
f = open("out_4pipe5.txt", "r")
fw = open("4pipe5_processed.txt", "w")

line = f.readline()
GPU_time = 0
sub_problem = 0
while line:
    if line.startswith("./Knapsack"):
        print(line)
        
        fw.write(line)
        GPU_time = 0
        sub_problem = 0
    elif line.strip().startswith("Total time"):
        time_line = line.strip()[12:]
        ms_idx = time_line.find('ms')
        GPU_time += float(time_line[:ms_idx])
    elif line.startswith("execu"):
        fw.write("total GPU time:" + str(GPU_time) + "\n")
        fw.write("#subproblems: " + str(sub_problem) + "\n")
        fw.write(line)
    elif line.startswith("max profit"):
        fw.write(line)
    elif line.startswith("blockIdx"):
        line = f.readline()
        while line[0].isnumeric():
            nums = line.split(",")
            if int(nums[1]) != 4:
                sub_problem += int(nums[2])

            line = f.readline()

        

    line = f.readline()
