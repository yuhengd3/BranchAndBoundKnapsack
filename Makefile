# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/research/d.yuheng/m/mercator/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/research/d.yuheng/m/mercator/examples

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	cd /home/research/d.yuheng/m/mercator/examples && $(CMAKE_COMMAND) -E cmake_progress_start /home/research/d.yuheng/m/mercator/examples/CMakeFiles /home/research/d.yuheng/m/mercator/examples/Knapsack//CMakeFiles/progress.marks
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Knapsack/all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/research/d.yuheng/m/mercator/examples/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Knapsack/clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Knapsack/preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Knapsack/preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	cd /home/research/d.yuheng/m/mercator/examples && $(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

# Convenience name for target.
Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/rule:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/rule
.PHONY : Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/rule

# Convenience name for target.
BranchAndBoundKnapsack: Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/rule
.PHONY : BranchAndBoundKnapsack

# fast build rule for target.
BranchAndBoundKnapsack/fast:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/build.make Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/build
.PHONY : BranchAndBoundKnapsack/fast

# Convenience name for target.
Knapsack/CMakeFiles/Knapsack.dir/rule:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Knapsack/CMakeFiles/Knapsack.dir/rule
.PHONY : Knapsack/CMakeFiles/Knapsack.dir/rule

# Convenience name for target.
Knapsack: Knapsack/CMakeFiles/Knapsack.dir/rule
.PHONY : Knapsack

# fast build rule for target.
Knapsack/fast:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/Knapsack.dir/build.make Knapsack/CMakeFiles/Knapsack.dir/build
.PHONY : Knapsack/fast

BranchAndBoundKnapsack_dev_combined.o: BranchAndBoundKnapsack_dev_combined.cu.o
.PHONY : BranchAndBoundKnapsack_dev_combined.o

# target to build an object file
BranchAndBoundKnapsack_dev_combined.cu.o:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/build.make Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/BranchAndBoundKnapsack_dev_combined.cu.o
.PHONY : BranchAndBoundKnapsack_dev_combined.cu.o

BranchAndBoundKnapsack_dev_combined.i: BranchAndBoundKnapsack_dev_combined.cu.i
.PHONY : BranchAndBoundKnapsack_dev_combined.i

# target to preprocess a source file
BranchAndBoundKnapsack_dev_combined.cu.i:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/build.make Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/BranchAndBoundKnapsack_dev_combined.cu.i
.PHONY : BranchAndBoundKnapsack_dev_combined.cu.i

BranchAndBoundKnapsack_dev_combined.s: BranchAndBoundKnapsack_dev_combined.cu.s
.PHONY : BranchAndBoundKnapsack_dev_combined.s

# target to generate assembly for a file
BranchAndBoundKnapsack_dev_combined.cu.s:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/build.make Knapsack/CMakeFiles/BranchAndBoundKnapsack.dir/BranchAndBoundKnapsack_dev_combined.cu.s
.PHONY : BranchAndBoundKnapsack_dev_combined.cu.s

NtrackKnapsackDriver.o: NtrackKnapsackDriver.cu.o
.PHONY : NtrackKnapsackDriver.o

# target to build an object file
NtrackKnapsackDriver.cu.o:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/Knapsack.dir/build.make Knapsack/CMakeFiles/Knapsack.dir/NtrackKnapsackDriver.cu.o
.PHONY : NtrackKnapsackDriver.cu.o

NtrackKnapsackDriver.i: NtrackKnapsackDriver.cu.i
.PHONY : NtrackKnapsackDriver.i

# target to preprocess a source file
NtrackKnapsackDriver.cu.i:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/Knapsack.dir/build.make Knapsack/CMakeFiles/Knapsack.dir/NtrackKnapsackDriver.cu.i
.PHONY : NtrackKnapsackDriver.cu.i

NtrackKnapsackDriver.s: NtrackKnapsackDriver.cu.s
.PHONY : NtrackKnapsackDriver.s

# target to generate assembly for a file
NtrackKnapsackDriver.cu.s:
	cd /home/research/d.yuheng/m/mercator/examples && $(MAKE) $(MAKESILENT) -f Knapsack/CMakeFiles/Knapsack.dir/build.make Knapsack/CMakeFiles/Knapsack.dir/NtrackKnapsackDriver.cu.s
.PHONY : NtrackKnapsackDriver.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... BranchAndBoundKnapsack"
	@echo "... Knapsack"
	@echo "... BranchAndBoundKnapsack_dev_combined.o"
	@echo "... BranchAndBoundKnapsack_dev_combined.i"
	@echo "... BranchAndBoundKnapsack_dev_combined.s"
	@echo "... NtrackKnapsackDriver.o"
	@echo "... NtrackKnapsackDriver.i"
	@echo "... NtrackKnapsackDriver.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	cd /home/research/d.yuheng/m/mercator/examples && $(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

