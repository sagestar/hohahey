# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/acs/ByteYoungDB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/acs/ByteYoungDB/build

# Include any dependencies generated for this target.
include src/main/CMakeFiles/bydb.dir/depend.make

# Include the progress variables for this target.
include src/main/CMakeFiles/bydb.dir/progress.make

# Include the compile flags for this target's objects.
include src/main/CMakeFiles/bydb.dir/flags.make

src/main/CMakeFiles/bydb.dir/main.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/main.cpp.o: ../src/main/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/main/CMakeFiles/bydb.dir/main.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/main.cpp.o -c /home/acs/ByteYoungDB/src/main/main.cpp

src/main/CMakeFiles/bydb.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/main.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/main.cpp > CMakeFiles/bydb.dir/main.cpp.i

src/main/CMakeFiles/bydb.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/main.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/main.cpp -o CMakeFiles/bydb.dir/main.cpp.s

src/main/CMakeFiles/bydb.dir/executor.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/executor.cpp.o: ../src/main/executor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/main/CMakeFiles/bydb.dir/executor.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/executor.cpp.o -c /home/acs/ByteYoungDB/src/main/executor.cpp

src/main/CMakeFiles/bydb.dir/executor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/executor.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/executor.cpp > CMakeFiles/bydb.dir/executor.cpp.i

src/main/CMakeFiles/bydb.dir/executor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/executor.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/executor.cpp -o CMakeFiles/bydb.dir/executor.cpp.s

src/main/CMakeFiles/bydb.dir/metadata.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/metadata.cpp.o: ../src/main/metadata.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/main/CMakeFiles/bydb.dir/metadata.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/metadata.cpp.o -c /home/acs/ByteYoungDB/src/main/metadata.cpp

src/main/CMakeFiles/bydb.dir/metadata.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/metadata.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/metadata.cpp > CMakeFiles/bydb.dir/metadata.cpp.i

src/main/CMakeFiles/bydb.dir/metadata.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/metadata.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/metadata.cpp -o CMakeFiles/bydb.dir/metadata.cpp.s

src/main/CMakeFiles/bydb.dir/optimizer.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/optimizer.cpp.o: ../src/main/optimizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/main/CMakeFiles/bydb.dir/optimizer.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/optimizer.cpp.o -c /home/acs/ByteYoungDB/src/main/optimizer.cpp

src/main/CMakeFiles/bydb.dir/optimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/optimizer.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/optimizer.cpp > CMakeFiles/bydb.dir/optimizer.cpp.i

src/main/CMakeFiles/bydb.dir/optimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/optimizer.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/optimizer.cpp -o CMakeFiles/bydb.dir/optimizer.cpp.s

src/main/CMakeFiles/bydb.dir/parser.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/parser.cpp.o: ../src/main/parser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/main/CMakeFiles/bydb.dir/parser.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/parser.cpp.o -c /home/acs/ByteYoungDB/src/main/parser.cpp

src/main/CMakeFiles/bydb.dir/parser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/parser.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/parser.cpp > CMakeFiles/bydb.dir/parser.cpp.i

src/main/CMakeFiles/bydb.dir/parser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/parser.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/parser.cpp -o CMakeFiles/bydb.dir/parser.cpp.s

src/main/CMakeFiles/bydb.dir/storage.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/storage.cpp.o: ../src/main/storage.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/main/CMakeFiles/bydb.dir/storage.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/storage.cpp.o -c /home/acs/ByteYoungDB/src/main/storage.cpp

src/main/CMakeFiles/bydb.dir/storage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/storage.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/storage.cpp > CMakeFiles/bydb.dir/storage.cpp.i

src/main/CMakeFiles/bydb.dir/storage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/storage.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/storage.cpp -o CMakeFiles/bydb.dir/storage.cpp.s

src/main/CMakeFiles/bydb.dir/trx.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/trx.cpp.o: ../src/main/trx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/main/CMakeFiles/bydb.dir/trx.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/trx.cpp.o -c /home/acs/ByteYoungDB/src/main/trx.cpp

src/main/CMakeFiles/bydb.dir/trx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/trx.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/trx.cpp > CMakeFiles/bydb.dir/trx.cpp.i

src/main/CMakeFiles/bydb.dir/trx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/trx.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/trx.cpp -o CMakeFiles/bydb.dir/trx.cpp.s

src/main/CMakeFiles/bydb.dir/util.cpp.o: src/main/CMakeFiles/bydb.dir/flags.make
src/main/CMakeFiles/bydb.dir/util.cpp.o: ../src/main/util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/main/CMakeFiles/bydb.dir/util.cpp.o"
	cd /home/acs/ByteYoungDB/build/src/main && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bydb.dir/util.cpp.o -c /home/acs/ByteYoungDB/src/main/util.cpp

src/main/CMakeFiles/bydb.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bydb.dir/util.cpp.i"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/acs/ByteYoungDB/src/main/util.cpp > CMakeFiles/bydb.dir/util.cpp.i

src/main/CMakeFiles/bydb.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bydb.dir/util.cpp.s"
	cd /home/acs/ByteYoungDB/build/src/main && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/acs/ByteYoungDB/src/main/util.cpp -o CMakeFiles/bydb.dir/util.cpp.s

# Object files for target bydb
bydb_OBJECTS = \
"CMakeFiles/bydb.dir/main.cpp.o" \
"CMakeFiles/bydb.dir/executor.cpp.o" \
"CMakeFiles/bydb.dir/metadata.cpp.o" \
"CMakeFiles/bydb.dir/optimizer.cpp.o" \
"CMakeFiles/bydb.dir/parser.cpp.o" \
"CMakeFiles/bydb.dir/storage.cpp.o" \
"CMakeFiles/bydb.dir/trx.cpp.o" \
"CMakeFiles/bydb.dir/util.cpp.o"

# External object files for target bydb
bydb_EXTERNAL_OBJECTS =

bin/bydb: src/main/CMakeFiles/bydb.dir/main.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/executor.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/metadata.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/optimizer.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/parser.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/storage.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/trx.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/util.cpp.o
bin/bydb: src/main/CMakeFiles/bydb.dir/build.make
bin/bydb: ../sql-parser/lib/libsqlparser.so
bin/bydb: src/main/CMakeFiles/bydb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/acs/ByteYoungDB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable ../../bin/bydb"
	cd /home/acs/ByteYoungDB/build/src/main && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bydb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/main/CMakeFiles/bydb.dir/build: bin/bydb

.PHONY : src/main/CMakeFiles/bydb.dir/build

src/main/CMakeFiles/bydb.dir/clean:
	cd /home/acs/ByteYoungDB/build/src/main && $(CMAKE_COMMAND) -P CMakeFiles/bydb.dir/cmake_clean.cmake
.PHONY : src/main/CMakeFiles/bydb.dir/clean

src/main/CMakeFiles/bydb.dir/depend:
	cd /home/acs/ByteYoungDB/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/acs/ByteYoungDB /home/acs/ByteYoungDB/src/main /home/acs/ByteYoungDB/build /home/acs/ByteYoungDB/build/src/main /home/acs/ByteYoungDB/build/src/main/CMakeFiles/bydb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/main/CMakeFiles/bydb.dir/depend
