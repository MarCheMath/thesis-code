# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_SOURCE_DIR = /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python

# Utility rule file for rwt_swig_compilation.

# Include the progress variables for this target.
include CMakeFiles/rwt_swig_compilation.dir/progress.make

CMakeFiles/rwt_swig_compilation: CMakeFiles/_rwt.dir/rwtPYTHON.stamp


CMakeFiles/_rwt.dir/rwtPYTHON.stamp: rwt.i
CMakeFiles/_rwt.dir/rwtPYTHON.stamp: rwt.i
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Swig compile rwt.i for python"
	/usr/bin/cmake -E make_directory /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/CMakeFiles/_rwt.dir
	/usr/bin/cmake -E touch /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/CMakeFiles/_rwt.dir/rwtPYTHON.stamp
	/usr/bin/cmake -E env SWIG_LIB=/usr/share/swig/4.0.0 /usr/bin/swig -python -outdir /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python -c++ -I/usr/include/python3.6m -I/homes/numerik/cheng/.local/lib/python2.7/site-packages/numpy/core/include -I/homes/numerik/cheng/localstore/Masterthesis/rwt-master/python -I/homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/../lib/inc -o /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/CMakeFiles/_rwt.dir/rwtPYTHON_wrap.cxx /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/rwt.i

rwt_swig_compilation: CMakeFiles/rwt_swig_compilation
rwt_swig_compilation: CMakeFiles/_rwt.dir/rwtPYTHON.stamp
rwt_swig_compilation: CMakeFiles/rwt_swig_compilation.dir/build.make

.PHONY : rwt_swig_compilation

# Rule to build all files generated by this target.
CMakeFiles/rwt_swig_compilation.dir/build: rwt_swig_compilation

.PHONY : CMakeFiles/rwt_swig_compilation.dir/build

CMakeFiles/rwt_swig_compilation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rwt_swig_compilation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rwt_swig_compilation.dir/clean

CMakeFiles/rwt_swig_compilation.dir/depend:
	cd /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python /homes/numerik/cheng/localstore/Masterthesis/rwt-master/python/CMakeFiles/rwt_swig_compilation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rwt_swig_compilation.dir/depend
