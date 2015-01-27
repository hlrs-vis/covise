#
# Makefile for "external" (shared lib) compilation of ParaView modules
#


# adapt these two paths accordingly

# this is the path to the CMake package
CMAKE_PATH = /pub/flowvis/local/cmake-2.4

# this is the path where ParaView has been compiled to
PARAVIEW_COMPILED_PATH = /pub/flowvis/local/paraview_dev/2.6.1/ParaView-2.6.1-x86-Shared



CMAKE = $(CMAKE_PATH)/bin/cmake -D ParaView_DIR=$(PARAVIEW_COMPILED_PATH) -D CMAKE_BUILD_TYPE=Release .

all: 
	(cd ./field_to_lines; $(CMAKE); make;)
	(cd ./FLE; $(CMAKE); make;)
	(cd ./flow_topo; $(CMAKE); make;)
	(cd ./ridge_surface; $(CMAKE); make;)
	(cd ./statistics; $(CMAKE); make;)
	(cd ./vortex_cores; $(CMAKE); make;)
	(cd ./vortex_criteria; $(CMAKE); make;)

clean:
	(cd ./field_to_lines; make clean;)
	(cd ./FLE; make clean;)
	(cd ./flow_topo; make clean;)
	(cd ./ridge_surface; make clean;)
	(cd ./statistics; make clean;)
	(cd ./vortex_cores; make clean;)
	(cd ./vortex_criteria; make clean;)
