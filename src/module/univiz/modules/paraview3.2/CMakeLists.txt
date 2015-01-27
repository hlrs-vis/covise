# Set the project/library name here.  Classes should be declared as
# "class VTK_PVLocal_EXPORT vtkFoo", where PVLocal is the name of the
# project set here.
PROJECT(Univiz)

ADD_DEFINITIONS(-DVTK)

# Configure output directories.
SET (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR} CACHE INTERNAL
     "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR} CACHE INTERNAL
     "Single output directory for building all executables.")

SUBDIRS(field_to_lines)
SUBDIRS(FLE)
SUBDIRS(flow_topo)
SUBDIRS(ridge_surface)
SUBDIRS(vortex_cores)
SUBDIRS(vortex_criteria)
SUBDIRS(statistics)
SUBDIRS(write_unstructured)
SUBDIRS(read_unstructured)
