USING(OpenSceneGraph)
Using(XercesC)
Using(Boost)
USING(gdal)

# Find the OpenMP package
find_package(OpenMP REQUIRED)

ADD_DEFINITIONS(
)

INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)


SET(SOURCES
  tileCitygml.cpp
)


ADD_COVISE_EXECUTABLE(tileCitygml)

TARGET_LINK_LIBRARIES(tileCitygml coUtil ${EXTRA_LIBS} ${OPENSCENEGRAPH_LIBRARIES} OpenMP::OpenMP_CXX )

COVISE_INSTALL_TARGET(tileCitygml)
