USING(GDAL)
USING(XERCESC)
USING(OSGTERRAIN)
USING(V8)
USING(VehicleUtil)

ADD_DEFINITIONS(-DVPB_LIBRARY)
SET(LIB_NAME vpb)

SET(HEADER_PATH ../../include/${LIB_NAME})
SET(VPB_HEADERS
    ${HEADER_PATH}/BlockOperation
    ${HEADER_PATH}/BuildLog
    ${HEADER_PATH}/BuildOperation
    ${HEADER_PATH}/BuildOptions
    ${HEADER_PATH}/Commandline
    ${HEADER_PATH}/DatabaseBuilder
    ${HEADER_PATH}/DataSet
    ${HEADER_PATH}/Date
    ${HEADER_PATH}/Destination
    ${HEADER_PATH}/Export
    ${HEADER_PATH}/ExtrudeVisitor
    ${HEADER_PATH}/FileCache
    ${HEADER_PATH}/FileDetails
    ${HEADER_PATH}/FileUtils
    ${HEADER_PATH}/FilePathManager
    ${HEADER_PATH}/GeospatialDataset
    ${HEADER_PATH}/HeightFieldMapper
    ${HEADER_PATH}/MachinePool
    ${HEADER_PATH}/ObjectPlacer
    ${HEADER_PATH}/PropertyFile
    ${HEADER_PATH}/ShapeFilePlacer
    ${HEADER_PATH}/Source
    ${HEADER_PATH}/SourceData
    ${HEADER_PATH}/SpatialProperties
    ${HEADER_PATH}/System
    ${HEADER_PATH}/TextureUtils
    ${HEADER_PATH}/Task
    ${HEADER_PATH}/TaskManager
    ${HEADER_PATH}/ThreadPool
    ${HEADER_PATH}/Version
    ${HEADER_PATH}/VPBRoad
)

SET(VPB_SOURCES
    BuildLog.cpp
    BuildOperation.cpp
    BuildOptions.cpp
    BuildOptionsIO.cpp
    Commandline.cpp
    DatabaseBuilder.cpp
    DatabaseBuilderIO.cpp
    DataSet.cpp
    Date.cpp
    Destination.cpp
    ExtrudeVisitor.cpp
    FileCache.cpp
    FileDetails.cpp
    FileUtils.cpp
    FilePathManager.cpp
    GeospatialDataset.cpp
    HeightFieldMapper.cpp
    MachinePool.cpp
    ObjectPlacer.cpp
    PropertyFile.cpp
    ShapeFilePlacer.cpp
    Source.cpp
    SourceData.cpp
    SpatialProperties.cpp
    System.cpp
    TextureUtils.cpp
    Task.cpp
    TaskManager.cpp
    ThreadPool.cpp
    Version.cpp
    VPBRoad.cpp
)

INCLUDE_DIRECTORIES(../../include ${OPENSCENEGRAPH_INCLUDE_DIRS} )

ADD_COVISE_LIBRARY(coVPB SHARED ${VPB_SOURCES} ${VPB_HEADERS})
if(NOT MSVC)
    add_covise_compile_flags(coVPB "-Wno-error=unused-result")
endif()
TARGET_LINK_LIBRARIES(coVPB ${OPENSCENEGRAPH_LIBRARIES} ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(coVPB)

