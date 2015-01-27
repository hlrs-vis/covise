# __ autoconverted __
# PROJECT(ReadABAQUS)

#COVISE_UNFINISHED()

#  SET(CMAKE_BUILD_TYPE Release)
USING(ABAQUS)
ADD_DEFINITIONS(
-DABQ_WIN86_64
)

INCLUDE_DIRECTORIES(
  
  
)

SET(HEADERS
  
)

SET(SOURCES
  ChoiceState.cpp
  ComponentTranslator.cpp
  Data.cpp
  Element.cpp
  FieldContainer.cpp
  FieldLabel.cpp
  InstanceMesh.cpp
  LocalCS_Data.cpp
  Location.cpp
  Node.cpp
  NonScalarData.cpp
  ReadABAQUS.cpp
  ResultMesh.cpp
  ScalarData.cpp
  SectionPoint.cpp
  TensorData.cpp
  VectorData.cpp
)

SET(EXTRASOURCES
  ChoiceState.h
  ComponentTranslator.h
  Data.h
  Element.h
  FieldContainer.h
  FieldLabel.h
  InstanceMesh.h
  LocalCS_Data.h
  Location.h
  Node.h
  NonScalarData.h
  ReadABAQUS.h
  ResultMesh.h
  ScalarData.h
  SectionPoint.h
  TensorData.h
  VectorData.h
)

ADD_COVISE_MODULE(IO ReadABAQUS ${EXTRASOURCES} )
# old CONFIG: qt warn_on release incremental link_prl colib coapi coappl warn_off abaqus
# old LIBS: 
# old links: 
TARGET_LINK_LIBRARIES(ReadABAQUS  coApi coAppl coCore ${EXTRA_LIBS})
ADD_COVISE_COMPILE_FLAGS(ReadABAQUS "-fpermissive")
COVISE_WNOERROR(ReadABAQUS)

COVISE_INSTALL_TARGET(ReadABAQUS)
