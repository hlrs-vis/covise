INCLUDE_DIRECTORIES(../Colors)
ADD_DEFINITIONS(-DNO_COLORMAP_PARAM)
ADD_COVISE_MODULE(Mapper PredefinedColors ../Colors/Colors.cpp ../Colors/Colors.h)
TARGET_LINK_LIBRARIES(PredefinedColors  coApi coAppl coCore )
COVISE_INSTALL_TARGET(PredefinedColors)

