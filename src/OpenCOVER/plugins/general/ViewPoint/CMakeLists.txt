SET(HEADERS
 BezierCurveVisualizer.h
 FlightPathVisualizer.h
 QuickNavDrawable.h
 ViewPoint.h
 coVRDOMDocument.h
 Interpolator.h
 ViewDesc.h
)
SET(SOURCES
BezierCurveVisualizer.cpp
FlightPathVisualizer.cpp
QuickNavDrawable.cpp
ViewPoint.cpp
coVRDOMDocument.cpp
Interpolator.cpp
ViewDesc.cpp
)
cover_add_plugin(ViewPoint ${COVISE_GRMSG_LIBRARY})
qt_use_modules(ViewPoint Xml)
