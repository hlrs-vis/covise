USING(Boost)
SET(HEADERS
DistortViewer.h
Projector.h
ScreenCylinder.h
ScreenLoad.h
VisScene.h
HelpFuncs.h
Scene.h
ScreenDome.h
ScreenPlane.h
XmlTools.h
KoordAxis.h
SceneVis.h
Screen.h
Settings.h
)

SET(SOURCES
DistortViewer.cpp
Projector.cpp
ScreenCylinder.cpp
Settings.cpp
Scene.cpp
ScreenDome.cpp
VisScene.cpp
HelpFuncs.cpp
SceneVis.cpp
ScreenLoad.cpp
XmlTools.cpp
KoordAxis.cpp
Screen.cpp
ScreenPlane.cpp
)



cover_add_plugin(DistortViewer)
