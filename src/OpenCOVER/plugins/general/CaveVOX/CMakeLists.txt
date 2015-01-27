#OpenCOVER plugins general CaveVOX

SET(HEADERS
  CaveVOX.h
  CellMaker.h
  TFColorBar.h
  TFColorBox.h
  TFColorWidget.h
  TFPanel.h
  UserStudy.h
  Virvo.h
  VirvoNode.h
  VirvoPickBox.h
  VolumePickBox.h
  vtkActorToOSG.h
  vtkVolumeGeode.h
  WandInt.h
)

SET(SOURCES
  CaveVOX.cpp
  CellMaker.cpp
  TFColorBar.cpp
  TFColorBox.cpp
  TFColorWidget.cpp
  TFPanel.cpp
  UserStudy.cpp
  Virvo.cpp
  VirvoNode.cpp
  VirvoPickBox.cpp
  VolumePickBox.cpp
  vtkActorToOSG.cpp
  vtkVolumeGeode.cpp
  WandInt.cpp
)

cover_add_plugin(CaveVOX)
