TEMPLATE	= app
LANGUAGE	= C++

CONFIG	+= qt warn_on release

FORMS	= Tecplot2CoviseGuiBase.ui \
	Basic2DGridBase.ui \
	BottomBase.ui \
	WatersurfaceBase.ui \
	Vector2DVariableBase.ui \
	ScalarVariableBase.ui \
	Basic3DGridBase.ui \
	Vector3DVariableBase.ui

IMAGES	= ../../designerfiles/images/wuerfel.png

unix {
  UI_DIR = .ui
  MOC_DIR = .moc
  OBJECTS_DIR = .obj
}



