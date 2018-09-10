set(SERVERMON_INC
  vvservermon.h
  vvservermondialog.h
)

set(SERVERMON_SRC
  vvservermon.cpp
  vvservermondialog.cpp
)

set(SERVERMON_UI
  vvservermondialog.ui
)

qt_wrap_cpp(SERVERMON_MOC_INC ${SERVERMON_INC})
qt_wrap_ui(SERVERMON_UI_INC ${SERVERMON_UI})

# moc'd headers reside here
include_directories(${CMAKE_CURRENT_BINARY_DIR})

deskvox_add_plugin(servermon ${SERVERMON_SRC} ${SERVERMON_MOC_INC} ${SERVERMON_UI_INC})

qt_use_modules(servermon Core Widgets Gui Network Xml)

