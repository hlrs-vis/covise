#project (TestPipe)

using(qt)


QT_WRAP_CPP(TESTPIPE_MOC_SRCS mainwindow.h)
QT_WRAP_UI(TESTPIPE_UI_HEADERS mainwindow.ui)

ADD_COVISE_EXECUTABLE(TestPipe main.cpp mainwindow.cpp ${TESTPIPE_MOC_SRCS} mainwindow.h ${TESTPIPE_UI_HEADERS})

qt_use_modules(TestPipe Xml UiTools Widgets)
TARGET_LINK_LIBRARIES(TestPipe ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(TestPipe)
