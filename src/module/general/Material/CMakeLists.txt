
ADD_DEFINITIONS(-DMATERIAL)
#INCLUDE_DIRECTORIES()

# TODO: cmakeify use ADD_CUSTOM_COMMAND()

# unix:!macx{
# !contains(BASEARCHSUFFIX,cayman) {
    # filegenerator.target = $$OBJECTS_DIR/file_generator
    # filegenerator.depends = $$IN_PWD/file_generator.cpp
    # filegenerator.commands = $(CXX) $(CXXFLAGS) $(INCPATH) -I$$(COVISEDIR)/src/sys $< -o $@
    # QMAKE_EXTRA_TARGETS += filegenerator
    # POST_TARGETDEPS += $$OBJECTS_DIR/file_generator
  # }
# }

SET(HEADERS
  
)

SET(SOURCES
  ../Collect/Collect.cpp
)

SET(EXTRASOURCES
  ../Collect/Collect.h
#  file_generator.cpp
)

ADD_COVISE_MODULE(Tools Material ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Material  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Material)
