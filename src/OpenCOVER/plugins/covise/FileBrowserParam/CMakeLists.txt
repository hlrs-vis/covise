SET(HEADERS
)
SET(SOURCES
  FileBrowser.cpp
)

include_directories(..)
cover_add_plugin(FileBrowserParam ${HEADERS} ${SOURCES})
target_link_libraries(FileBrowserParam CovisePluginUtil)

