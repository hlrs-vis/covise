using(Virvo)

set(HEADERS
    FileBrowserPlugin.h SizedPanel.h coFileBrowser.h
)

set(SOURCES
    FileBrowserPlugin.cpp SizedPanel.cpp coFileBrowser.cpp
)

cover_add_plugin(FileBrowser osgcaveui)
