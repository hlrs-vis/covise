set(HEADERS
  TUIWebviewPlugin.h
)
set(SOURCES
  TUIWebviewPlugin.cpp
)

# build the plugin "TestWebview" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(TestWebview)
