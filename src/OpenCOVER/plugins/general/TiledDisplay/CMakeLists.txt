#OpenCOVER plugins general TiledDisplay
USING(VIRVO)

SET(HEADERS
  IBVerbsTransport.h
  TiledDisplayClient.h
  TiledDisplayClientIBVerbs.h
  TiledDisplayClientVV.h
  TiledDisplayCompositor.h
  TiledDisplayDefines.h
  TiledDisplayDimension.h
  TiledDisplay.h
  TiledDisplayOGLTexQuadCompositor.h
  TiledDisplayOGLTexQuadDrawable.h
  TiledDisplayOSGTexQuadCompositor.h
  TiledDisplayServer.h
  TiledDisplayServerIBVerbs.h
  TiledDisplayServerVV.h
)

SET(SOURCES
  IBVerbsTransport.cpp
  TiledDisplayClient.cpp
  TiledDisplayClientIBVerbs.cpp
  TiledDisplayClientVV.cpp
  TiledDisplayCompositor.cpp
  TiledDisplay.cpp
  TiledDisplayDimension.cpp
  TiledDisplayOGLTexQuadCompositor.cpp
  TiledDisplayOGLTexQuadDrawable.cpp
  TiledDisplayOSGTexQuadCompositor.cpp
  TiledDisplayServer.cpp
  TiledDisplayServerIBVerbs.cpp
  TiledDisplayServerVV.cpp
)
cover_add_plugin(TiledDisplay)
