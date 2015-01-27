#OpenCOVER plugins general ParallelRendering

SET(HEADERS
  IBVerbsTransport.h
  ParallelRenderingClient.h
  ParallelRenderingClientIBVerbs.h
  ParallelRenderingClientSocket.h
  ParallelRenderingCompositor.h
  ParallelRenderingDefines.h
  ParallelRenderingDimension.h
  ParallelRendering.h
  ParallelRenderingOGLTexQuadCompositor.h
  ParallelRenderingOGLTexQuadDrawable.h
  ParallelRenderingServer.h
  ParallelRenderingServerIBVerbs.h
  ParallelRenderingServerSocket.h
)

SET(SOURCES
  IBVerbsTransport.cpp
  ParallelRenderingClient.cpp
  ParallelRenderingClientIBVerbs.cpp
  ParallelRenderingClientSocket.cpp
  ParallelRenderingCompositor.cpp
  ParallelRendering.cpp
  ParallelRenderingDimension.cpp
  ParallelRenderingOGLTexQuadCompositor.cpp
  ParallelRenderingServer.cpp
  ParallelRenderingServerIBVerbs.cpp
  ParallelRenderingServerSocket.cpp
)

cover_add_plugin(ParallelRendering)
