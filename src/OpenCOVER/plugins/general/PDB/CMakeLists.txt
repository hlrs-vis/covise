#OpenCOVER plugins general PDB

USING(Virvo:fileio)
USING(CAVEUI)

SET(HEADERS
  ComputeBBVisitor.h
  FrameVisitor.h
  GeodeCountVisitor.h
  HighDetailTransVisitor.h
  LowDetailTransVisitor.h
  PDBPlugin.h
  TransparentVisitor.h
  coFileBrowser.h
  SequenceViewer.h
  TabbedDialogPanel.h
  SizedPanel.h
  TopsanViewer.h
)

#coFileBrowser.h ComputeBBVisitor.h FrameVisitor.h GeodeCountVisitor.h HighDetailTransVisitor.h LowDetailTransVisitor.h PDBPlugin.h SequenceViewer.h SizedPanel.h SurfaceTransVisitor.h TabbedDialogPanel.h TopsanViewer.h TransparentVisitor.h

SET(SOURCES
  ComputeBBVisitor.cpp
  FrameVisitor.cpp
  GeodeCountVisitor.cpp
  HighDetailTransVisitor.cpp
  LowDetailTransVisitor.cpp
  PDBPlugin.cpp
  TransparentVisitor.cpp
  coFileBrowser.cpp
  SequenceViewer.cpp
  TabbedDialogPanel.cpp
  SizedPanel.cpp
  TopsanViewer.cpp
)

cover_add_plugin(PDB)
covise_wnoerror(PDB)
