USING(JPEG)

SET(HEADERS
   DocumentViewer.h
   coImageViewer.h
)
SET(SOURCES
   DocumentViewer.cpp
   coImageViewer.cpp
)
cover_add_plugin(DocumentViewer ${COVISE_IMAGE_LIBRARY})
