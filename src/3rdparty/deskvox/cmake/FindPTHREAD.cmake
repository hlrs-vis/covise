FIND_PATH(PTHREAD_INCLUDE_DIR pthread.h
  /usr/include
  DOC "The directory where pthread.h resides"
)

FIND_LIBRARY(PTHREAD_LIBRARIES pthread
  /usr/lib
  /usr/lib32
  /usr/lib64
  /usr/local/lib
  /opt/local/lib
  DOC "The pthread library"
)

IF(PTHREAD_INCLUDE_DIR AND PTHREAD_LIBRARIES)
  SET(PTHREAD_FOUND 1 CACHE STRING "Set to 1 if pthread is found, 0 otherwise")
ELSE(PTHREAD_INCLUDE_DIR AND PTHREAD_LIBRARIES)
  SET(PTHREAD_FOUND 0 CACHE STRING "Set to 1 if pthread is found, 0 otherwise")
ENDIF(PTHREAD_INCLUDE_DIR AND PTHREAD_LIBRARIES)

MARK_AS_ADVANCED(PTHREAD_FOUND)

