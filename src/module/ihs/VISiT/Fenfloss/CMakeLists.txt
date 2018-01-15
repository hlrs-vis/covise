INCLUDE_DIRECTORIES(
../lib
)

SET(HEADERS
  Fenfloss.h
)
SET(SOURCES
  CreateUserMenu.cpp 
  Fenfloss.cpp
)
covise_add_module(VISiT Fenfloss ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(Fenfloss coApi coAppl coCore visit_general)
covise_install_target(Fenfloss)
