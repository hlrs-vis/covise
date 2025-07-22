find_package(Arrow REQUIRED)

if (NOT Arrow_FOUND)
	message("Arrow library not found. Please install it or set the Arrow_DIR variable.")
endif()

set(SOURCES
	arrow.cpp
)

set(HEADERS
	arrow.h
	enums.h
)

set(Name "Apache_Arrow")
add_library(${Name} OBJECT ${SOURCES} ${HEADERS})
target_link_libraries(${Name} PRIVATE Arrow::arrow_shared)
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)