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

set(Name "ApacheArrow")
add_library(${Name} STATIC ${SOURCES} ${HEADERS})
target_link_libraries(${Name} PRIVATE Arrow::arrow_shared)
target_include_directories(${Name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
COVISE_INSTALL_TARGET(${Name})