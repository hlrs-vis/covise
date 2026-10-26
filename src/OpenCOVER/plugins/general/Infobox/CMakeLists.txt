USING(OpenSceneGraph)
USING(VRML)

# TODO: these might be necessary later for point-at-terrain-to-get-value feature
#USING(GDAL)
#USING(PROJ)

set(HEADERS
    Infobox.h
    VrmlNodeInfobox.h
)

set(SOURCES
    Infobox.cpp
    VrmlNodeInfobox.cpp
)


cover_add_plugin(Infobox)
