if(NOT OSGANIMATION_FOUND)
return()
endif()

set(HEADERS
    GhostAvatar.h

    controls/BoneParser.h
    controls/RigGeometryBoundsFixer.h
    controls/PlanarAvatarControls.h
    controls/TestAvatarControls.h
    controls/GhostAvatarControls.h

    texture/SplotchTerroirTexture.h
    texture/StripesTerroirTexture.h
    texture/TerroirTexture.h
    texture/RenderToTextureCamera.h

    ui/GhostAvatarControlsUI.h
    ui/CoverDrawObject.h
)

set(SOURCES
    GhostAvatar.cpp

    controls/BoneParser.cpp
    controls/RigGeometryBoundsFixer.cpp
    controls/PlanarAvatarControls.cpp
    controls/TestAvatarControls.cpp
    controls/GhostAvatarControls.cpp

    texture/SplotchTerroirTexture.cpp
    texture/StripesTerroirTexture.cpp
    texture/TerroirTexture.cpp
    texture/RenderToTextureCamera.cpp

    ui/GhostAvatarControlsUI.cpp
    ui/CoverDrawObject.cpp
)

cover_add_plugin(GhostAvatar ${HEADERS} ${SOURCES})

target_link_libraries(GhostAvatar ${OPENSCENEGRAPH_LIBRARIES})