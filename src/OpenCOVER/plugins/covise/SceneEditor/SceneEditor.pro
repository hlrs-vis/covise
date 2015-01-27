!include($$(COFRAMEWORKDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TARGET      = SceneEditor
PROJECT     = visenso

TEMPLATE    = opencoverplugin

CONFIG      *= grmsg openpluginutil bullet boost

QT          += xml

SOURCES     = \
        Events/Event.cpp \
        Events/EventSender.cpp \
        Events/PreFrameEvent.cpp \
        Events/MouseEvent.cpp \
        Events/StartMouseEvent.cpp \
        Events/StopMouseEvent.cpp \
        Events/DoMouseEvent.cpp \
        Events/DoubleClickEvent.cpp \
        Events/SetTransformAxisEvent.cpp \
        Events/MouseEnterEvent.cpp \
        Events/MouseExitEvent.cpp \
        Events/SelectEvent.cpp \
        Events/DeselectEvent.cpp \
        Events/MountEvent.cpp \
        Events/UnmountEvent.cpp \
        Events/ApplyMountRestrictionsEvent.cpp \
        Events/GetCameraEvent.cpp \
        Events/RepaintEvent.cpp \
        Events/SetSizeEvent.cpp \
        Events/PostInteractionEvent.cpp \
        Events/TransformChangedEvent.cpp \
        Events/SwitchVariantEvent.cpp \
        Events/SetAppearanceColorEvent.cpp \
        Events/SettingsChangedEvent.cpp \
        Events/MoveObjectEvent.cpp \
        Events/InitKinematicsStateEvent.cpp \
        Behaviors/Behavior.cpp \
        Behaviors/SinusScalingBehavior.cpp \
        Behaviors/CameraBehavior.cpp \
        Behaviors/TransformBehavior.cpp \
        Behaviors/HighlightBehavior.cpp \
        Behaviors/AppearanceBehavior.cpp \
        Behaviors/MyShader.cpp \
        Behaviors/VariantBehavior.cpp \
        Behaviors/MountBehavior.cpp \
        Behaviors/Connectors/Connector.cpp \
        Behaviors/Connectors/PointConnector.cpp \
        Behaviors/Connectors/CeilingConnector.cpp \
        Behaviors/Connectors/FloorConnector.cpp \
        Behaviors/Connectors/WallConnector.cpp \
        Behaviors/Connectors/ShapeConnector.cpp \
        Behaviors/KinematicsBehavior.cpp \
        Behaviors/KardanikXML/Anchor.cpp \
        Behaviors/KardanikXML/Body.cpp \
        Behaviors/KardanikXML/BodyJointDesc.cpp \
        Behaviors/KardanikXML/Construction.cpp \
        Behaviors/KardanikXML/Joint.cpp \
        Behaviors/KardanikXML/KardanikConstructor.cpp \
        Behaviors/KardanikXML/Line.cpp \
        Behaviors/KardanikXML/LineStrip.cpp \
        Behaviors/KardanikXML/Point.cpp \
        Behaviors/KardanikXML/PointRelative.cpp \
        Behaviors/KardanikXML/ConstructionParser.cpp \
        Behaviors/KardanikXML/OperatingRange.cpp \
        Behaviors/KardanikXML/MotionState.cpp \
        SceneUtils.cpp \
        Settings.cpp \
        SceneObject.cpp \
        SceneObjectCreator.cpp \
        Asset.cpp \
        AssetCreator.cpp \
        SceneObjectManager.cpp \
        Room.cpp \
        Wall.cpp \
        Ceiling.cpp \
        Floor.cpp \
        Barrier.cpp \
        RoomCreator.cpp \
        Shape.cpp \
        ShapeSegment.cpp \
        ShapeCreator.cpp \
        Ground.cpp \
        GroundCreator.cpp \
        Light.cpp \
        LightCreator.cpp \
        Window.cpp \
        WindowCreator.cpp \
        SceneEditor.cpp 


EXTRASOURCES    = \
        *.h \
        Behaviors/*.h \
        Events/*.h


### don't modify anything below this line ###
!include ($$(COFRAMEWORKDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)
