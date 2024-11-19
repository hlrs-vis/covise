/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_SCENE_GRAPH_H
#define VR_SCENE_GRAPH_H

/*! \file
 \brief  scene graph class

 \author Daniela Rainer
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   05.09.1996
 \date   10.07.1998 (Performer c++ interface)
 */

#include <util/common.h>
#include <osg/Node>
#include <osg/Matrix>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Multisample>
#include <osg/Material>
#include <osg/ClipNode>
#include <osgFX/Scribe>

#include "ui/Owner.h"

namespace opencover {
namespace ui {
class Menu;
class Action;
class Button;
class SelectionList;
}
}

namespace osg
{
class StateSet;
class ClipNode;
}

namespace vrui
{
class coCombinedButtonInteraction;
}

namespace opencover
{
class COVEREXPORT VRSceneGraph: public ui::Owner
{
public:
    enum WireframeMode {
        Disabled,
        Enabled,
        HiddenLineBlack,
        HiddenLineWhite,
        Points,
    };

    enum MenuMode
    {
        MenuHidden,
        MenuAndObjects,
        MenuOnly,
    };
    VRSceneGraph();
    virtual ~VRSceneGraph();
    static VRSceneGraph *instance();

    bool saveScenegraph(const std::string &filename, bool withMenu=false);
#ifdef PHANTOM_TRACKER
    static void manipulateCallback(void *sceneGraph, buttonSpecCell *spec);
#endif

    osg::MatrixTransform *loadAxisGeode(float scale);
    osg::Node *loadHandIcon(const char *name);
    osg::Node *loadHandLine();

    void addMenuItem(osg::Group *itemGroup);
    osg::Group *getMenuGroup()
    {
        return m_menuGroupNode.get();
    }
    osg::Group *getAlwaysVisibleGroup()
    {
        return m_alwaysVisibleGroupNode.get();
    }
    bool menuVisible() const;
    void toggleMenu();
    void setMenu(MenuMode state);
    void setMenuMode(bool state);
    void applyMenuModeToMenus();
    void toggleHeadTracking(bool state);
    void setObjects(bool state);

    // process key events
    bool keyEvent(int type, int keySym, int mod);
    osg::Group *getScene()
    {
        return m_scene.get();
    };

    osg::Group *getObjectsScene()
    {
        return m_objectsScene.get();
    }

    void config();
    void init();
    void update();

    osg::MatrixTransform *getTransform() const
    {
        return m_objectsTransform.get();
    }
    osg::MatrixTransform *getScaleTransform() const
    {
        return (m_scaleTransform);
    }
    osg::MatrixTransform *getHandTransform() const
    {
        return (m_handTransform.get());
    }
    osg::Vec3 getWorldPointOfInterest() const;
    void getHandWorldPosition(float *, float *, float *);
    void addPointerIcon(osg::Node *node);
    void removePointerIcon(osg::Node *node);

    osg::StateSet *loadDefaultGeostate(osg::Material::ColorMode mode = osg::Material::OFF);
    osg::StateSet *loadGlobalGeostate();
    osg::StateSet *loadUnlightedGeostate(osg::Material::ColorMode mode = osg::Material::OFF);
    osg::StateSet *loadTransparentGeostate(osg::Material::ColorMode mode = osg::Material::OFF);
    osg::StateSet *loadDefaultPointstate(float pointSize = 4, osg::Material::ColorMode mode = osg::Material::OFF);
    void setWireframe(WireframeMode mode);
    void setPointerType(int pointerType);
    int getPointerType()
    {
        return (m_pointerType);
    }

    osg::BoundingSphere getBoundingSphere();

    void setNodeBounds(osg::Node *node, const osg::BoundingSphere *bs);

    bool highQuality() const;

    template <class T>
    T *findFirstNode(const char *name, bool startsWith = false, osg::Node * rootNode = NULL)
    {
        if (!rootNode)
            rootNode = m_objectsRoot;

        if (!rootNode)
            return NULL;

        if (!name)
            return NULL;

        if (!(startsWith ? strncmp(rootNode->getName().c_str(), name, strlen(name)) : strcmp(rootNode->getName().c_str(), name)))
        {
            T *node = dynamic_cast<T *>(rootNode);
            if (node)
                return node;
        }

        osg::Group *group = dynamic_cast<osg::Group *>(rootNode);
        if (group)
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                T *node = findFirstNode<T>(name, startsWith, group->getChild(i));
                if (node)
                {
                    return node;
                }
            }
            return NULL;
        }
        else
        {
            return NULL;
        }
    }

    float scaleMode()
    {
        return m_scaleMode;
    }
    void setScaleMode(float scaleMode)
    {
        m_scaleMode = scaleMode;
    }
    float scaleFactor()
    {
        return m_scaleFactor;
    }
    void setScaleFactor(float scaleFactor, bool sync = true);
    void scaleAllObjects(bool resetView = false, bool simple = false);
    bool isScalingAllObjects() const
    {
        return m_scalingAllObjects;
    }
    void boundingSphereToMatrices(const osg::BoundingSphere &boundingSphere,
                                  bool resetView, osg::Matrix *currentMatrix, float *scaleFactor) const;
    void adjustScale();

    void toggleAxis(bool state);
    void toggleHighQuality(bool state);
    void viewAll(bool resetView = false, bool simple = false);
    float floorHeight()
    {
        return m_floorHeight;
    }
    osg::ClipNode *objectsRoot()
    {
        return m_objectsRoot;
    }

    osg::ref_ptr<osg::Node> getHandSphere()
    {
        return m_handSphere;
    }

    void setMultisampling(osg::Multisample::Mode);

    void setScaleFromButton(float direction);
    void setScaleFactorButton(float f);
    void setScaleRestrictFactor(float min, float max);

    void setRestrictBox(float minX, float maxX, float minY, float maxY, float minZ, float maxZ);

    //Coloring
    void setColor(const char *nodeName, int *color, float transparency);
    void setColor(osg::Geode *geode, int *color, float transparency);
    void setTransparency(const char *nodeName, float transparency);
    void setTransparency(osg::Geode *geode, float transparency);
    void setShader(const char *nodeName, const char *shaderName, const char *paraFloat, const char *paraVec2, const char *paraVec3, const char *paraVec4, const char *paraInt, const char *paraBool, const char *paraMat2, const char *paraMat3, const char *paraMat4);
    void setShader(osg::Geode *geode, const char *shaderName, const char *paraFloat, const char *paraVec2, const char *paraVec3, const char *paraVec4, const char *paraInt, const char *paraBool, const char *paraMat2, const char *paraMat3, const char *paraMat4);
    void setMaterial(const char *nodeName, int *ambient, int *diffuse, int *specular, float shininess, float transparency);
    void setMaterial(osg::Geode *geode, const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency);

    // disables "store scenegraph"
    void protectScenegraph();

    int m_vectorInteractor; //< don't use - for COVISE plugin only

    bool isHighQuality() const;

private:
    static VRSceneGraph *s_instance;
    int readConfigFile();
    void initAxis();
    void initHandDeviceGeometry();
    void initMatrices();
    void initSceneGraph();
    bool saveScenegraph(bool withMenu);
    void applyObjectVisibility();

#ifdef PHANTOM_TRACKER
    int m_forceFeedbackON;
    int m_forceFeedbackMode;
    float m_forceScale;
#endif

    osg::ref_ptr<osg::Group> m_scene, m_objectsScene;
    osg::ref_ptr<osgFX::Scribe> m_lineHider;
    osg::ref_ptr<osg::MatrixTransform> m_handTransform;
    osg::ref_ptr<osg::MatrixTransform> m_handAxisTransform, m_viewerAxisTransform, m_smallSceneAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> m_worldAxis, m_handAxis, m_viewerAxis, m_objectAxis, m_smallSceneAxis;
    osg::ref_ptr<osg::Group> m_menuGroupNode;
    osg::ref_ptr<osg::Group> m_alwaysVisibleGroupNode;
    osg::ref_ptr<osg::MatrixTransform> m_pointerDepthTransform;
    float m_pointerDepth;
    osg::ref_ptr<osg::Node> m_handPlane;
    osg::ref_ptr<osg::Node> m_handLine;
    osg::ref_ptr<osg::Node> m_handNormal;
    osg::ref_ptr<osg::Node> m_handCube;
    osg::ref_ptr<osg::Node> m_handFly;
    osg::ref_ptr<osg::Node> m_handDrive;
    osg::ref_ptr<osg::Node> m_handWalk;
    osg::ref_ptr<osg::Node> m_handPyramid;
    osg::ref_ptr<osg::Node> m_handProbe;
    osg::ref_ptr<osg::Node> m_handAnchor;
    osg::ref_ptr<osg::Node> m_handSphere;

    bool showSmallSceneAxis_;
    bool transparentPointer_;

    osg::StateSet *m_rootStateSet, *m_objectsStateSet;
    osg::ClipNode *m_objectsRoot;

    float m_floorHeight;
    WireframeMode m_wireframe;
    bool m_textured = true; /* =true: textures are drawn as intended */
    bool m_shaders = true; /* =true: shaders are applied */
    bool m_coordAxis = false; /* =true: coord Axis will be drawn */
    MenuMode m_showMenu = MenuAndObjects;
    double m_menuToggleTime = 0.;
    bool m_showObjects = true;
    bool m_firstTime = true;
    bool m_pointerVisible = false;

    osg::Matrix m_invBaseMatrix;
    osg::Matrix m_oldInvBaseMatrix;

    int m_pointerType;

    // attribute SCALE attached to PerformerScene objects:
    // SCALE viewAll                        : scaleMode=1.0
    // SCALE keep                           : scaleMode=0.0
    // SCALE <pos number eg. 2.0 or 0.5>    : scaleMode=<number>
    // no SCALE attribute                   : scaleMode=-1.0
    float m_scaleMode, m_scaleFactor;
    float m_scaleFactorButton;
    float m_scaleRestrictFactorMin, m_scaleRestrictFactorMax;
    float m_transRestrictMinX, m_transRestrictMinY, m_transRestrictMinZ;
    float m_transRestrictMaxX, m_transRestrictMaxY, m_transRestrictMaxZ;
    bool m_scaleAllOn;
    bool m_scalingAllObjects;
    osg::MatrixTransform *m_scaleTransform, *m_handIconScaleTransform, *m_handAxisScaleTransform;
    float m_handIconSize;
    float m_handIconOffset;

    float wiiPos;

    osg::ref_ptr<osg::MatrixTransform> m_objectsTransform;
    osg::ref_ptr<osg::Multisample> m_Multisample;
    typedef std::set<osg::Node *> NodeSet;
    NodeSet m_specialBoundsNodeList;
    void dirtySpecialBounds();

    bool menusAreHidden;
    osg::ref_ptr<osg::Program> emptyProgram_;

    bool isScenegraphProtected_;

    typedef std::map<osg::Drawable *, osg::ref_ptr<osg::Material> > StoredMaterialsMap;
    StoredMaterialsMap storedMaterials;
    void storeMaterial(osg::Drawable *drawable);
    bool m_enableHighQualityOption, m_switchToHighQuality, m_highQuality;
    vrui::coCombinedButtonInteraction *m_interactionHQ;

    ui::Menu *m_miscMenu=nullptr;
    ui::SelectionList *m_drawStyle=nullptr;
    ui::Button *m_trackHead=nullptr;;
    ui::Button *m_hidePointer=nullptr;
    ui::SelectionList *m_showStats=nullptr;
    ui::Button *m_showAxis=nullptr, *m_allowHighQuality=nullptr;
    ui::Button *m_useTextures=nullptr, *m_useShaders=nullptr;
    ui::Button *m_showMenuButton=nullptr;
};
}
#endif
