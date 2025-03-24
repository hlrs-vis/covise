/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <util/common.h>
#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>
#include <vsg/nodes/Node.h>
#include <vsg/nodes/Group.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>

#include "ui/Owner.h"
#include <../../OpenCOVER/OpenVRUI/coCombinedButtonInteraction.h>




namespace vive
{
    namespace ui {
        class Menu;
        class Action;
        class Button;
        class SelectionList;
    }
class VVCORE_EXPORT vvSceneGraph: public ui::Owner
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
    vvSceneGraph();
    virtual ~vvSceneGraph();
    static vvSceneGraph *instance();

    bool saveScenegraph(const std::string &filename, bool withMenu=false);
#ifdef PHANTOM_TRACKER
    static void manipulateCallback(void *sceneGraph, buttonSpecCell *spec);
#endif

    vsg::ref_ptr<vsg::MatrixTransform> loadAxisGeode(float scale);
    vsg::ref_ptr<vsg::Node> loadHandIcon(const std::string& name);
    vsg::ref_ptr<vsg::Node> loadHandLine();

    void addMenuItem(vsg::Group *itemGroup);
    vsg::ref_ptr<vsg::Group> getMenuGroup()
    {
        return m_menuGroupNode;
    }
    vsg::ref_ptr<vsg::Group> getAlwaysVisibleGroup()
    {
        return m_alwaysVisibleGroupNode;
    }
    bool menuVisible() const;
    void toggleMenu();
    void setMenu(MenuMode state);
    void setMenuMode(bool state);
    void applyMenuModeToMenus();
    void toggleHeadTracking(bool state);
    void setObjects(bool state);

    // process key events
    bool keyEvent(vsg::KeyPressEvent& keyPress);
    vsg::ref_ptr<vsg::Group> getScene()
    {
        return m_scene;
    };

    vsg::ref_ptr<vsg::Group>getObjectsScene()
    {
        return m_objectsScene;
    }

    void config();
    void init();
    void update();

    vsg::ref_ptr<vsg::MatrixTransform> getTransform() const
    {
        return m_objectsTransform;
    }
    vsg::ref_ptr<vsg::MatrixTransform> getScaleTransform() const
    {
        return m_scaleTransform;
    }
    vsg::ref_ptr<vsg::MatrixTransform> getHandTransform() const
    {
        return m_handTransform;
    }
    vsg::dvec3 getWorldPointOfInterest() const;
    void getHandWorldPosition(double *, double*, double*);
    void addPointerIcon(vsg::ref_ptr<vsg::Node> node);
    void removePointerIcon(const vsg::Node *node);

    void setWireframe(WireframeMode mode);
    void setPointerType(int pointerType);
    int getPointerType()
    {
        return (m_pointerType);
    }

    vsg::dbox getBoundingBox();

    bool highQuality() const;

    template <class T>
    T* findFirstNode(const std::string& name, bool startsWith = false, vsg::Node* rootNode = NULL)
    {
        if (!rootNode)
            rootNode = m_objectsRoot;

        if (!rootNode)
            return NULL;


        std::string nodeName;
        if (rootNode->getValue("name", nodeName))
        {
            if (!(startsWith ? strncmp(nodeName.c_str(), name.c_str(), name.length()) : strcmp(nodeName.c_str(), name.c_str())))
            {
                T* node = dynamic_cast<T*>(rootNode);
                if (node)
                    return node;
            }
        }

        vsg::Group *group = dynamic_cast<vsg::Group *>(rootNode);
        if (group)
        {

            for (const auto& child : group->children)
            {
                T *node = findFirstNode<T>(name, startsWith, child);
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
    void setScaleFactor(double scaleFactor, bool sync = true);
    void scaleAllObjects(bool resetView = false, bool simple = false);
    bool isScalingAllObjects() const
    {
        return m_scalingAllObjects;
    }
    void boundingBoxToMatrices(const vsg::dbox &boundingSphere,
                                  bool resetView, vsg::dmat4 &currentMatrix, double &scaleFactor) const;

    void adjustScale();

    void toggleAxis(bool state);
    void toggleHighQuality(bool state);
    void viewAll(bool resetView = false, bool simple = false);
    float floorHeight()
    {
        return m_floorHeight;
    }
    vsg::ref_ptr<vsg::MatrixTransform>  objectsRoot()
    {
        return m_objectsRoot;
    }


   // void setMultisampling(vsg::Multisample::Mode);

    void setScaleFromButton(double direction);
    void setScaleFactorButton(float f);
    void setScaleRestrictFactor(float min, float max);

    void setRestrictBox(float minX, float maxX, float minY, float maxY, float minZ, float maxZ);

    // disables "store scenegraph"
    void protectScenegraph();

    int m_vectorInteractor; //< don't use - for COVISE plugin only

    bool isHighQuality() const;

private:
    static vvSceneGraph *s_instance;
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

    vsg::ref_ptr<vsg::Group> m_scene, m_objectsScene;
    vsg::ref_ptr<vsg::MatrixTransform> m_handTransform;
    vsg::ref_ptr<vsg::Switch> m_handSwitch;
    vsg::ref_ptr<vsg::Switch> m_handIconSwitch;
    vsg::ref_ptr<vsg::MatrixTransform> m_handAxisTransform, m_viewerAxisTransform, m_smallSceneAxisTransform;
    vsg::ref_ptr<vsg::MatrixTransform> m_worldAxis, m_handAxis, m_viewerAxis, m_objectAxis, m_smallSceneAxis;
    vsg::ref_ptr<vsg::Group> m_menuGroupNode;
    vsg::ref_ptr<vsg::Group> m_alwaysVisibleGroupNode;
    vsg::ref_ptr<vsg::MatrixTransform> m_pointerDepthTransform;
    float m_pointerDepth;
    vsg::ref_ptr<vsg::Node> m_handLine;
    vsg::ref_ptr<vsg::Node> m_AxisGeometry;
    

    bool showSmallSceneAxis_;
    bool transparentPointer_;

    vsg::ref_ptr <vsg::MatrixTransform> m_objectsRoot;

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

    vsg::dmat4 m_invBaseMatrix;
    vsg::dmat4 m_oldInvBaseMatrix;

    int m_pointerType;

    // attribute SCALE attached to PerformerScene objects:
    // SCALE viewAll                        : scaleMode=1.0
    // SCALE keep                           : scaleMode=0.0
    // SCALE <pos number eg. 2.0 or 0.5>    : scaleMode=<number>
    // no SCALE attribute                   : scaleMode=-1.0
    double m_scaleMode, m_scaleFactor;
    float m_scaleFactorButton;
    float m_scaleRestrictFactorMin, m_scaleRestrictFactorMax;
    float m_transRestrictMinX, m_transRestrictMinY, m_transRestrictMinZ;
    float m_transRestrictMaxX, m_transRestrictMaxY, m_transRestrictMaxZ;
    bool m_scaleAllOn;
    bool m_scalingAllObjects;
    vsg::ref_ptr<vsg::MatrixTransform> m_scaleTransform;
    vsg::ref_ptr<vsg::MatrixTransform> m_handIconScaleTransform;
    vsg::ref_ptr<vsg::MatrixTransform> m_handAxisScaleTransform;
    float m_handIconSize;
    float m_handIconOffset;

    float wiiPos;

    vsg::ref_ptr<vsg::MatrixTransform> m_objectsTransform;
    typedef std::set<vsg::Node *> NodeSet;

    bool menusAreHidden;

    bool isScenegraphProtected_;

    bool m_enableHighQualityOption, m_switchToHighQuality, m_highQuality;
    vrui::coCombinedButtonInteraction* m_interactionHQ;

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
