/* This file is part of COVISE.
   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.
 * License: LGPL 2+ */
#ifndef _MapLink_PLUGIN_H
#define _MapLink_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <net/covise_connect.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Group>
#include <stack>
#include <map>
#include <cover/coTabletUI.h>
#include <cover/coVRSceneView.h>
#include <OpenVRUI/sginterface/vruiActionUserData.h>
#include "MapLinkMessageTypes.h"
#include <array>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ref_ptr>
#include <unordered_map>
#include <unordered_set>
#include <osg/observer_ptr>
#include <osg/Matrix>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coInteractionManager.h>
#include <atomic> // Snapshot-Kamera wird aus Render-Callback abgeschlossen
#include <sigslot/signal.hpp>

std::map<int, std::array<osg::Vec3d, 4>> m_modules;

namespace vrui
{
class coCheckboxMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxGroup;
class coButtonMenuItem;
}

class MapLinkPlugin;
class MapLinkParameter;

using namespace vrui;
using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class DrawCallback : public osg::Camera::DrawCallback
{
public:
    DrawCallback(MapLinkPlugin *plugin)
        : plugin(plugin)
    {
    }
    virtual void operator()(const osg::Camera &cam) const;

private:
    MapLinkPlugin *plugin;
};

class MapLinkPlugin : public coVRPlugin, public coMenuListener, public coTUIListener, public osgUtil::SceneView::ComputeStereoMatricesCallback
{
public:
    MapLinkPlugin();
    ~MapLinkPlugin() override;
    bool init() override;
    static MapLinkPlugin *instance() { return plugin; }
    bool update() override;
    void preFrame() override;
    void key(int type, int keySym, int mod) override;
    void destroyMenu();
    void createMenu();
    void createCamera();
    void menuEvent(coMenuItem *aButton) override;
    void tabletEvent(coTUIElement *tUIItem) override;
    void tabletPressEvent(coTUIElement *tUIItem) override;
    void sendMessage(Message &m);
    void message(int toWhom, int type, int len, const void *buf) override;
    void sendImage();

protected:
    void createModule(int moduleId, const std::array<osg::Vec3d, 4> &corners, osg::Node *pvModel);
    void deleteModule(int moduleId);
    void sendDeleteModuleRequest(int moduleId);
    void clearAllModules();
    void updateModuleDeleteHighlight();
    void setModuleDeleteHighlight(osg::MatrixTransform *module, bool enabled);
    void setModuleDeleteOutline(int moduleId, bool enabled);
    osg::Matrixd computeLeftEyeProjection(const osg::Matrixd &projection) const override;
    osg::Matrixd computeLeftEyeView(const osg::Matrixd &view) const override;
    osg::Matrixd computeRightEyeProjection(const osg::Matrixd &projection) const override;
    osg::Matrixd computeRightEyeView(const osg::Matrixd &view) const override;
    void setProjection(float centerX, float centerY, float width, float height, const osg::Vec3d &up);
    static MapLinkPlugin *plugin;
    void sendProjectTransformReady();

    bool m_projectTransformReady = false;
    bool m_transformReadySent = false;

    sigslot::connection m_projectionConnection;

    ServerConnection *serverConn;
    std::unique_ptr<ServerConnection> toMapLink;
    coTUITab *MapLinkTab = nullptr;
    osg::ref_ptr<osg::Group> m_pvModuleGroup;
    std::unordered_map<int, osg::ref_ptr<osg::MatrixTransform>> m_moduleNodes;
    std::unordered_map<int, osg::ref_ptr<osg::Geode>>m_moduleDeleteOutlines;
    osg::ref_ptr<osg::Node> m_pvModel; // Einmalig geladenes PV-Modell

    // Originalabmessungen des OBJ-Modells
    double m_pvModelWidth = 0.0;
    double m_pvModelLength = 0.0;

    // Geometrischer Mittelpunkt in der lokalen XY-Ebene
    double m_pvModelCenterX = 0.0;
    double m_pvModelCenterY = 0.0;

    int m_selectedModuleId = -1;

    std::unordered_set<int> m_pendingModuleDeletes; // Verhindert mehrfache Loeschanforderungen fuer dasselbe Modul.

    coCheckboxMenuItem *m_selectModulesButton = nullptr;
    coTrackerButtonInteraction *m_selectInteraction = nullptr;
    osg::observer_ptr<osg::Group> m_cityModelParent;
    osg::Matrixd m_worldToCityParent;
    osg::ref_ptr<osg::Camera> camera;
    osg::ref_ptr<osg::Image> image;
    osg::ref_ptr<DrawCallback> drawCallback;
    int resX, resY;

    // Nur waehrend einer MSG_GetMap-Anfrage registrieren; danach abmelden.
    bool m_snapshotCameraRegistered = false;
    std::atomic<bool> m_snapshotRenderCompleted { false };

    float x, y, width, height;
    int xRes, yRes;
    void handleMessage(Message *m);
    Message *msg;
    osg::Matrix projMat;
    osg::Matrix viewMat;
};
#endif
