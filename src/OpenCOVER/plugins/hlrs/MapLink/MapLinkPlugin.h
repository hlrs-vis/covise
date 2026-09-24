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
#include <osg/ref_ptr>
#include <unordered_map>
#include <osg/observer_ptr>
#include <osg/Matrix>

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
class MapLinkPlugin;

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

// connection to the OpenDrive Road Editor
class MapLinkPlugin : public coVRPlugin, public coMenuListener, public coTUIListener, public osgUtil::SceneView::ComputeStereoMatricesCallback
{
public:

    
    MapLinkPlugin();
    ~MapLinkPlugin() override;
    bool init() override;
    static MapLinkPlugin *instance()
    {
        return plugin;
    };

    // this will be called in PreFrame
    bool update() override;
    void preFrame() override;

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

    //void showLocationMarker(double x, double y, const osg::Vec4 &color);
    void createModule(int moduleId, const std::array<osg::Vec3d, 4> &corners, osg::Node *pvModel);
    void deleteModule(int moduleId);
    void clearAllModules();
	osg::Matrixd computeLeftEyeProjection(const osg::Matrixd &projection) const override;
	osg::Matrixd computeLeftEyeView(const osg::Matrixd &view) const override;
	osg::Matrixd computeRightEyeProjection(const osg::Matrixd &projection) const override;
	osg::Matrixd computeRightEyeView(const osg::Matrixd &view) const override;
    void setProjection(float xPos, float yPos, float width, float height);
    static MapLinkPlugin *plugin;

    ServerConnection *serverConn;
    std::unique_ptr<ServerConnection> toMapLink;
    coTUITab *MapLinkTab = nullptr;
    osg::ref_ptr<osg::Group> m_pvModuleGroup; // Gemeinsame Gruppe für alle PV-Module
    std::unordered_map<int, osg::ref_ptr<osg::MatrixTransform>> m_moduleNodes; // Zuordnung zwischen Modul-ID und sichtbarem 3D-Modell
    osg::observer_ptr<osg::Group> m_cityModelParent;
    osg::Matrixd m_worldToCityParent;
    osg::ref_ptr<osg::Camera> camera;
    osg::ref_ptr<osg::Image> image;
    osg::ref_ptr<DrawCallback> drawCallback;
    int resX,resY;
    
    float x,y,width,height;
    int xRes,yRes;
    void handleMessage(Message *m);
    Message *msg;
	osg::Matrix projMat;
	osg::Matrix viewMat;
};
#endif
