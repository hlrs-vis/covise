/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Oddlot_PLUGIN_H
#define _Oddlot_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Oddlot Plugin (connection to the OpenDrive Road Editor)     **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Apr-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
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
#include "oddlotMessageTypes.h"

namespace vrui
{
class coCheckboxMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxGroup;
class coButtonMenuItem;
}

class OddlotPlugin;
class OddlotParameter;

using namespace vrui;
using namespace opencover;
using covise::Message;
using covise::ServerConnection;
class OddlotPlugin;

class DrawCallback : public osg::Camera::DrawCallback
{
public:
    DrawCallback(OddlotPlugin *plugin)
        : plugin(plugin)
    {
    }
    virtual void operator()(const osg::Camera &cam) const;

private:
    OddlotPlugin *plugin;
};

class OddlotPlugin : public coVRPlugin, public coMenuListener, public coTUIListener, public osgUtil::SceneView::ComputeStereoMatricesCallback
{
public:

    
    OddlotPlugin();
    ~OddlotPlugin();
    virtual bool init();
    static OddlotPlugin *instance()
    {
        return plugin;
    };

    // this will be called in PreFrame
    bool update();
    void preFrame();

    void destroyMenu();
    void createMenu();
    
    void createCamera();
    virtual void menuEvent(coMenuItem *aButton);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    void sendMessage(Message &m);
    
    void message(int toWhom, int type, int len, const void *buf);
    void sendImage();
protected:

	virtual osg::Matrixd computeLeftEyeProjection(const osg::Matrixd &projection) const;
	virtual osg::Matrixd computeLeftEyeView(const osg::Matrixd &view) const;
	virtual osg::Matrixd computeRightEyeProjection(const osg::Matrixd &projection) const;
	virtual osg::Matrixd computeRightEyeView(const osg::Matrixd &view) const;
    void setProjection(float xPos, float yPos, float width, float height);
    static OddlotPlugin *plugin;

    ServerConnection *serverConn;
    std::unique_ptr<ServerConnection> toOddlot;
    coTUITab *oddlotTab;
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
