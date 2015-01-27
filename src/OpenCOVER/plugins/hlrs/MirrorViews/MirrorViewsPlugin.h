/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MIRRORVIEWS_PLUGIN_H
#define _MIRRORVIEWS_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: MirrorViews Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/coCoverConfig.h>
#include <cover/RenderObject.h>
#include <cover/VRViewer.h>
#include <cover/coTabletUI.h>
#ifdef WIN32
#define __x86_64__
#endif
#include "osc/OscOutboundPacketStream.h"

#include "ip/UdpSocket.h"
#define IP_MTU_SIZE 1536
using namespace opencover;

struct mirrorInfo
{
    osg::Vec3 pos; // center position of the Mirror
    osg::Vec3 ori; // orientation of the Mirror
    bool enabled;
};

class MirrorViewsPlugin : public coVRPlugin, public coTUIListener
{
public:
    MirrorViewsPlugin();
    ~MirrorViewsPlugin();

    // this will be called in PreFrame
    void preFrame();

private:
    int numSlaves;
    int numScreens;
    std::vector<screenStruct *> screens;
    mirrorInfo *mirrors;
    void computeFrustum(osg::Vec3 viewerPos, float &h, float &p, float &r, float &top, float &left, float &right, float &bottom, screenStruct *screen);

    coTUITab *tuiMirrorTab;
    coTUIEditFloatField *tuiPosX;
    coTUIEditFloatField *tuiPosY;
    coTUIEditFloatField *tuiPosZ;
    coTUIEditFloatField *tuiOriH;
    coTUIEditFloatField *tuiOriP;
    coTUIEditFloatField *tuiOriR;

    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);

    mirrorInfo mi;
    osg::Vec3 oldViewer;
    osc::OutboundPacketStream *packet;
    UdpTransmitSocket *udpSocket;
    char buffer[IP_MTU_SIZE];
};
#endif
