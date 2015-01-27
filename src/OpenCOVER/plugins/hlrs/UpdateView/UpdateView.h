/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _UPDATE_VIEW_PLUGIN_H
#define _UPDATE_VIEW_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: UpdateView Plugin (does nothing)                              **
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
#include <cover/coCoverConfig.h>
#include <cover/coVRConfig.h>

#include "osc/OscOutboundPacketStream.h"
#include "osg/Vec3"

#include "ip/UdpSocket.h"
#include "ip/IpEndpointName.h"
#define IP_MTU_SIZE 1536

using namespace opencover;

class UpdateView : public coVRPlugin
{
public:
    UpdateView();
    ~UpdateView();

    // this will be called in PreFrame
    void preFrame();

private:
    osc::OutboundPacketStream *p;
    char buffer[IP_MTU_SIZE];
    UdpTransmitSocket *socket;
    osg::Vec3 oldPos;
    int m_numViews;
    screenStruct *screens;
};
#endif
