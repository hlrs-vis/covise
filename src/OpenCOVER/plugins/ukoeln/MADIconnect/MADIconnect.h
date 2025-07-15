/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MADI_CONNECT_PLUGIN_H
#define MADI_CONNECT_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: MADiConnect OpenCOVER Plugin (connects to MADI       )      **
 **                                                                          **
 **                                                                          **
 ** Author: D. Wickeroth                                                     **
 **                                                                          **
 ** History:                                                                 **
 ** July 2025  v1  				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/Geode>

#include <net/covise_connect.h>

using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class MADIconnect : public opencover::coVRPlugin
{
public:
    MADIconnect();
    ~MADIconnect();
    virtual bool destroy();

private:
    osg::ref_ptr<osg::Geode> basicShapesGeode;

protected:
    ServerConnection *serverConn = nullptr;
};
#endif
