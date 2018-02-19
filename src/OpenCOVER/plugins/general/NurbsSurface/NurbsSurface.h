/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NurbsSurface_PLUGIN_H
#define _NurbsSurface_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: NurbsSurface OpenCOVER Plugin (draws a NurbsSurface)        **
 **                                                                          **
 **                                                                          **
 ** Author: F.Karle/ K.Ahmann	                                             **
 **                                                                          **
 ** History:  			  	                                     **
 ** December 2017  v1		                                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/Geode>
#include <cover/coVRCommunication.h>
#include <net/message.h>


class NurbsSurface : public opencover::coVRPlugin
{
public:
    NurbsSurface();
    ~NurbsSurface();
    virtual bool destroy();
    void message(int toWhom, int type, int len, const void *buf); ///< handle incoming messages

private:
    osg::ref_ptr<osg::Geode> geode;
};
#endif

