/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rift_PLUGIN_H
#define _Rift_PLUGIN_H
/****************************************************************************\
 **                                                           (C)2014 HLRS **
 **                                                                        **
 ** Description: Rift Plugin                                               **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Wössner                                                    **
 **                                                                        **
 ** History:                                                               **
 ** 2014-Dec-23  v1	     		                                           **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>

#include <cover/coVRTui.h>

#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>

#include "OVR_CAPI_GL.h"
#include "oculusdevice.h"

using namespace covise;
using namespace opencover;

class PLUGINEXPORT RiftPlugin : public coVRPlugin,
                                public coTUIListener
{

public:
    static RiftPlugin *plugin;

    RiftPlugin();
    ~RiftPlugin();

    bool init();

    void tabletReleaseEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);
    // this will be called in PreFrame
    virtual void preDraw(osg::RenderInfo &);
    void preFrame();
    void postFrame();
    void preSwapBuffers(int windowNumber);
    virtual void getMatrix(int station, osg::Matrix &mat);

private:
    coTUITab *tab;
    coTUILabel *res;
    coTUIEditField *residue;
    coTUIButton *showSticks;
    coTUIButton *hideSticks;
    osg::ref_ptr<OculusDevice> hmd;
    int m_frameIndex;
};

#endif
