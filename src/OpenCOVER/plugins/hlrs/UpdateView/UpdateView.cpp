/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "UpdateView.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <config/coConfig.h>
#include <config/CoviseConfig.h>
#include <osg/Matrix>
#include <osg/Vec3>
#include <OpenVRUI/osg/mathUtils.h>
#include "coVRMSController.h"

UpdateView::UpdateView()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "UpdateView::UpdateView\n");
    std::string hostname = covise::coCoviseConfig::getEntry("host", "COVER.Plugin.UpdateView.Server", "localhost");
    int port = covise::coCoviseConfig::getInt("port", "COVER.Plugin.UpdateView.Server", 7000);
    fprintf(stderr, "UpdateView::%s", hostname.c_str());
    IpEndpointName host(hostname.c_str(), port);
    socket = new UdpTransmitSocket(host);
    p = new osc::OutboundPacketStream(buffer, IP_MTU_SIZE);
    covise::coConfig *config = covise::coConfig::getInstance();
    QString myHost = config->getActiveHost();
    m_numViews = 14;

    screens = new screenStruct[m_numViews];
    for (int i = 0; i < m_numViews; i++)
    {
        char hName[1000];
        sprintf(hName, "fasi%02d", i + 1);

        config->setActiveHost(QString(hName));
        float h, p, r;
        int x, y, z;
        int hsize, vsize;
        bool state = coCoverConfig::getScreenConfigEntry(0, screens[i].name, &hsize, &vsize, &x, &y, &z, &h, &p, &r);
        if (!state)
        {
            cerr << "UpdateView::wrong ScreenConfig entry." << endl;
        }
        else
        {
            screens[i].hsize = (float)hsize;
            screens[i].vsize = (float)vsize;
            screens[i].xyz.set((float)x, (float)y, (float)z);
            screens[i].hpr.set(h, p, r);
        }
    }

    config->setActiveHost(myHost);
}

// this is called if the plugin is removed at runtime
UpdateView::~UpdateView()
{
    fprintf(stderr, "UpdateView::~UpdateView\n");
    delete socket;
    delete p;
}

void
UpdateView::preFrame()
{
    if (!coVRMSController::instance()->isMaster())
        return;
    if (oldPos != cover->getViewerMat().getTrans())
    {
        oldPos = cover->getViewerMat().getTrans();
        for (int i = 0; i < m_numViews; i++)
        {
            osg::Matrix rot;
            osg::Matrix invRot;
            osg::Vec3 viewerTransformed;
            osg::Vec3 normal(0, 1, 0);
            MAKE_EULER_MAT_VEC(rot, screens[i].hpr);
            invRot.invert(rot);
            viewerTransformed = osg::Matrix::transform3x3(oldPos, invRot);

            osg::Vec3 pos(screens[i].xyz);
            osg::Vec3 center = osg::Matrix::transform3x3(pos, invRot);
            float distance = center[1];

            float nd = distance - (normal * viewerTransformed);

            osg::Vec3 fusspunkt = viewerTransformed + (normal * nd);
            osg::Vec3 fpxy = fusspunkt;
            osg::Vec3 fpyz = fusspunkt;
            fpxy[2] = 0;
            fpyz[0] = 0;
            osg::Vec3 dist(0, distance, 0);
            osg::Vec3 tmp = fusspunkt - center + dist;
            osg::Vec3 rp = tmp + osg::Vec3(screens[i].hsize / 2.0, 0, 0);
            osg::Vec3 lp = tmp - osg::Vec3(screens[i].hsize / 2.0, 0, 0);
            osg::Vec3 tp = tmp + osg::Vec3(0, 0, screens[i].vsize / 2.0);
            osg::Vec3 bp = tmp - osg::Vec3(0, 0, screens[i].vsize / 2.0);
            rp[2] = 0;
            lp[2] = 0;
            tp[0] = 0;
            bp[0] = 0;

            float hr, hl, vt, vb;
            fpxy.normalize();
            fpyz.normalize();
            rp.normalize();
            lp.normalize();
            tp.normalize();
            bp.normalize();

            hr = -acos(normal * rp) / M_PI * 180;
            hl = acos(normal * lp) / M_PI * 180;
            vt = -acos(normal * tp) / M_PI * 180;
            vb = acos(normal * bp) / M_PI * 180;
            std::cerr << i << (int)0 << "pos " << oldPos[0] << ", " << oldPos[1] << ", " << oldPos[2] << " h " << screens[i].hpr[0] << " p " << screens[i].hpr[1] << " r " << screens[i].hpr[2] << " vt " << vt << " vb " << vb << " hl " << hl << " hr " << hr << std::endl;

            p->Clear();
            *p << osc::BeginMessage("/updateView") << i << (int)0 << oldPos[0] << oldPos[1] << oldPos[2] << screens[i].hpr[1] << screens[i].hpr[0] << screens[i].hpr[2] << vt << vb << hl << hr << osc::EndMessage;
            socket->Send(p->Data(), p->Size());

            fprintf(stderr, "UpdateView::%s\n", p->Data());

            p->Clear();
            *p << osc::BeginMessage("/updateOpenWarp") << osc::EndMessage;
            socket->Send(p->Data(), p->Size());
        }
    }
}

COVERPLUGIN(UpdateView)
