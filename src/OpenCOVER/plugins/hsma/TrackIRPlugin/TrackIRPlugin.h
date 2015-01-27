/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:   TrackIRPlugin\TrackIRPlugin.h
//
// author:   Peter Gehrt p.gehrt@hs-mannheim.de
//         Hochschule Mannheim - Virtual Reality Center
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _TrackIR_H
#define _TrackIR_H
#ifndef __TABLETUSERINTERFACE_H__
#define __TABLETUSERINTERFACE_H__

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <util/UDP_Sender.h>
#include <osg/MatrixTransform>

using namespace opencover;
using namespace covise;

#define MAXPUFFER 1023
#define NOMINAL_ROLL 0.0
#define NOMINAL_PITCH 0.0
#define NOMINAL_YAW 0.0
#define PORT 7050

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   TrackIRPlugin. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
class TrackIRPlugin : public coVRPlugin, public coTUIListener
{

private:
    void initConnection();
    void initTUI();
    double smooth(float numbers[]);
    void update_tracking_values();
    void smooth_values();
    void update_matrix();
    int receiveTrackingData(const int);
    int extractTrackingData();

    covise::UDP_Sender *startStop;
    osg::ref_ptr<osg::MatrixTransform> mTF;
    osg::Matrixd mRz, mRy, mRx, mRzxy, mT;
    coTUITab *m_Tab;
    coTUILabel *m_Factor_Label;
    coTUISlider *m_Factor_Slider;
    coTUIButton *m_Reset_Button_Factor;
    coTUIButton *m_Calibrate_Button;
    WSADATA wsa;
    socklen_t address_lenght;
    socklen_t cli_size;
    fd_set fds;

    double diffRoll, diffPitch, diffYaw;
    double m_x, m_y, m_z, m_yaw, m_pitch, m_roll;
    float x[5], y[5], z[5], roll[5], pitch[5], yaw[5];
    int m_factor;
    char *a, *b, *c, *d, *e, *f, *limiter;
    int mySocket, client;
    struct sockaddr_in address;
    struct hostent *host;
    struct sockaddr_in cli;
    struct timeval timeout;
    char buffer[MAXPUFFER];
    bool firstStart, lastTimeTimedOut;
    unsigned long opt;
    int ret, counter, receiveCounter, errorCounter, timedOutCounter, skipCounter;

public:
    virtual unsigned int button(int station);
    static TrackIRPlugin *plugin;
    TrackIRPlugin();
    ~TrackIRPlugin();
    void preFrame();

    osg::Matrixd matrix;
    void getMatrix(int station, osg::Matrix &mat);
    void tabletEvent(coTUIElement *tUIItem);
};
#endif __TABLETUSERINTERFACE_H__
#endif