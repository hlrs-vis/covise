/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:   TrackIRPlugin\TrackIRPlugin.cpp
//
// summary:    See attached readme
//
// author:   Peter Gehrt p.gehrt@hs-mannheim.de
//          Virtual Reality Center Hochschule Mannheim - University of Applied Sciences
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "TrackIRPlugin.h"
TrackIRPlugin *TrackIRPlugin::plugin = NULL;
const double TORAD = (M_PI / 180.0);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Default constructor. Initialize some variables </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
TrackIRPlugin::TrackIRPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "TrackIRPlugin started\n");
    m_x = 0.0, m_y = 0.0, m_z = 0.0;
    m_roll = 0.0, m_pitch = 0.0, m_yaw = 0.0;
    for (int i = 0; i < 5; i++)
    {
        x[i] = 0.0;
        y[i] = 0.0;
        z[i] = 0.0;
        roll[i] = 0.0;
        pitch[i] = 0.0;
        yaw[i] = 0.0;
    }
    diffPitch = 0.0, diffYaw = 0.0, diffRoll = 0.0;
    receiveCounter = 0, errorCounter = 0, skipCounter = 0, counter = 0, timedOutCounter = 0;
    firstStart = true;
    lastTimeTimedOut = false;
    opt = 1;
    m_factor = 1;
    // dummy matrix
    mRz.makeRotate(1.0, osg::Vec3d(0.0, -1.0, 0.0));
    mRy.makeRotate(-1.0, osg::Vec3d(0.0, 0.0, 1.0));
    mRx.makeRotate(-2.0, osg::Vec3d(-1.0, 0.0, 0.0));
    mT.setTrans(osg::Vec3d(-200.0, -1000.0, -100.0));
    matrix = (mRy * mRx * mRz) * mT;
    // initialize socket
    WSAStartup(MAKEWORD(1, 1), &wsa);
    initConnection();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Initialises tablet UI. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::initTUI()
{
    m_Tab = new coTUITab("TrackIR5 Plugin", coVRTui::instance()->mainFolder->getID());
    m_Tab->setPos(0, 0);

    m_Calibrate_Button = new coTUIButton("calibrate", m_Tab->getID());
    m_Calibrate_Button->setEventListener(this);
    m_Calibrate_Button->setPos(1, 1);

    m_Factor_Slider = new coTUISlider("factor", m_Tab->getID());
    m_Factor_Slider->setEventListener(this);
    m_Factor_Slider->setPos(1, 2);
    m_Factor_Slider->setRange(0, 5);
    m_Factor_Slider->setValue(1);

    m_Factor_Label = new coTUILabel("yaw/pitch scale factor:", m_Tab->getID());
    m_Factor_Label->setPos(0, 2);

    m_Reset_Button_Factor = new coTUIButton("reset", m_Tab->getID());
    m_Reset_Button_Factor->setEventListener(this);
    m_Reset_Button_Factor->setPos(3, 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Initialises network connection to TrackIRServer. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::initConnection()
{

    mySocket = 0;
    mySocket = socket(PF_INET, SOCK_STREAM, 0);

    // trackingserver have to be running before starting this plugin
    fprintf(stderr, "TrackIRPlugin searching TrackingData Server (TrackIRServer)...\n");
    host = gethostbyname("localhost");
    if (!host)
        fprintf(stderr, "gethostbyname() failed");
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT); //free port > 1024
    address.sin_family = AF_INET; // UDP connection
    socklen_t address_lenght;
    address_lenght = sizeof(address);
    if (mySocket == -1)
        fprintf(stderr, "socket() failed\n");
    if (bind(mySocket, (struct sockaddr *)&address, address_lenght) == -1)
        fprintf(stderr, "bind() failed\n");
    if (listen(mySocket, 3) == -1)
        fprintf(stderr, "listen() failed\n");
    cli_size = sizeof(cli);
    client = 0;
    client = accept(mySocket, (struct sockaddr *)&cli, &cli_size);
    // activate non-blocking socket for select()
    ioctlsocket(mySocket, FIONBIO, &opt);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Unnecessary button method, not used in this plugin</summary>
///
/// <param name="station">   The requested station. </param>
///
/// <returns>   always 0 </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned int TrackIRPlugin::button(int station)
{
    return 0; // no buttons on this device
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Gets the matrix </summary>
///
/// <param name="station">   The requested station. </param>
/// <param name="mat">      [in,out] the matrix. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::getMatrix(int station, osg::Matrix &mat)
{
    // skip frames in case of no incoming tracking data
    if (lastTimeTimedOut)
    {
        int max = 0; // number of skipped frames to gain some performance
        switch (timedOutCounter)
        {
        case 0:
            max = 1;
            break;
        case 1:
            max = 10;
            break;
        case 2:
            max = 100;
            break;
        case 3:
            max = 200;
            break;
        //case 4: max = 750; break;
        //case 5: max = 1000; break;
        default:
            max = 500;
        }
        if (skipCounter > max)
        { // skipping
            receiveTrackingData(client);
            skipCounter = 0;
        }
        else
            skipCounter++;
    }
    else
        receiveTrackingData(client);
    //fprintf(stderr,"TrackIR Data: %i - Errors: %i \n",receiveCounter,errorCounter); //debug print
    mat = matrix;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Receive tracking data from TrackIRServer. </summary>
///
/// <param name="sock">   The network socket </param>
///
/// <returns>   Error code 0/1 </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
int TrackIRPlugin::receiveTrackingData(const int sock)
{
    FD_ZERO(&fds);
    FD_SET(sock, &fds);
    timeout.tv_sec = 1; // timeout in seconds
    timeout.tv_usec = 0;
    ret = select(sock + 1, &fds, NULL, NULL, &timeout);
    if (ret <= 0)
    {
        errorCounter++;
        lastTimeTimedOut = true; // set flag
        timedOutCounter++;
        return 0;
    }
    // continue receiving data
    lastTimeTimedOut = false; //reset flag
    timedOutCounter = 0;
    int bytes = 0;
    bytes = recv(sock, buffer, MAXPUFFER, 0);
    if (bytes > 0)
    { // somethine was received..
        int ret = extractTrackingData();
        smooth_values();
        update_tracking_values();
        update_matrix();
        return ret;
    }
    else
    { // if receive returns 0 the connection was closed
        fprintf(stderr, "Connection closed..trying reconnect\n"); // debug print
        closesocket(mySocket);
        initConnection();
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Extracts received trackingdata </summary>
///
/// <returns>   errorcode 0/1 </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
int TrackIRPlugin::extractTrackingData()
{
    char *a, *b, *c, *d, *e, *f, *limiter;
    char *expected_limiter = "$$$";
    limiter = "a"; // initialization
    //incoming order: x;y;z;yaw;pitch;roll;$$$
    a = strtok(buffer, ";");
    b = strtok(NULL, " ;");
    c = strtok(NULL, " ;");
    d = strtok(NULL, " ;");
    e = strtok(NULL, " ;");
    f = strtok(NULL, " ;");
    limiter = strtok(NULL, ";");
    // validate received values
    if ((limiter != NULL) && (limiter[0] == expected_limiter[0]))
    {
        x[0] = atof(a);
        y[0] = atof(b);
        z[0] = atof(c);
        yaw[0] = atof(d);
        pitch[0] = atof(e);
        roll[0] = atof(f);

        //fprintf ( stderr,"x:%.0f y:%.0f z:%.0f roll:%.0f yaw:%.0f pitch:%.0f\n",m_x,m_y,m_z,m_roll,m_yaw,m_pitch); // debug print
        //fprintf ( stderr,"x:%.0f x2:%.0f x3:%.0f x4:%.0f x5:%.0f\n",x,x2,x3,x4,x5); // debug print
        //receiveCounter++;

        return 1;
    }

    else
    { // invalid data
        //errorCounter++;
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Update matrix </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::update_matrix()
{
    // rotation matrix: y(yaw), z(roll), x(pitch)
    mRz.makeRotate((m_roll + diffRoll) * TORAD, osg::Vec3d(0.0, -1.0, 0.0));
    mRy.makeRotate((m_yaw + diffYaw) * TORAD * m_factor, osg::Vec3d(0.0, 0.0, -1.0));
    mRx.makeRotate((m_pitch + diffPitch) * TORAD * m_factor, osg::Vec3d(1.0, 0.0, 0.0));

    // camera to covise: x=-x,y=-z,z=y
    mT.setTrans(osg::Vec3d(-m_x, -m_z, m_y));
    mRzxy = mRy * mRx * mRz;
    matrix = mRzxy * mT;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Smooths given trackingvalue </summary>
///
/// <param name="numbers[]"> Array of numbers</param>
///
/// <returns>   smoothed number </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
double TrackIRPlugin::smooth(float numbers[])
{
    return (floor(numbers[0] + numbers[1] + numbers[2] + numbers[3] + numbers[4]) / 5);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Smooth tracking values. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::smooth_values()
{
    m_yaw = smooth(yaw);
    m_pitch = smooth(pitch);
    m_roll = smooth(roll);
    m_x = smooth(x);
    m_y = smooth(y);
    m_z = smooth(z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Updates all tracking values. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::update_tracking_values()
{
    for (int i = 4; i > 0; i--)
    {
        x[i] = x[i - 1];
        y[i] = y[i - 1];
        z[i] = z[i - 1];
        roll[i] = roll[i - 1];
        pitch[i] = pitch[i - 1];
        yaw[i] = yaw[i - 1];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Pre frame, only used at first startup. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::preFrame()
{
    // initialize TabletUI at first start
    if (firstStart)
    {
        initTUI();
        firstStart = false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   TabletUI event manager. </summary>
///
/// <param name="tUIItem">   [in,out] If not null, the user interface item. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
void TrackIRPlugin::tabletEvent(coTUIElement *tUIItem)
{

    if (tUIItem == m_Factor_Slider)
    {
        m_factor = (m_Factor_Slider->getValue());
    }

    if (tUIItem == m_Reset_Button_Factor)
    {
        m_factor = 1.0;
        m_Factor_Slider->setValue(1);
    }

    if (tUIItem == m_Calibrate_Button)
    {
        diffRoll = (NOMINAL_ROLL - roll[0]);
        diffPitch = (NOMINAL_PITCH - pitch[0]);
        diffYaw = (NOMINAL_YAW - yaw[0]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Destructor. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
TrackIRPlugin::~TrackIRPlugin()
{
    closesocket(client);
    closesocket(mySocket);
    delete m_Tab;
    delete m_Factor_Label;
    delete m_Factor_Slider;
    delete m_Reset_Button_Factor;
    delete m_Calibrate_Button;
    fprintf(stderr, "TrackIRPlugin stopped\n");
}

COVERPLUGIN(TrackIRPlugin)
