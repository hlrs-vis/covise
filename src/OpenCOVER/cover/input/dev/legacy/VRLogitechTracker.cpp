/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRLogitechTrackerg.h (Performer 2.0)	*
 *									*
 *	Description		logitech tracker class			*
 *				based on logidrvr from NASA AMES	*
 *				this tracker class measures in cm ! 	*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			27. October 96				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include "VRLogitechTracker.h"
#include <util/common.h>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef __sgi
#ifndef _WIN32
#include <sys/ioctl.h>
#include <asm/ioctls.h>
#else
#include <io.h>
#endif
#ifndef fsin
#define fsin sin
#define fcos cos
#define facos acos
#endif
#endif

void
print_bin(char a)
{
    int i;

    for (i = 7; i >= 0; i--)
        printf("%c", (a & (1 << i)) ? '1' : '0');
}

/************************************************************************/
/*									*/
/* 	VRLogitechTracker	base class				*/
/*									*/
/************************************************************************/

VRLogitechTracker::VRLogitechTracker()
{
#ifdef DBGPRINT
    printf("\n... new VRLogitechTracker\n");
#endif
    systemOffset.set(0.0f, -18.0f * 2.54, 12.0f * 2.54);
}

VRLogitechTracker::~VRLogitechTracker()
{

    closeSerialPort();
}

int
VRLogitechTracker::init(char *portname, osg::Vec3 &offset)
{
#ifdef DBGPRINT
    printf("... VRLogitechTracker::init\n");
#endif
    strcpy(this->portname, portname);

    printf("...... portname: %s\n", this->portname);
    screenToTransmitterOffset = offset;

    if (openSerialPort() < 0)
        return (-1);
    else
    {
        resetControlUnit();
        setDemandReportingMode();
        setFilter();
        getCurrentOperatingInformation();
        if (isSlave)
        {
            if (receiverType == logitech_CRYSTAL_EYES)
                setSlaveTransmitterType(logitech_MOUSE);
            else if (receiverType == logitech_MOUSE)
                setSlaveTransmitterType(logitech_HEAD_TRACKER);
        }
        return (0);
    }
}

int
VRLogitechTracker::openSerialPort()
{

    struct termios t; /* termio struct */

    /* open a serial port, read/write */
    if ((fd = open(portname, O_RDWR | O_NDELAY)) < 0)
    {
        perror(portname);
        return (-1);
    }

    /* disable all input mode processing */
    t.c_iflag = 0;

    /* disable all output mode processing */
    t.c_oflag = 0;

    /* hardware control flags: 19200 baud, 8 data bits, 1 stop bits,
      no parity, enable receiver */
    t.c_cflag = B19200 | CS8 | CSTOPB | CREAD;

    /* disable local control processing (canonical, control sigs, etc) */
    t.c_lflag = 0;

    /* set control characters for non-canonical reads: VMIN = 1, VTIME = 0
      i.e., read not satisfied until at least 1 char is read, see termio(7) */
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
        return (-1);
    }

    /* do diagnostics, results are in "data" */
    getDiagnostics();

    /* check diagnostic return */
    if ((diagnosticsData[0] != 0xbf) || (diagnosticsData[1] != 0x3f))
    {
        fprintf(stderr, "Diagnostics failed\n");
        //return (-1);
        return (0);
    }

    return (0);
}

int
VRLogitechTracker::closeSerialPort()
{
    if (close(fd) < 0)
    {
        perror("error closing serial port");
        return (-1);
    }
    else
        return (0);
}

void
VRLogitechTracker::setSlaveTransmitterType(int type)
{
    switch (type)
    {
    case logitech_HEAD_TRACKER:
    {
#ifdef DBGPRINT
        printf("Setting slave transmitter type to HEAD_TRACKER\n");
#endif
        char buf[] = { 0x2A, 0x24, 0x02, 0x01, 0x13 };
        if (write(fd, buf, 5) != 5)
        {
            cerr << "VRLogitechTracker::setSlaveTransmitterType: short write" << endl;
        }
        break;
    }
    case logitech_MOUSE:
    {
#ifdef DBGPRINT
        printf("Setting slave transmitter type to MOUSE\n");
#endif
        char buf[] = { 0x2A, 0x24, 0x02, 0x01, 0x14 };
        if (write(fd, buf, 5) != 5)
        {
            cerr << "VRLogitechTracker::setSlaveTransmitterType: short write2" << endl;
        }
        break;
    }
    }
}

void
VRLogitechTracker::setDemandReportingMode()
{
    struct termios t;
#ifdef DEBUG
    printf("demand reporting enabled\n");
#endif

    tcgetattr(fd, &t);

    /* set control characters for non-canonical reads: VMIN, VTIME
      i.e., read a complete euler record packet */
    t.c_cc[VMIN] = EULER_RECORD_SIZE;
    t.c_cc[VTIME] = 1;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
    }

    if (write(fd, "*D", 2) != 2)
    {
        cerr << "VRLogitechTracker::setDemandReportingMode: short write" << endl;
    }
}

void
VRLogitechTracker::resetControlUnit()
{

#ifdef DEBUG
    printf("resetting control unit\n");
#endif

    if (write(fd, "*R", 2) != 2)
    {
        cerr << "VRLogitechTracker::resetControlUnit: short write" << endl;
    }

    sleep(1);
}

void
VRLogitechTracker::setFilter()
{
    char buf[] = { 0x2A, 0x24, 0x02, 0x07, 0x08 };
    if (write(fd, buf, 5) != 5)
    {
        cerr << "VRLogitechTracker::setFilter: short write" << endl;
    }
}

void
VRLogitechTracker::getDiagnostics()
{

    requestDiagnostics(); /* command diagnostics */
    sleep(1); /* wait 1/10 second */
    if (read(fd, diagnosticsData, DIAGNOSTIC_SIZE) != DIAGNOSTIC_SIZE)
    {
        cerr << "VRLogitechTracker::getDiagnostics: short read" << endl;
    }

#ifdef DBGPRINT
    printf("...... Diagnostics Data: \n");
    print_bin(diagnosticsData[0]);
    printf("\n");
    print_bin(diagnosticsData[1]);
    printf("\n");
#endif
}

void
VRLogitechTracker::requestDiagnostics()
{
    struct termios t;

#ifdef DEBUG
    printf("performing diagnostics\n");
#endif

    tcgetattr(fd, &t);

    /* set control characters for non-canonical reads: VMIN, VTIME
      i.e., read a complete diagnostics packet */
    t.c_cc[VMIN] = DIAGNOSTIC_SIZE;
    t.c_cc[VTIME] = 1;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
    }

    if (write(fd, "*\05", 2) != 2)
    {
        cerr << "VRLogitechTracker::requestDiagnostics: short write" << endl;
    }
}

void
VRLogitechTracker::getRecord()
{
    int num_read;
    char record[EULER_RECORD_SIZE];

    requestReport();
    num_read = read(fd, record, EULER_RECORD_SIZE);

    /* if didn't get a complete record or if invalid record, then try
      to get a good one */
    while ((num_read < EULER_RECORD_SIZE) || !(record[0] & logitech_FLAGBIT))
    {

        /* flush the buffer */
        ioctl(fd, TCFLSH, 0);

        requestReport();
        num_read = read(fd, record, EULER_RECORD_SIZE);
    }

#ifdef DEBUG
    printf("%d bytes read...", num_read);
#endif

    /* convert the raw euler record to absolute record */
    eulerToAbsolute(record);
}

void
VRLogitechTracker::eulerToAbsolute(char record[EULER_RECORD_SIZE])
{
    long ax, ay, az, arx, ary, arz;

    buttons = (((char)record[0]) & (logitech_SUSPENDBUTTON | logitech_LEFTBUTTON | logitech_MIDDLEBUTTON | logitech_RIGHTBUTTON));

    ax = (record[1] & 0x40) ? 0xFFE00000 : 0;
    ax |= (long)(record[1] & 0x7f) << 14;
    ax |= (long)(record[2] & 0x7f) << 7;
    ax |= (record[3] & 0x7f);

    ay = (record[4] & 0x40) ? 0xFFE00000 : 0;
    ay |= (long)(record[4] & 0x7f) << 14;
    ay |= (long)(record[5] & 0x7f) << 7;
    ay |= (record[6] & 0x7f);

    az = (record[7] & 0x40) ? 0xFFE00000 : 0;
    az |= (long)(record[7] & 0x7f) << 14;
    az |= (long)(record[8] & 0x7f) << 7;
    az |= (record[9] & 0x7f);

    x = ((float)ax) / 1000.0;
    y = ((float)ay) / 1000.0;
    z = ((float)az) / 1000.0;

    arx = (record[10] & 0x7f) << 7;
    arx += (record[11] & 0x7f);

    ary = (record[12] & 0x7f) << 7;
    ary += (record[13] & 0x7f);

    arz = (record[14] & 0x7f) << 7;
    arz += (record[15] & 0x7f);

    pitch = ((float)arx) / 40.0;
    yaw = ((float)ary) / 40.0;
    roll = ((float)arz) / 40.0;

#ifdef DEBUG
    printf("raw: %ld %ld %ld %ld %ld %ld\n", ax, ay, az, arx, ary, arz);
    printf("%7.2f, %7.2f, %7.2f, %7.2f, %7.2f, %7.2f\n",
           x, y, z, pitch, yaw, roll);
#endif
}

void
VRLogitechTracker::requestReport()
{
#ifdef DEBUG
    printf("asking for a single report\n");
#endif

    if (write(fd, "*d", 2) != 2)
    {
        cerr << "VRLogitechTracker::requestReport: short write" << endl;
    }
}

void
VRLogitechTracker::getCurrentOperatingInformation()
{
    struct termios t;
    char informationData[300];

    tcgetattr(fd, &t);

    /* set control characters for non-canonical reads: VMIN, VTIME
      i.e., read a complete info packet */
    t.c_cc[VMIN] = 30;
    t.c_cc[VTIME] = 1;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
    }

    if (write(fd, "*m", 2) != 2)
    {
        cerr << "VRLogitechTracker::getCurrentOperatingInformation: short write" << endl;
    }
    sleep(1);

    if (read(fd, informationData, 30) != 30)
    {
        cerr << "VRLogitechTracker::getCurrentOperatingInformation: short read" << endl;
    }

    printf("...... Information data:\n");
    printf("...... Master Slave Status %d: ", informationData[6]);
    print_bin(informationData[6]);
    printf("\n");
    printf("...... Receiver type: %d ", informationData[8]);
    print_bin(informationData[8]);
    printf("\n");
    printf("...... Transmitter type: %d ", informationData[9]);
    print_bin(informationData[9]);
    printf("\n");

    if (informationData[6] == 1)
        isSlave = true;
    else
        isSlave = false;

    switch (informationData[8])
    {
    case 14:
        receiverType = logitech_MOUSE;
        break;

    case 13:
        receiverType = logitech_HEAD_TRACKER;
        break;

    case 11:
        receiverType = logitech_CRYSTAL_EYES;
    }
}

/************************************************************************/
/*									*/
/* 	VRLogitechSpacemouse	derived class				*/
/*									*/
/************************************************************************/

VRLogitechSpacemouse::VRLogitechSpacemouse()
{

    //mouseMode();
    //getCurrentOperatingInformation();
}

void
VRLogitechSpacemouse::getButtons(char *mouseButtons)
{

    *mouseButtons = buttons;
}

void
VRLogitechSpacemouse::getMatrix(osg::Matrix &mat)
{

    getRecord();

    // change the cood system to Performer
    float h;
    h = y;
    y = -z;
    z = h;
    roll = -roll;

    osg::Vec3 angles, pos;
    angles.set(yaw / 180.0 * M_PI, pitch / 180.0 * M_PI, roll / 180.0 * M_PI);
    pos.set(x, y, z);

    // scale from inch to cm: 2.54
    pos *= 2.54f;

    // origin is the center of the screen
    // offset screen center - logitech transmitter origin is osg::Vec3 screenToTransmitterOffset
    // offset between logitech transmitter foot and logitech origin is systemOffset

    pos = pos + systemOffset;
    pos = pos + screenToTransmitterOffset;

    osg::Matrix rx, ry, rz;

    rz.makeRotate(angles[0], 0.0f, 0.0f, 1.0f);
    rx.makeRotate(angles[1], 1.0f, 0.0f, 1.0f);
    ry.makeRotate(angles[2], 0.0f, 1.0f, 1.0f);

    mat.mult(ry, rx);
    mat.postMult(rz);
    //mat.setRow(3, pos);
    for (int i = 0; i < 3; i++)
    {
        mat.ptr()[3 * 4 + i] = pos[i];
    }

#ifdef DBGPRINT1
    printf("Maus Pos: %f %f %f [cm]\n\n", pos[0], pos[1], pos[2]);
    printf("Maus pitch roll yaw: %f %f %f\n\n", pitch, roll, yaw);
#endif
}

void
VRLogitechSpacemouse::getTranslationMatrix(osg::Matrix &mat)
{

    getRecord();

    // change the cood system to Performer
    float h;
    h = y;
    y = -z;
    z = h;
    roll = -roll;

    osg::Vec3 pos(x, y, z);

    // scale from inch to cm: 2.54
    pos *= 2.54f;

    // origin is the center of the screen
    // offset screen center - logitech transmitter origin is osg::Vec3 screenToTransmitterOffset
    // offset between logitech transmitter foot and logitech origin is systemOffset

    pos += systemOffset;
    pos += screenToTransmitterOffset;

    mat.makeTranslate(pos[0], pos[1], pos[2]);
}

void
VRLogitechSpacemouse::getRotationMatrix(osg::Matrix &mat)
{

    getRecord();

    // change the cood system to Performer
    float h;
    h = y;
    y = -z;
    z = h;
    roll = -roll;

    osg::Vec3 angles(yaw / 180.0f * M_PI, pitch / 180.0f * M_PI, roll / 180.0f * M_PI), pos(x, y, z);

    // scale from inch to cm: 2.54
    pos *= 2.54f;

    // origin is the center of the screen
    // offset screen center - logitech transmitter origin is osg::Vec3 screenToTransmitterOffset
    // offset between logitech transmitter foot and logitech origin is systemOffset

    pos += systemOffset;
    pos += screenToTransmitterOffset;

    osg::Matrix rx, ry, rz;

    rz.makeRotate(angles[0], 0.0f, 0.0f, 1.0f);
    rx.makeRotate(angles[1], 1.0f, 0.0f, 1.0f);
    ry.makeRotate(angles[2], 0.0f, 1.0f, 1.0f);

    mat.mult(ry, rx);
    mat.postMult(rz);
    for (int i = 0; i < 3; i++)
    {
        mat.ptr()[3 * 4 + i] = pos[i];
    }
//mat.setRow(3, pos);

#ifdef DBGPRINT1
    printf("Maus Pos: %f %f %f [cm]\n\n", pos[0], pos[1], pos[2]);
    printf("Maus pitch roll yaw: %f %f %f\n\n", pitch, roll, yaw);
#endif
}

void
VRLogitechSpacemouse::getMatrix_new(osg::Matrix &mat)
{

    getRecord();

    // change the coord system to Performer
    float h;
    h = y;
    y = -z;
    z = h;
    roll = -roll;

    float w = fcos(roll / 2.0f) * fcos(pitch / 2.0f) * fcos(yaw / 2.0f)
              + fsin(roll / 2.0f) * fsin(pitch / 2.0f) * fsin(yaw / 2.0f);

    float q1 = fcos(roll / 2.0f) * fsin(pitch / 2.0f) * fcos(yaw / 2.0f)
               + fsin(roll / 2.0f) * fcos(pitch / 2.0f) * fsin(yaw / 2.0f);

    float q2 = fcos(roll / 2.0f) * fcos(pitch / 2.0f) * fsin(yaw / 2.0f)
               + fsin(roll / 2.0f) * fsin(pitch / 2.0f) * fcos(yaw / 2.0f);

    float q3 = fsin(roll / 2.0f) * fcos(pitch / 2.0f) * fcos(yaw / 2.0f)
               + fcos(roll / 2.0f) * fsin(pitch / 2.0f) * fsin(yaw / 2.0f);

    //float phi = 2 * facos(w);

    //osg::Vec3 n, pos;
    //pfSetVec3(n, q1/fsin(phi/2), q2/fsin(phi/2), q3/fsin(phi/2));
    //pfMakeRotMat(mat, phi*180/M_PI, n[0], n[1], n[2]);

    //pfSetVec3(pos, x, y, z);
    //pfSetMatRowVec3(mat, 3, pos);

    osg::Quat quat(q1, q2, q3, w);
    quat.get(mat);
}

void
VRLogitechSpacemouse::mouseMode()
{
    if (write(fd, "*h", 2) != 2)
    {
        cerr << "VRLogitechTracker::mouseMode: short write" << endl;
    }
}

/************************************************************************/
/*									*/
/* 	VRLogitechHeadtracker	derived class				*/
/*									*/
/************************************************************************/

VRLogitechSensor::VRLogitechSensor()
{

    //headtrackerMode();
    //getCurrentOperatingInformation();
}

void
VRLogitechSensor::getPosition(osg::Vec3 &pos)
{
#ifdef DBGPRINT1
    printf("... VRLogitechSensor::getPosition\n");
#endif
    getRecord();

    // change the cood system to Performer
    float h;
    h = y;
    y = -z;
    z = h;

    pos.set(x, y, z);

    // scale from inch to cm: 2.54
    pos *= 2.54f;

    // origin is the center of the screen
    // offset screen center - logitech transmitter origin is osg::Vec3 screenToTransmitterOffset
    // offset between logitech transmitter foot and logitech origin is systemOffset

    pos = pos + systemOffset;
    pos = pos + screenToTransmitterOffset;

#ifdef DBGPRINT2
    printf("Head Pos: %f %f %f [cm]\n\n", pos[0], pos[1], pos[2]);
//printf("Head pitch roll yaw: %f %f %f\n\n",pitch, roll, yaw);
#endif

#ifdef DBGPRINT1
    printf("......VRLogitechSensor::getPosition: %f %f %f [cm]\n", pos[0], pos[1], pos[2]);
#endif
}

void
VRLogitechSensor::headtrackerMode()
{

    if (write(fd, "*H", 2) != 2)
    {
        cerr << "VRLogitechTracker::headtrackerMode: short write" << endl;
    }
}
