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
 *	File			logitechTracking.h (Performer 2.0)	*
 *									*
 *	Description		logitech tracker class			*
 *				based on logidrvr from NASA AMES	*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			27. October 96				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#ifndef __VR_LOGITECH_TRACKER_H
#define __VR_LOGITECH_TRACKER_H

#include <util/common.h>
#include <sys/stat.h> /* open */

#ifndef _WIN32
#include <termios.h> /* tcsetattr */
#endif
#include <limits.h> /* sginap */
#include <osg/Vec3>
#include <osg/Matrix>

#undef buttons
#undef setAttributes
#undef lines
#undef num_labels

#ifndef true
#define true 1
#endif

#ifndef false
#define false 0
#endif

/* record sizes */
#define DIAGNOSTIC_SIZE 2
#define EULER_RECORD_SIZE 16

/* euler record "button" bits */
#define logitech_FLAGBIT 0x80
#define logitech_FRINGEBIT 0x40
#define logitech_OUTOFRANGEBIT 0x20
#define logitech_RESERVED 0x10
#define logitech_SUSPENDBUTTON 0x08
#define logitech_LEFTBUTTON 0x04
#define logitech_MIDDLEBUTTON 0x02
#define logitech_RIGHTBUTTON 0x01

#define logitech_MOUSE 14
#define logitech_HEAD_TRACKER 13
#define logitech_CRYSTAL_EYES 11

// base class

class VRLogitechTracker
{

protected:
    float x;
    float y;
    float z;
    float pitch;
    float yaw;
    float roll;
    char buttons;

    osg::Vec3 screenToTransmitterOffset;
    osg::Vec3 systemOffset;

    int isSlave;
    int receiverType;

    void getRecord();
    int fd;
    void getCurrentOperatingInformation();

private:
    /* for diagnostics info */
    unsigned char diagnosticsData[DIAGNOSTIC_SIZE];

    char portname[10];
    int openSerialPort();
    int closeSerialPort();

    //	void cu_incremental_reporting (int fd);
    //	void cu_euler_mode (int fd);
    //	void cu_headtracker_mode (int fd);
    //	void cu_mouse_mode (int fd);

    void setDemandReportingMode();
    void requestDiagnostics();
    void requestReport();
    void resetControlUnit();
    void setFilter();
    void eulerToAbsolute(char record[]);

    void getDiagnostics();

    void setSlaveTransmitterType(int type);

    //	void change_coordsyst(MouseRecordPtr data);

public:
    VRLogitechTracker();
    ~VRLogitechTracker();
    int init(char *portname, osg::Vec3 &offset);
};

class VRLogitechSpacemouse : public VRLogitechTracker
{

private:
    void mouseMode();

public:
    VRLogitechSpacemouse();

    void getRotationMatrix(osg::Matrix &rotMat);

    void getTranslationMatrix(osg::Matrix &transMat);

    void getMatrix(osg::Matrix &mat);
    void getMatrix_new(osg::Matrix &mat);

    void getButtons(char *buttons);
};

class VRLogitechSensor : public VRLogitechTracker
{

private:
    void headtrackerMode();

public:
    VRLogitechSensor();

    void getPosition(osg::Vec3 &pos);
};
#endif
