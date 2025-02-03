/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1999                                  *
 *                                                                      *
 *                                                                      *
 *                       D-70469 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 *Descriptionflock of birds low level driver class for windows          *
 *no! extra process for reading the                                     *
 *serial port                                                           *
 *                                                                      *
 *AuthorL. Frenzel, D. Rainer, U. Woessner                              * 
 *                                                                      *
 *DateMai 99                                                            *
 *                                                                      *
 ************************************************************************/
#ifndef _WINFOB_H_
#define _WINFOB_H_

// low level flock of birds driver
// classes: fob, birdReceiver
// supports
// - standard transmitter or erc/ert
// - normal receiver or 6dof mouse
//
// restrictions:
// - only one serial connection to the master bird is supported
// - master bird at address 1 (master bird is the one with the serial connection)
// - other birds have to be at consecutive address (2, 3. 4,...)
// - if erc/ert is used, erc has to be he master
// - supports only the steram mode
// - only one instance of this driver class is allowed.

#ifdef WIN32
#include "windows.h"
#endif

//#include "coQueue.h"
class birdReceiver
{
protected:
public:
    birdReceiver(){};
    ~birdReceiver(){};

    unsigned int address;
    unsigned short int buttons;
    int buttons_flag;

    int add_button_data;
    signed short int range;

    float x, y, z;
    float u, v, w, a;

    float h, p, r;
    float m[3][3];
};

class fob
{
private:
#ifdef WIN32
    HANDLE serialChannel;
#else
    int serialChannel; //file descriptor
#endif
    bool serverRunning;

    int childID;
    bool *terminate;

    int numBytesPerData; //24 for posmatrix +1 for each 6DOF
    int connected;
    int extendedRange;
    int mode; //point or stream

    unsigned int numBirds; // number of birds (ERC is bird without receiver)
    unsigned int numReceivers;
    //coQueue **queues;
    //coEvent bev; // buttonEvent
    int numERCs;
    int askedForSystemStatus; // flag if already asked for system status
    birdReceiver *receivers; // array of receivers, index ranges from 0 to 40
    // receivers[0]: always zero because fobs first address is 1
    // receivers[1]: zero for ERC or values for bird #1
    // receivers[2]: values for bird #2
    int receiveSer(char *bfr, int num);
    int sendSer(unsigned char *bfr, int num);
    int flush();

    float getSerPositionValue(unsigned char *ptr);
    float getSerMatrixValue(unsigned char *ptr);
    void printBitMask(char b);
    bool _stopping;

#ifdef VISENSO
#ifndef _WINDOWS
    void allocSharedMemoryData();
#endif
#endif

public:
    void setStopping()
    {
        _stopping = true;
    };
    static const int maxNumReceivers = 40;

    enum hemisphere
    {
        FRONT_HEMISPHERE = 0,
        REAR_HEMISPHERE = 1,
        UPPER_HEMISPHERE = 2,
        LOWER_HEMISPHERE = 3,
        LEFT_HEMISPHERE = 4,
        RIGHT_HEMISPHERE = 5
    };

    enum dataformat
    {
        FLOCK_NOBIRDDATA = 0,
        FLOCK_POSITION = 1,
        FLOCK_ANGLES = 2,
        FLOCK_MATRIX = 3,
        FLOCK_POSITIONANGLES = 4,
        FLOCK_POSITIONMATRIX = 5,
        FLOCK_QUATERNION = 7,
        FLOCK_POSITIONQUATERNION = 8,
        FLOCK_FEEDTHROUGH_DATA = 14,
        FLOCK_ERROR = 15
    };

    dataformat ourDataFormat;

    fob(const char *portname, int baudrate, int nb, int fmode);
    ~fob();

    // testing the connection to serial port
    int testConnection();

    // user configuration
    void reset();
    void autoconfig();
    void getSystemStatus();
    void printSystemStatus();
    void setDataFormat(int birdAddress, dataformat df);
    void setHemisphere(int birdAddress, hemisphere hemi);
    void setFilter(int birdAddress, int ac_nn, int ac_wn, int dc);
    void lockSuddenChange(int birdAddress);
    void setMeasurementRate(float rate);

    // report rate can be
    // 'Q' (report rate = every measuremnt cycle, default)
    // 'R' (report rate = every other measuremnt cycle)
    // 'S' (report rate = every eight measuremnt cycle)
    // 'T' (report rate = every thirty-two measuremnt cycle)
    void reportRate(char rate);

    /* 1 full 0 half(default)*/
    void changeRange(int birdAddress, int range);
    void stopStreaming();
    void enableStreamMode();
    void sendGroupMode();
    void initQueues();
    int getNumReceivers(); // return no of receivers 0-16
    int getNumERCs(); // return no of ERCs 0 or 1

#ifdef VISENSO
#ifndef _WINDOWS
    void startServerProcess();
#endif
#endif
    void processSerialStream();

    int getPositionMatrix(int nr, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    int getButtons(int nr, unsigned short int *buttons);
    //coEvent *getButtonEvent(int nr);
};
#endif
