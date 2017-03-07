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
#include <covise/covise.h>
#include <util/unixcompat.h>

#ifdef __linux__
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <termio.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <signal.h>
#define sigset signal
#endif

#include "fob.h"

#define PERMS 0666

#define INCHES_IN_MM 25.4

#undef VERBOSE

int
fob::testConnection()
{
#ifdef VERBOSE
    fprintf(stderr, "fob::testConnection\n");
#endif

    return (connected);
}

void
fob::initQueues()
{
    if (serverRunning)
    {
        cerr << "fob::initQueues warn: server already started." << endl;
        return;
    }
    if (!askedForSystemStatus)
        getSystemStatus();
}

void fob::autoconfig()
{
    if (serverRunning)
    {
        cerr << "fob::autoconfig warn: server already started." << endl;
        return;
    }
#ifdef VERBOSE
    fprintf(stderr, "fob::autoconfig %d birds\n", numBirds);
#endif

    unsigned char cmdString[5];

    // autoconfig is sent only to the master
    cmdString[0] = 0xF1;
    cmdString[1] = 'P';
    cmdString[2] = 50;
    cmdString[3] = numBirds;
    sendSer(cmdString, 4);
    sleep(1);
}

void fob::setMeasurementRate(float rate)
{
    unsigned char cmdString[5];
    cmdString[0] = 0xF1;
    cmdString[1] = 'P';
    cmdString[2] = 6;
    unsigned short i = (unsigned)(4000.0 * (((1000. / rate) - 0.3 / 4.0)));
    cmdString[3] = (unsigned char)(i & 0xff);
    cmdString[4] = (unsigned char)((i & 0xff00) >> 8);
    sendSer(cmdString, 5);
    sleep(2);
    cerr << "Changing measurement rate to " << rate << endl;
}

void
fob::getSystemStatus()
{
    if (serverRunning)
    {
        cerr << "fob::getSystemStatus warn: server already started." << endl;
        return;
    }

#ifdef VERBOSE
    fprintf(stderr, "fob::getSystemStatus\n");
#endif
    int i;
    unsigned char cmdString[3];
    char dataString[14];

    // request system status from master
    cmdString[0] = 0xF1;
    cmdString[1] = 'O'; //'O'=examine
    cmdString[2] = 36; // 36=flock system status
    sendSer(cmdString, 3);

    receiveSer(dataString, 14);

    numReceivers = 0;
    numERCs = 0;

    int accessible = false;
    for (i = 0; i < 14; i++)
    {

        if (dataString[i] & 1 << 7) // test accessible (bit7)
        {
            accessible = true;
        }

        if (dataString[i] & 1 << 5) // test receiver (bit5)
        {
            if (accessible)
                numReceivers++;
        }

        if (dataString[i] & 1 << 4) // test ert (bit4)
        {
            numERCs = 1;
        }
    }

    askedForSystemStatus = true;
}

void
fob::printSystemStatus()
{
    if (serverRunning)
    {
        cerr << "fob::printSystemStatus warn: server already started." << endl;
        return;
    }

#ifdef VERBOSE
    fprintf(stderr, "fob::printSystemStatus\n");
#endif
    //test uwe fhkn
    //stopStreaming();
    flush();

    int i;
    unsigned char cmdString[3];
    char dataString[14];

    // request system status from master
    cmdString[0] = 0xF1;
    cmdString[1] = 'O'; //'O'=examine
    cmdString[2] = 36; // 36=flock system status
    sendSer(cmdString, 3);

    receiveSer(dataString, 14);

    fprintf(stderr, "  BirdNo\tAccess.\tRec.\tTrans.\tERT\n");
    for (i = 0; i < 14; i++)
    {
        fprintf(stderr, "  ");

        if (dataString[i] & 1 << 7) // test accessible (bit7)
        {
            fprintf(stderr, "%d\tX", i + 1);
        }
        else
            fprintf(stderr, "%d\t-", i + 1);

        if (dataString[i] & 1 << 5) // test receiver (bit5)
        {
            fprintf(stderr, "\tX");
        }
        else
            fprintf(stderr, "\t-");

        if (dataString[i] & 1) // test transmitter (bit0)
        {
            fprintf(stderr, "\tX");
        }
        else
            fprintf(stderr, "\t-");
        if (dataString[i] & 1 << 4) // test ert (bit4)
        {
            fprintf(stderr, "\tX\n");
        }
        else
            fprintf(stderr, "\t-\n");
    }
}

int
fob::getNumReceivers()
{
#ifdef VERBOSE
    fprintf(stderr, "fob::getNumReceivers num=%d\n", numReceivers);
#endif
    if (!askedForSystemStatus)
        getSystemStatus();

    return (numReceivers);
}

int fob::getNumERCs()
{
    //fprintf(stderr,"fob::getNumERCs num=%d\n", numERCs);

    if (!askedForSystemStatus)
        getSystemStatus();

    return (numERCs);
}

void
fob::stopStreaming()
{
#ifdef VERBOSE
    fprintf(stderr, "fob::stopStreaming\n");
#endif
    unsigned char cmdString[4];

    // as the flock is in group mode we don't need a bird address
    //cmdString[0] = 'V';
    //sendSer (cmdString, 1);

    // Don't know if a sleep is really necessary, but...
    //cmdString[0] = 'G';
    //sendSer (cmdString, 1);

    sendGroupMode();
    //Send a stream stop command
    cmdString[0] = 0x3F;
    sendSer(cmdString, 1);
    cmdString[0] = 0x2f;
    sendSer(cmdString, 1);

    /////////////////////////////////
    this->flush();
}

void
fob::enableStreamMode()
{
    if (serverRunning)
    {
        cerr << "fob::enableStreamMode warn: server already started." << endl;
        return;
    }
#ifdef VERBOSE
    fprintf(stderr, "fob::enableStreamMode\n");
#endif
    sendGroupMode();

    // enable stream mode
    unsigned char cmdString[4];

    cmdString[0] = '@';
    sendSer(cmdString, 1);
}

void
fob::sendGroupMode()
{
    if (serverRunning)
    {
        cerr << "fob::sendGroupMode warn: server already started." << endl;
        return;
    }
    // send the group mode command to the master
    unsigned char cmdString[4];

    cmdString[0] = 0xF1;
    cmdString[1] = 'P';
    cmdString[2] = 35;
    cmdString[3] = 1;
    sendSer(cmdString, 4);
}

void processFOB(void *userdata)
{

    fob *bt = (fob *)userdata;
    while (1)
    {
        bt->processSerialStream();
    }
}

void
fob::processSerialStream()
{
    if (serverRunning)
    {
        cerr << "fob::processSerialStream warn: server already started." << endl;
        return;
    }
    // static int synced=0;
    unsigned int i;
    int o;
    unsigned char *udataString;
    char dataString[1024];
    float ratio;
    int addr; //bird address
    int firstReceiverAddress;
    unsigned char cmdString[4];

    if (!mode)
    {
        // POINT command
        cmdString[0] = 'B';
        sendSer(cmdString, 1);
    }

    udataString = (unsigned char *)dataString;
    //      receiveSer( dataString, 1024 );
    //      while(1)
    ;
    //      printf("--------------------------\n");

    //standard range
    if (extendedRange)
    {
        ratio = 72.0f * ((float)INCHES_IN_MM);
    }
    else
    {
        ratio = 36.0f * ((float)INCHES_IN_MM);
    }
    // ERC
    if (getNumERCs() > 0)
        ratio = 144.0f * ((float)INCHES_IN_MM);

    // get one byte and see if we're really synced
    // look if the first byte contains the phasing bit
    //    7 6 5 4 3 2 1 0
    // -> 1 x x x x x x x
    //    0 x x x x x x x
    //    0 y y y y y y y
    //    0 y y y y y y y
    //    0 . . . . . . .
    //    0 . . . . . . .
    //    0 0 0 a a a a a address only in group mode

    // we have to find a new start
    do
    {
        receiveSer(dataString, 1);
        if ((*terminate == true)
#ifdef VISENSO
#ifndef _WIN32
            || (getppid() == 1)
#endif
#endif
                )
        {
            fprintf(stderr, "SERVER: exit\n");
            stopStreaming();
            exit(1);
        }
    } while (!(dataString[0] & (1 << 7)));
    //fprintf(stderr,"found\n");

    // we already have one byte
    // if we get now numBytesPerData bytes we get
    // also the group byte
    receiveSer(dataString + 1, numBytesPerData);

    // see if we are really synced
    // we look for the first address of the first bird with receiver
    // for a flock without ERC this is bird1
    // for a flock with ERC this is bird2
    // (requires: flock starts at address1 and birds at consecutive addresses)
    if (numReceivers == numBirds)
        firstReceiverAddress = 1;
    else
        firstReceiverAddress = 2;
    //      cout << "vorher dataString[numBytesPerData] :" << (int)dataString[numBytesPerData] << endl;
    while (dataString[numBytesPerData] != firstReceiverAddress)
    {
        do
        {
            //      fprintf(stderr,". ");
            receiveSer(dataString, 1);
            if ((*terminate == true)
#ifdef VISENSO
#ifndef _WIN32
                || (getppid() == 1)
#endif
#endif
                    )
            {
                fprintf(stderr, "SERVER: exit\n");
                stopStreaming();
                exit(1);
            }
        } while (!(dataString[0] & (1 << 7)));
        //fprintf(stderr,"found\n");

        // we already have one byte
        // if we get now numBytesPerData bytes we get
        // also the group byte
        receiveSer(dataString + 1, numBytesPerData);
        //              cout << "dataString[numBytesPerData] :" << (int)dataString[numBytesPerData] << endl;
    }

    // now we are really really synced
    // so read the next fbbs
    receiveSer(dataString + numBytesPerData + 1,
               (numBytesPerData + 1) * (numReceivers - 1));

    o = 0;
    for (i = 0; i < numReceivers; i++)
    {

        //bev.setValue((int)dataString[o+numBytesPerData-1]);
        //bev.setTimestamp();
        //queues[i]->putEvent(bev);

        switch (ourDataFormat)
        {
        case fob::FLOCK_POSITIONMATRIX:
            addr = (int)udataString[o + numBytesPerData];
            //            int n;
            //             for(n=0;n<13;n++)
            //                    printf("%x %x\n",udataString[o+2*n], udataString[o+1+2*n]);
            //             printf("\n\n");
            // Fixed wrong adressing of buttons - was i now is addr - RWB

            if (addr >= 0 && addr < maxNumReceivers)
            {
                receivers[addr].buttons = (int)dataString[o + numBytesPerData - 1];
                receivers[addr].x = getSerPositionValue(udataString + o) * ratio;
                receivers[addr].y = getSerPositionValue(udataString + o + 2) * ratio;
                receivers[addr].z = getSerPositionValue(udataString + o + 4) * ratio;
                receivers[addr].m[0][0] = getSerMatrixValue(udataString + o + 6);
                receivers[addr].m[1][0] = getSerMatrixValue(udataString + o + 8);
                receivers[addr].m[2][0] = getSerMatrixValue(udataString + o + 10);
                receivers[addr].m[0][1] = getSerMatrixValue(udataString + o + 12);
                receivers[addr].m[1][1] = getSerMatrixValue(udataString + o + 14);
                receivers[addr].m[2][1] = getSerMatrixValue(udataString + o + 16);
                receivers[addr].m[0][2] = getSerMatrixValue(udataString + o + 18);
                receivers[addr].m[1][2] = getSerMatrixValue(udataString + o + 20);
                receivers[addr].m[2][2] = getSerMatrixValue(udataString + o + 22);
            }
            break;

        default:
            fprintf(stderr, "df not implemented\n");
            break;
        }
        o += numBytesPerData + 1;
    }
    // done
    return;
}

float
fob::getSerPositionValue(unsigned char *ptr)
{
    short s;

    s = (short)(((short)(*(unsigned char *)ptr) & 0x7F) | (short)(*((unsigned char *)ptr + 1)) << 7) << 2;
    return ((float)s / (float)0x7FFF);
}

float
fob::getSerMatrixValue(unsigned char *ptr)
{
    short s;
    s = (short)(((short)(*(unsigned char *)ptr) & 0x7F) | (short)(*((unsigned char *)ptr + 1)) << 7) << 2;
    return ((float)s / (float)0x7FFF);
}

void
fob::setDataFormat(int birdAddress, dataformat df)
{
    if (serverRunning)
    {
        cerr << "fob::setDataFormat warn: server already started." << endl;
        return;
    }
#ifdef VERBOSE
    fprintf(stderr, "DATA FORMAT bird %d ", birdAddress);
#endif

    unsigned char cmdString[3];

    numBytesPerData = 0;

    cmdString[0] = 0xF0 + birdAddress;

    switch (df)
    {
    case fob::FLOCK_POSITION:
        cmdString[1] = 'V';
        numBytesPerData += 6;
        break;
    case fob::FLOCK_ANGLES:
        cmdString[1] = 'W';
        numBytesPerData += 6;
        break;
    case fob::FLOCK_MATRIX:
        cmdString[1] = 'X';
        numBytesPerData += 18;
        break;
    case fob::FLOCK_POSITIONANGLES:
        cmdString[1] = 'Y';
        numBytesPerData += 12;
        break;
    case fob::FLOCK_POSITIONMATRIX:
        cmdString[1] = 'Z';
        numBytesPerData += 24;
        break;
    case fob::FLOCK_QUATERNION:
        cmdString[1] = 0x5C;
        numBytesPerData += 8;
        break;
    case fob::FLOCK_POSITIONQUATERNION:
        cmdString[1] = ']';
        numBytesPerData += 14;
        break;
    default:
        cerr << "setDataFormat wrong format" << endl;
        break;
    }
    sendSer(cmdString, 2);
    ourDataFormat = df;
    // get the button byte for all receivers also of its
    // not have a 6DOF mouse
    // this makes finding the phasing bit easier
    cmdString[0] = 0xF0 + birdAddress;
    cmdString[1] = 'M';
    cmdString[2] = 0x01;
    sendSer(cmdString, 3);

    numBytesPerData++;
    fprintf(stderr, "numBytesPerData=%d\n", numBytesPerData);
}

void
fob::setHemisphere(int birdAddress, hemisphere hemi)
{
    if (serverRunning)
    {
        cerr << "fob::setHemisphere warn: server already started." << endl;
        return;
    }
#ifdef VERBOSE
    fprintf(stderr, "HEMISPHERE bird %d = %d\n", birdAddress, (int)hemi);
#endif

    unsigned char cmdString[4];

    cmdString[0] = 0xF0 + birdAddress;

    cmdString[1] = 'L';
    switch (hemi)
    {
    case fob::FRONT_HEMISPHERE:
        cmdString[2] = 0x00;
        cmdString[3] = 0x00;
        break;
    case fob::REAR_HEMISPHERE:
        cmdString[2] = 0x00;
        cmdString[3] = 0x01;
        break;
    case fob::UPPER_HEMISPHERE:
        cmdString[2] = 0x0C;
        cmdString[3] = 0x01;
        break;
    case fob::LOWER_HEMISPHERE:
        cmdString[2] = 0x0C;
        cmdString[3] = 0x00;
        break;
    case fob::LEFT_HEMISPHERE:
        cmdString[2] = 0x06;
        cmdString[3] = 0x01;
        break;
    case fob::RIGHT_HEMISPHERE:
        cmdString[2] = 0x06;
        cmdString[3] = 0x00;
        break;
    }

    sendSer(cmdString, 4);
}

void
fob::setFilter(int birdAddress, int ac_nn, int ac_wn, int dc)
{
    if (serverRunning)
    {
        cerr << "fob::setFilter warn: server already started." << endl;
        return;
    }
    unsigned char cmdString[5];

    cmdString[0] = 0xF0 + birdAddress;
    cmdString[1] = 'P';
    cmdString[2] = 4;
    cmdString[3] = 0x00;
    cmdString[4] = 0x00;
    if (ac_nn)
        cmdString[3] |= (1 << 2);
    if (ac_wn)
        cmdString[3] |= 2;
    if (dc)
        cmdString[3] |= 1;
    sendSer(cmdString, 5);
}

void
fob::lockSuddenChange(int birdAddress)
{

    if (serverRunning)
    {
        cerr << "fob::lockSuddenChange warn: server already started." << endl;
        return;
    }
    unsigned char cmdString[4];

    cmdString[0] = 0xF0 + birdAddress;
    cmdString[1] = 'P';
    cmdString[2] = 14;
    cmdString[3] = 1;

    sendSer(cmdString, 4);
}

void
fob::reportRate(char rate)
{
    if (serverRunning)
    {
        cerr << "fob::reportRate warn: server already started." << endl;
        return;
    }
    if ((rate == 'Q') || (rate == 'R') || (rate == 'S') || (rate == 'T'))
    {

        fprintf(stderr, "setting report rate to %c\n", rate);
        unsigned char cmdString[4];

        cmdString[0] = rate;
        sendSer(cmdString, 1);
    }
    else
    {
        fprintf(stderr, "report rate '%c' not supported\n", rate);
    }
}

void
fob::changeRange(int birdAddress, int range) /* 1 full 0 half(default) */
{
    if (serverRunning)
    {
        cerr << "fob::changeRange warn: server already started." << endl;
        return;
    }

    if (numERCs == 0)
    {
        extendedRange = range;
        unsigned char cmdString[5];

        cmdString[0] = 0xF0 + birdAddress;
        cmdString[1] = 'P';
        cmdString[2] = 3;
        cmdString[3] = range;
        cmdString[4] = 0;

        sendSer(cmdString, 5);
    }
}

int
fob::getPositionMatrix(int nr, float *x, float *y, float *z,
                       float *m00, float *m01, float *m02, float *m10,
                       float *m11, float *m12, float *m20, float *m21,
                       float *m22)
{
    // get data
    *x = receivers[nr].x;
    *y = receivers[nr].y;
    *z = receivers[nr].z;
    *m00 = receivers[nr].m[0][0];
    *m01 = receivers[nr].m[0][1];
    *m02 = receivers[nr].m[0][2];
    *m10 = receivers[nr].m[1][0];
    *m11 = receivers[nr].m[1][1];
    *m12 = receivers[nr].m[1][2];
    *m20 = receivers[nr].m[2][0];
    *m21 = receivers[nr].m[2][1];
    *m22 = receivers[nr].m[2][2];

    // done
    return (0);
}

int
fob::getButtons(int nr, unsigned short int *buttons)
{
    // set the button-state
    *buttons = receivers[nr].buttons;

    // done
    return (0);
}

void
fob::printBitMask(char b)
{
    int i;

    for (i = 0; i < 8; i++)
        if (b & (1 << i))
            fprintf(stderr, "1");
        else
            fprintf(stderr, "0");

    fprintf(stderr, "\n");
    return;
}

int
fob::flush()
{
    char buf[100];
    receiveSer(buf, 100);
    return (1);
}
