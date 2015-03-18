/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996-2001                             *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *	File			coVRTrackingSystems.cpp                   *
 *                                                                      *
 *	Author			D. Rainer, U. Woessner                  *
 *                                                                      *
 ************************************************************************/
// CAVELIB support is for CAVELIB 2.6

#include <util/common.h>
#include <signal.h>
using std::ios;
using std::flush;

#ifndef _WIN32
#include <sys/shm.h>
#include <sys/ipc.h>
#else
#include "HeadFind.h"
#endif

#ifdef __sgi
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#endif

#include <cover/coVRFileManager.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginList.h>
#include <cover/VRSceneGraph.h>
#include <OpenVRUI/osg/mathUtils.h>
#include "coVRTrackingSystems.h"
#include "VRTracker.h"
#include "coVRTrackingUtil.h"
#include <cover/coVRPluginSupport.h>
#include "DTrack.h"
#include "VRCTracker.h"

#include <config/CoviseConfig.h>

#include <osg/Matrix>
using namespace covise;
using namespace opencover;

#undef DBGPRINT

unsigned char *buttonData;
float *analogData; // for IWR
void *g_motion;
int outData[3] = { 0, 0, 0 };

float staticViewerX = 0.0;
float staticViewerY = -2000;
float staticViewerZ = 0.0;

#ifdef __linux__
#include <asm/ioctls.h>
#ifndef fsin
#define fsin sin
#endif
#ifndef fcos
#define fcos cos
#endif
#ifndef facos
#define facos acos
#endif
//#  define sigset signal
#endif

#if defined(__linux__)
#define DEFAULTSERIAL "/dev/ttyS0"
#elif defined(__sgi)
#define DEFAULTSERIAL "/dev/ttyd2"
#elif defined(_WIN32)
#define DEFAULTSERIAL "COM1"
#else
#define DEFAULTSERIAL "/dev/null"
#endif

#include "MouseButtons.h"
#include "Tarsus.h"
#include "SSD.h"
#include "VRPN.h"
#include "DynaSight.h"
#include "DTrack.h"
#include "CGVTrack.h"
#include "serialcom.h"

#if !defined(_WIN32) && !defined(__APPLE__)
#include <termio.h>
#include <termios.h>
#endif

#ifndef _WIN32
#define SHMKEY ((key_t)321)
#define PERMS 0666
#endif

//double start, now;
//struct itimerval itv;
//void catcher(int)
//{
//}

#ifndef __APPLE__
#ifndef _WIN32
static ssize_t checkedWrite(int fd, const void *buf, size_t count)
{
    ssize_t n = write(fd, buf, count);
    if (n == ssize_t(count))
    {
        cerr << "write failed: " << strerror(errno) << endl;
    }
    else if (n != ssize_t(count))
    {
        cerr << "short write (" << n << "/" << count << endl;
    }

    return n;
}
#endif

#ifdef _WIN32
int cmdPinch(HANDLE pinchfd, const char *command, unsigned char *reply);
#else
int cmdPinch(int pinchfd, const char *command, unsigned char *reply);
#endif

#ifdef _WIN32
HANDLE openPinch(char *device, int baudRate)
#else
int openPinch(char *ttyport, int baudRate)
#endif
{
#ifdef _WIN32
    DCB dcb;
    BOOL fSuccess;
#define BUF_LEN 100
    unsigned char buf[BUF_LEN];
    int cnt;
    HANDLE pinchfd = 0;
    //device ==  "COM2";

    pinchfd = CreateFile(device,
                         GENERIC_READ | GENERIC_WRITE,
                         0, // must be opened with exclusive-access
                         NULL, // no security attributes
                         OPEN_EXISTING, // must use OPEN_EXISTING
                         0, // not overlapped I/O
                         NULL // hTemplate must be NULL for comm devices
                         );

    if (pinchfd == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", device, GetLastError());
        return (false);
    }

    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    fSuccess = GetCommState(pinchfd, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

    /* First, get the parameters which can be set in the Preferences */
    switch (baudRate)
    {
    case 1200:
        dcb.BaudRate = CBR_1200;
        break;
    case 2400:
        dcb.BaudRate = CBR_2400;
        break;
    case 4800:
        dcb.BaudRate = CBR_4800;
        break;

    case 9600:
        dcb.BaudRate = CBR_9600;
        break;

    case 19200:
        dcb.BaudRate = CBR_19200;
        break;

    case 38400:
        dcb.BaudRate = CBR_38400;
        break;
    case 57600:
        dcb.BaudRate = CBR_57600;
        break;

    default:
        dcb.BaudRate = CBR_19200;
        break;
    }
    // Fill in rest of DCB:  8 data bits, no parity, and 1 stop bit.

    dcb.ByteSize = 8; // data size, xmit, and rcv
    dcb.Parity = NOPARITY; // no parity bit
    dcb.StopBits = ONESTOPBIT; // one stop bit

    fSuccess = SetCommState(pinchfd, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

#else
    struct termio data;
#define BUF_LEN 100
    unsigned char buf[BUF_LEN];
    int cnt;
    int pinchfd = -1;

    data.c_lflag = 0;
    data.c_iflag = 1;
    data.c_oflag = 0;
    data.c_cc[VMIN] = 1;
    data.c_cflag = CS8 | CREAD | CLOCAL;
#ifndef _OLD_TERMIOS
    data.c_ospeed = baudRate;
    data.c_ispeed = baudRate;
#else
    switch (baudRate)
    {
    case 50:
        data.c_cflag |= B50;
        break;
    case 75:
        data.c_cflag |= B75;
        break;
    case 134:
        data.c_cflag |= B134;
        break;
    case 150:
        data.c_cflag |= B150;
        break;
    case 200:
        data.c_cflag |= B200;
        break;
    case 300:
        data.c_cflag |= B300;
        break;
    case 600:
        data.c_cflag |= B600;
        break;
    case 1200:
        data.c_cflag |= B1200;
        break;
    case 1800:
        data.c_cflag |= B1800;
        break;
    case 2400:
        data.c_cflag |= B2400;
        break;
    case 4800:
        data.c_cflag |= B4800;
        break;
    case 9600:
        data.c_cflag |= B9600;
        break;
    case 19200:
        data.c_cflag |= B19200;
        break;
    case 38400:
        data.c_cflag |= B38400;
        break;
    default:
        data.c_cflag |= B9600;
    }
#endif

    if ((pinchfd = open(ttyport, O_RDWR | O_NDELAY)) == -1)
        return (0);

    if (ioctl(pinchfd, TCSETAF, &data) < 0)
        return (0);
#ifdef _sgi
    sginap(15);
#endif

    if (ioctl(pinchfd, TCFLSH, 2) < 0)
        return (0);

#ifdef _sgi
    sginap(15);
#endif
#endif
    // Turn time stamps on
    cmdPinch(pinchfd, "T1", buf);
    if (buf[1] != '1')
    {
        printf("could not turn time stamps on\n");
#ifdef _WIN32
        return (INVALID_HANDLE_VALUE);
#else
        return (-1);
#endif
    } // Version compatability
    cmdPinch(pinchfd, "V1", buf);
    if (buf[1] != '1')
    {
        printf("could not set to version 1 formatting\n");
#ifdef _WIN32
        return (INVALID_HANDLE_VALUE);
#else
        return (-1);
#endif
    }

    // Get the configuration information and print it
    printf("Configuration:\n");
    // get rid of 0x8F
    cnt = cmdPinch(pinchfd, "CP", buf);
    buf[cnt - 1] = 0;
    printf("  %s\n", &buf[1]);
    // get rid of 0x8F
    cnt = cmdPinch(pinchfd, "CL", buf);
    buf[cnt - 1] = 0;
    printf("  %s\n", &buf[1]);
    // get rid of 0x8F
    cnt = cmdPinch(pinchfd, "CR", buf);
    buf[cnt - 1] = 0;
    printf("  %s\n", &buf[1]);
    // get rid of 0x8F
    cnt = cmdPinch(pinchfd, "CT", buf);
    buf[cnt - 1] = 0;
    printf("  %s\n", &buf[1]);

    return (pinchfd);
}

#ifdef _WIN32
int readPinch(HANDLE pinchfd, int /*rec_max_len*/, unsigned char *records)
#else
int readPinch(int pinchfd, int /*rec_max_len*/, unsigned char *records)
#endif
{
    int numbytes = 0;
    unsigned char buf[2048];
    clock_t t1 = 0, t2 = 0;

#define START_BYTE_DATA 0x80
#define START_BYTE_DATA_TS 0x81
#define START_BYTE_TEXT 0x82
#define END_BYTE 0x8F

    records[0] = 0;
#ifdef _WIN32
    DWORD s;
    bool bResult;

    //int te=  ;
    bResult = ReadFile(pinchfd, buf, 1, &s, NULL) != 0;
    while (s)
#else
    int s;
    while ((s = read(pinchfd, buf, 1)))
#endif
    {
        if ((buf[0] == START_BYTE_DATA) || (buf[0] == START_BYTE_DATA_TS) || (buf[0] == START_BYTE_TEXT))
        {

            records[numbytes++] = buf[0];
            t1 = clock();
            t2 = clock();
            if (t1 > t2)
            {
                t1 = t2;
                t2 = clock();
            }
            do
            {
#ifdef _WIN32
                bool bResult;
                bResult = ReadFile(pinchfd, &records[numbytes], 1, &s, NULL) != 0;
#else
                s = read(pinchfd, &records[numbytes], 1);
#endif
                numbytes += s;
                if (t1 > t2)
                {
                    t1 = t2;
                    t2 = clock();
                }
            } while ((records[numbytes - s] != END_BYTE) && ((clock_t)t2 - (clock_t)t1 < CLOCKS_PER_SEC));
        }
        if (t1 > t2)
        {
            t1 = t2;
            t2 = clock();
        }
    }

    return numbytes;
}

#ifdef _WIN32
int cmdPinch(HANDLE pinchfd, const char *command, unsigned char *reply)
#else
int cmdPinch(int pinchfd, const char *command, unsigned char *reply)
#endif
{
    char buf[100];
    static int first = 1;

#ifdef _WIN32
    bool bResult;
    DWORD nBytesWritten = 0;
    bResult = WriteFile(pinchfd, "*", 1, &nBytesWritten, NULL) != 0;
    nBytesWritten = 0;
    bResult = ReadFile(pinchfd, buf, 3, &nBytesWritten, NULL) != 0;
    if (nBytesWritten != 3)
    {
        bResult = WriteFile(pinchfd, "*", 1, &nBytesWritten, NULL) != 0;
        bResult = ReadFile(pinchfd, buf, 3, &nBytesWritten, NULL) != 0;
    }
    // Send the 2 byte command
    bResult = WriteFile(pinchfd, &command[0], 1, &nBytesWritten, NULL) != 0;
    bResult = WriteFile(pinchfd, &command[1], 1, &nBytesWritten, NULL) != 0;
#else
    if (first)
    {
        first = 0;
        checkedWrite(pinchfd, "*", 1);
#ifdef _sgi
        sginap(45);
#endif
        if (!read(pinchfd, buf, 3))
        {
            checkedWrite(pinchfd, "*", 1);
#ifdef _sgi
            sginap(45);
#endif
            if (read(pinchfd, buf, 3) != 3)
            {
                cerr << "cmdPinch: short read" << endl;
            }
        }
#ifdef _sgi
        sginap(25);
#endif
    }

    // Send the 2 byte command
    checkedWrite(pinchfd, &command[0], 1);
    if (ioctl(pinchfd, TCFLSH, 1) < 0)
        return (0);
#ifdef _sgi
    sginap(15);
#endif
    checkedWrite(pinchfd, &command[1], 1);
    if (ioctl(pinchfd, TCFLSH, 1) < 0)
        return (0);
#ifdef _sgi
    sginap(45);
#endif
#endif
    return (readPinch(pinchfd, 100, reply));
}
void processPinch(void *userdata)
{

    coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;
    int num;
    unsigned char rec[BUF_LEN];
    char data[BUF_LEN];
    int i, touch_count, newButton;

    while (1)
    {
#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        if ((num = readPinch(bt->pinchfd, BUF_LEN, rec)))
        {
            switch (rec[0])
            {
            case 0x80:
            /* Hand data only */
            case 0x81:
                /* Hand data and time stamp */
                //sprintf(record,"Rec = %02X",rec[0]);

                if (rec[0] == 0x80)
                {
                    for (i = 1; i < num - 2; i++)
                        data[i - 1] = rec[i];
                    touch_count = (num - 2) / 2;
                }
                else
                {
                    for (i = 1; i < num - 3; i++)
                        data[i - 1] = rec[i];
                    touch_count = (num - 4) / 2;
                }

                newButton = 0;
                for (i = 0; i < touch_count; i++)
                {
                    // sprintf(temp," %02X.%02X",data[2*i],data[2*i+1]);
                    // Right
                    newButton |= data[2 * i + 1] & 0xF;
                    // Left
                    newButton |= (data[2 * i] & 0xF) << 4;
                    // strcat(record,temp);
                }

                buttonData[0] = newButton;
                //fprintf(stderr,"Button: %x\n",newButton);
                break;
            case 0x82:
                /* Text reply */
                break;
            default:
                /* some new packrt type not understood. */
                break;
            }
        }
    }
}
#endif /* __APPLE__ */

#ifndef __APPLE__
void handleVP(void *userdata)
{

    coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;
    int i;
    unsigned char inputData[10];
    while (1)
    {

#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        get_answer(4, (unsigned char *)inputData);

        if (~inputData[3] & 0xc0)
        {
            fprintf(stderr, "VP Button: out of sync\n");
            i = 0;
            while (1)
            {
                inputData[0] = 0;
                get_answer(1, (unsigned char *)inputData);
                if (!(~inputData[0] & 0xc0))
                    break;
                if (i > 5)
                {
                    fprintf(stderr, "error no valid button data from port %s !!\n", bt->buttonDevice);
                    exit(0);
                }
                i++;
            }
        }
        else
        {
            char string[40];
            int mask, b = 0;
            for (i = 0; i < 4; i++)
            {
                mask = 1;
                while (mask < 129)
                {
                    if (inputData[i] & mask)
                        string[b] = '1';
                    else
                        string[b] = '0';
                    string[b + 1] = '\0';
                    mask <<= 1;
                    b++;
                }
                string[b + 1] = ' ';
                b++;
            }
            fprintf(stderr, "%s\n", string);
            buttonData[0] = (((~inputData[0]) & 0x1f) | (~inputData[1]) << 5);
        }
    }
}
#endif /* __APPLE__ */

#ifndef __APPLE__
void handleCYBER(void * /*userdata*/)
{

    //coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;

    //int i;
    while (1)
    {
#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        get_answer(8, buttonData);
        if (buttonData[0] != 255 || buttonData[1] != 255)
        {
            unsigned char oldByte = 0;
            unsigned char Byte = 0;
            cerr << "Cyberstick Out of sync" << endl;
            while (oldByte != 255 || Byte != 255)
            {
                oldByte = Byte;
                get_answer(1, &Byte);
            }
            get_answer(6, buttonData + 2);
            cerr << "Cyberstick resynced" << endl;
        }
        static int counter = 0;
        static int pcounter = 0;
        static int ncounter = 0;
        //for(i=0;i<8;i++)
        //    fprintf(stderr,"%2x ",buttonData[i]);
        //fprintf(stderr,"  %d p:%d n:%d\n",counter,pcounter,ncounter);
        if (buttonData[3] == 1)
        {
            counter++;
            pcounter++;
        }
        if (buttonData[3] == 2)
        {
            counter--;
            ncounter++;
        }
        buttonData[0] = 0;
    }
}
#endif /* __APPLE__ */

#ifndef __APPLE__
void handleHORNET(void * /*userdata*/)
{

    //coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;

    char command[5];
    command[0] = 'i';
    while (1)
    {
#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        send_command(command, 1);
        get_answer(5, buttonData);
        fprintf(stderr, "%2x\n", buttonData[0]);
        usleep(100000);
    }
}
#endif /* __APPLE__ */

#ifndef __APPLE__
void handleMIKE(void * /*userdata*/)
{
    //coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;
    while (1)
    {
#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        get_answer(1, buttonData);
        fprintf(stderr, "%2x\n", buttonData[0]);
    }
}
#endif /* __APPLE__ */

#ifndef __APPLE__
void handleDIVISION(void * /*userdata*/)
{
    //coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;
    while (1)
    {
#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        getDivisionAnswer(4, buttonData);
    }
}
#endif /* __APPLE__ */

#ifndef __APPLE__
void handleCEREAL(void *userdata)
{

    coVRTrackingSystems *bt = (coVRTrackingSystems *)userdata;
    int i;
    // Set digital output
    while (1)
    {
#ifndef _WIN32
        if (getppid() == 1)
        {
            //fprintf(stderr, "SERVER: exit\n");
            exit(1);
        }
#endif
        bt->bgdata.dout[0] = outData[0];
        bt->bgdata.dout[1] = outData[1];
        bt->bgdata.dout[2] = outData[2];
        send_outputs(&bt->bgdata);

        // wait for the timer
        //st = ms_time(&now);                   // why this ???
        //sigpause(SIGALRM);

        // get our input
        check_inputs(&bt->bgdata);

        // possibly 3 bytes of input data, handled in ints
        buttonData[0] = bt->bgdata.din[0];
        buttonData[1] = bt->bgdata.din[1];
        buttonData[2] = bt->bgdata.din[2];

        // add 8 analog data channels
        for (i = 0; i < 8; i++)
            analogData[i] = bt->bgdata.ain[i];
    }
}
#endif /* __APPLE */

coVRTrackingSystems::coVRTrackingSystems(int numStations, int stylusStation, int sensorStation, int worldXFormStation)
{
    tarsus = NULL;
    ssd = NULL;
    vrpn = NULL;
    dynasight = NULL;
    motion = NULL;
    fo = NULL;
    dtrack = NULL;
    cgvtrack = NULL;
    mousebuttons = NULL;
    tracker = NULL;
    fs = NULL;
    CaveLibTracker = NULL;
    buttonSystemPlugin = NULL;
    CaveLibWand = NULL;
    interpolationFile = NULL;

    buttonMask = coCoviseConfig::getInt("buttonMask", "COVER.Input.Mike", ~0x80);
#ifdef WIN32
    rawTarsusMouse = NULL;
#endif

    w_ni = 8;
    w_nj = 8;
    w_nk = 6;
    this->numStations = numStations;

    this->stylusStation = stylusStation;
    this->sensorStation = sensorStation;
    this->worldXFormStation = worldXFormStation;

    //int i;

    //TODO coConfig
    staticViewerX = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0f);
    staticViewerY = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -2000.0f);
    staticViewerZ = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 30.0f);

    fo = NULL;
    fs = NULL;
    tracker = NULL;

    trans_basis = NULL;
    x_coord = y_coord = z_coord = n1 = n2 = n3 = NULL;
    nx = ny = nz = ne = 0;
    interpolationFlag = false;
    orientInterpolationFlag = false;
    write_calibration_flag = false;
    dtrackWheel = 0;

    readConfigFile();

    //write calibration files
    if (write_calibration_flag == true)
    {
        const char calib_path_default[] = ".";
        const char *calib_path = getenv("COVISEDIR");
        if (NULL == calib_path)
        {
            calib_path = calib_path_default;
        }
        interpolationFlag = false;
        orientInterpolationFlag = false;
        orien_interp_files_flag = false;

        calib_pos_i[0] = -135;
        calib_pos_i[1] = -100;
        calib_pos_i[2] = -60;
        calib_pos_i[3] = -20;
        calib_pos_i[4] = 20;
        calib_pos_i[5] = 60;
        calib_pos_i[6] = 100;
        calib_pos_i[7] = 135;

        calib_pos_j[0] = -135;
        calib_pos_j[1] = -100;
        calib_pos_j[2] = -60;
        calib_pos_j[3] = -20;
        calib_pos_j[4] = 20;
        calib_pos_j[5] = 60;
        calib_pos_j[6] = 100;
        calib_pos_j[7] = 135;

        calib_pos_k[0] = -125;
        calib_pos_k[1] = -85;
        calib_pos_k[2] = -45;
        calib_pos_k[3] = -5;
        calib_pos_k[4] = 35;
        calib_pos_k[5] = 75;

        cout << "\nWrite 7 characters to diferentiate this file from others "
             << "for example the date in form 'apr2199'" << endl;
        cin >> end_file_name;
        sprintf(calib_name_x, "%s/calib_%s_ori_x.data", calib_path, end_file_name);
        sprintf(calib_name_y, "%s/calib_%s_ori_y.data", calib_path, end_file_name);
        sprintf(calib_name_z, "%s/calib_%s_ori_z.data", calib_path, end_file_name);
        sprintf(calib_name_p, "%s/calib_%s_posit.data", calib_path, end_file_name);

        calib_file_x.open(calib_name_x, ios::out);
        calib_file_y.open(calib_name_y, ios::out);
        calib_file_z.open(calib_name_z, ios::out);
        calib_file_p.open(calib_name_p, ios::out);

        //if ( calib_file_x && calib_file_y && calib_file_y && calib_file_p)
        {

            cout << "Following calibration files were opened to write:\n"
                 << calib_name_x << "\n"
                 << calib_name_y << "\n"
                 << calib_name_z << "\n"
                 << calib_name_p << endl;

            calib_file_x << w_ni << "\t" << w_nj << "\t" << w_nk << "\t" << 1 << endl;
            calib_file_y << w_ni << "\t" << w_nj << "\t" << w_nk << "\t" << 1 << endl;
            calib_file_z << w_ni << "\t" << w_nj << "\t" << w_nk << "\t" << 1 << endl;
            calib_file_p << w_ni << "\t" << w_nj << "\t" << w_nk << "\t" << 1 << endl;
        }
        /*  else
          {
          cout << " error by opening the calibration output files :/n"
          << calib_name_x << "\n"
          << calib_name_y << "\n"
          << calib_name_z << "\n"
          << calib_name_p << endl;

          write_calibration_flag=false;
          }*/
    }

    // read the interpolation file and create the interpolation arrays
    if (!interpolationFile)
        interpolationFlag = false;
    if (interpolationFlag)
    {
        interpolationFlag = readInterpolationFile(interpolationFile);
    }

    //determinate xyz_velocity
    if (interpolationFlag)
    {
        int xyz_velocity;
        xyz_velocity = find_xyz_velocity();
        switch (xyz_velocity)
        {
        case 123:
            interpolationFlag = true; //allready good
            break;

        case 213:
            reorganize_data(213);
            xyz_velocity = find_xyz_velocity();
            if (xyz_velocity == 123)
            {
                //cout << "data was reorganizated so that real z change faster "
                //        << "than real y and than real x" << endl;
                interpolationFlag = true;
            }
            else
            {
                //sprintf (interp_message,
                //        "Data reorganization for interpolation was not posible" );
                //cout << interp_message << endl;
                interpolationFlag = false;
            }
            break;

        default:
            //sprintf (interp_message,
            //        "Wrong interpolation data file,xyz_(changing)_velocity = %d is not supported by interpolation",xyz_velocity);
            //cout << interp_message << endl;
            interpolationFlag = false;
            break;
        }
    }

    // create a list tranformation matrix for the tracker orientation
    if (orientInterpolationFlag == true)
    {
        if (orien_interp_files_flag == true)
        {
            sprintf(interp_message, "orientation trans_basis will be read ");
            cout << interp_message << endl;
            read_trans_basis();
        };
        if (orien_interp_files_flag != true)
        {
            create_trans_basis();
            sprintf(interp_message, "trans_basis list was created ");
            cout << interp_message << endl;
        }
    }
    else
    {
        //sprintf (interp_message,"orientInterpolationFlag=false");
        //cout << interp_message << endl;
    }

    if (buttonSystem == B_MOUSE)
    {

        std::string buttonDev = coCoviseConfig::getEntry("COVER.Input.ButtonConfig.ButtonDevice");
        mousebuttons = new MouseButtons(buttonDev.c_str());
#ifdef WIN32
        rawTarsusMouse = new coRawMouse(buttonDev.c_str());
#endif
    }
    int key;
#ifndef WIN32
    int tracker_shmid;
#endif
    if (buttonSystem == B_VRC)
    {
        int port = 7777;
        int debugLevel = 0;
        float scale; // factor to make cm from "unit"
        port = coCoviseConfig::getInt("COVER.Input.VRC.Port", 7777);
        debugLevel = coCoviseConfig::getInt("COVER.Input.VRC.DebugLevel", 0);
        std::string unit = coCoviseConfig::getEntry("COVER.Input.VRC.Unit");
        std::string options = coCoviseConfig::getEntry("COVER.Input.VRC.Options");
        if (unit.empty())
            scale = 1.0; // assume tracker sends cm
        else if (0 == strncasecmp(unit.c_str(), "cm", 2))
            scale = 1.0;
        else if (0 == strncasecmp(unit.c_str(), "mm", 2))
            scale = 10.0;
        else if (0 == strncasecmp(unit.c_str(), "inch", 4))
            scale = 1.0 / 2.54;
        else
        {
            if (sscanf(unit.c_str(), "%f", &scale) != 1)
            {
                cerr << "VRTracker::VRTracker:: sscanf failed" << endl;
            }
        }

        vrcTracker = new VRCTracker(port, debugLevel, scale, options.c_str());

        if (vrcTracker->isOk())
            vrcTracker->mainLoop();
        else
        {
            cerr << "could not create VRC Tracking Server - exiting"
                 << endl;
            exit(-1);
        }
    }
    else
#ifndef __APPLE__
        if (buttonSystem == B_PINCH)
    {
// get shared memory segment for button Data
#ifdef _WIN32
        buttonData = new unsigned char[4];
#else
        int shmid;
        key_t shmkey = SHMKEY;

        while ((shmid = shmget(shmkey, 4, PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }
        else if (ret == 0) // child process
        {
            // read serial port and write data to shared memory

            cout << "INFO: PINCH button server forked" << endl;

#ifdef __sgi
            prctl(PR_TERMCHILD); // Exit when parent does
#endif
            signal(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
#endif
        pinchfd = openPinch(buttonDevice, 9600);
        if (pinchfd < 0)
        {
            cout << "Could not opened Pinch Glove on " << buttonDevice << endl;
            return;
        }
        else
        {
            cout << "Opened Pinch Glove on " << buttonDevice << endl;
        }

#ifdef _WIN32
        //Not used: "uintptr_t thread = "
        _beginthread(processPinch, 0, this);
#else

            processPinch(this);
        }
#endif
    }

    if (buttonSystem == B_VP)
    {
// get shared memory segment for button Data

#ifdef _WIN32
        if (!Init(buttonDevice, 9600))
        {
            fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
            exit(0);
        }
        buttonData = new unsigned char[4];
        _beginthread(handleVP, 0, this);
#else
        int shmid;
        key_t shmkey = SHMKEY;

        while ((shmid = shmget(shmkey, 4, PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }
        else if (ret == 0) // child process
        {
            // read serial port and write data to shared memory

            cout << "INFO: Virtual Presence button server forked" << endl;

#ifdef __sgi
            prctl(PR_TERMCHILD); // Exit when parent does
#endif
            signal(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
            if (!Init(buttonDevice, 9600))
            {
                fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
                exit(0);
            }
            int i = 0;
            unsigned char inputData[10];
            while (1)
            {
                inputData[0] = 0;
                get_answer(1, (unsigned char *)inputData);
                if (!(~inputData[0] & 0xc0))
                    break;
                if (i > 5)
                {
                    fprintf(stderr, "error no valid button data from port %s !!\n", buttonDevice);
                    exit(0);
                }
                i++;
            }
            while (1)
            {
                handleVP(this);
            }
        }
#endif
    }
#endif /* __APPLE__ */
#ifndef __APPLE__
    if (buttonSystem == B_CYBER)
    {
#ifdef _WIN32
        if (!Init(buttonDevice, 9600))
        {
            fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
            exit(0);
        }
        buttonData = new unsigned char[16];
        _beginthread(handleCYBER, 0, this);
#else
        rawButton_micha = 0;

        // get shared memory segment for button Data

        int shmid;
        key_t shmkey = SHMKEY;

        while ((shmid = shmget(shmkey, 16, PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }
        else if (ret == 0) // child process
        {
            // read serial port and write data to shared memory

            cout << "INFO: Cyberstick button server forked" << endl;

#ifdef __sgi
            prctl(PR_TERMCHILD); // Exit when parent does
#endif
            signal(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
            if (!Init(buttonDevice, 9600))
            {
                fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
                exit(0);
            }

            handleCYBER(this);
        }
#endif
    }
#endif /* __APPLE */
#ifndef __APPLE__
    if (buttonSystem == B_HORNET)
    {
#ifdef _WIN32
        if (!Init(buttonDevice, 19200))
        {
            fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
            exit(0);
        }
        buttonData = new unsigned char[5];
        _beginthread(handleHORNET, 0, this);
#else
        // get shared memory segment for button Data

        int shmid;
        key_t shmkey = SHMKEY;

        while ((shmid = shmget(shmkey, 5, PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }
        else if (ret == 0) // child process
        {
            // read serial port and write data to shared memory

            cout << "INFO: HORNET button server forked" << endl;

#ifdef __sgi
            prctl(PR_TERMCHILD); // Exit when parent does
#endif
            signal(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
            if (!Init(buttonDevice, 19200))
            {
                fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
                exit(0);
            }

            char command[5];
            command[0] = 'd';
            send_command(command, 1);
            get_answer(5, buttonData);
            strcpy(command, "D00");
            send_command(command, 3);
            strcpy(command, "O7F");
            send_command(command, 3);
            get_answer(5, buttonData);
            strcpy(command, "D00");
            command[0] = 'i';
            handleHORNET(this);
        }
#endif
    }
#endif /* __APPLE__ */

#ifdef __APPLE__
    if (false)
    {
    }
#else
        if (buttonSystem == B_MIKE)
    {
#ifdef _WIN32
        if (!Init(buttonDevice, 9600))
        {
            fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
            exit(0);
        }
        buttonData = new unsigned char[5];
        _beginthread(handleMIKE, 0, this);
#else
        // get shared memory segment for button Data

        int shmid;
        key_t shmkey = SHMKEY;

        while ((shmid = shmget(shmkey, 4, PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }
        else if (ret == 0) // child process
        {
            // read serial port and write data to shared memory

            cout << "INFO: MIKE button server forked" << endl;

#ifdef __sgi
            prctl(PR_TERMCHILD); // Exit when parent does
#endif
            signal(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
            if (!Init(buttonDevice, 9600))
            {
                fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
                exit(0);
            }

            handleMIKE(this);
        }
#endif
    }
#endif /* __APPLE__ */
#ifndef __APPLE__
    else if (buttonSystem == B_CEREAL)
    {
#ifdef _WIN32
        buttonData = new unsigned char[5];
        _beginthread(handleCEREAL, 0, this);
#else
        // get shared memory segment for button Data

        int shmid;
        key_t shmkey = SHMKEY;

        // AWe: read all 8 analog interfacese
        //      round up to 16 chars for secure alignment
        while ((shmid = shmget(shmkey, 16 + 8 * sizeof(float),
                               PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        analogData = (float *)&buttonData[16];

        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }

        // this is the button server, it will never leave this block
        else if (ret == 0) // child process
        {
            memset(&bgdata, 0, sizeof(bgdata));

            // read serial port and write data to shared memory
            cout << "INFO: CEREAL button server forked"
                 << ", pid=" << getpid() << endl;

#ifdef __sgi
            prctl(PR_TERMCHILD); // Exit when parent does
#endif
            signal(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD

            // ---- AWe: read config file for CEREAL box configuration -----

            // store data values in own field, might be overwritten otherwise

            // default values for CerealConfig.     D1  D2  D3
            static const char *defaultCereal[] = { "IN", "IN", "OUT 0x1" };
            static const int inFlags[] = { DIC1, DIC2, DIC3 };
            static const int outFlags[] = { DOC1, DOC2, DOC3 };

            // some buffers...
            char varname[32];
            char buffer[256];

            // loop: names IO1,IO2,IO3, but fields 0..2
            int i;
            bgdata.dig_in = 0;
            bgdata.dig_out = 0;
            for (i = 0; i < 3; i++)
            {
                sprintf(varname, "COVER.Input.CerealConfig.IO:%d", i + 1);
                string config = coCoviseConfig::getEntry(varname);
                if (config.empty())
                    config = defaultCereal[i];

                // OUT flag: read value
                if (strstr(config.c_str(), "OUT") || strstr(config.c_str(), "out"))
                {
                    bgdata.dig_out |= outFlags[i];
                    if (sscanf(config.c_str(), "%s %s", varname, buffer) != 2)
                    {
                        cerr << "coVRTrackingSystems::coVRTrackingSystems: sscanf2 failed" << endl;
                    }
                    // allow HEX input
                    outData[i] = strtol(buffer, NULL, 0);
                }

                // IN flag: no values
                else
                {
                    bgdata.dig_in |= inFlags[i];
                }
            }

            /// Read all analog inputs
            bgdata.analog_in = AIC1 | AIC2 | AIC3 | AIC4 | AIC5 | AIC6 | AIC7 | AIC8;

            /// Do not use analog output
            bgdata.analog_out = 0;
            bgdata.dout[0] = 0x0;
            bgdata.dout[1] = 0x0;
            bgdata.dout[2] = 0x0;

            /*
          *  Set the baud rate
          */
            bgdata.baud = BAUD192;

            int st = open_lv(&bgdata, buttonDevice, FB_NOBLOCK);
            if (st < 0)
            {
                fprintf(stderr, "error connecting to CerealBox on %s !!\n", buttonDevice);
                exit(0);
            }

            st = init_lv(&bgdata);
            if (st < 0)
            {
                fprintf(stderr, "error initializing CerealBox on %s !!\n", buttonDevice);
                exit(0);
            }

            handleCEREAL(this);
        }
#endif
    }
#endif /* __APPLE */
    else if (buttonSystem == B_CAVELIB)
    {
        XValuator = coCoviseConfig::getInt("COVER.Input.CaveLibConfig.XValuator", 0);
        YValuator = coCoviseConfig::getInt("COVER.Input.CaveLibConfig.YValuator", 1);
        std::string unit = coCoviseConfig::getEntry("value", "COVER.Input.CaveLibConfig.Unit", "feet");
        if (unit == "feet")
            scaleFactor = 304.8;
        else if (unit == "mm")
            scaleFactor = 1.0;
        else if (unit == "m")
            scaleFactor = 1000.0;
        else if (unit == "cm")
            scaleFactor = 10.0;
        else if (unit == "dm")
            scaleFactor = 100.0;
        else if (unit == "inch")
            scaleFactor = 25.4;
        else
        {
            scaleFactor = 1.0;
            sscanf(unit.c_str(), "%f", &scaleFactor);
        }
        Yup = coCoviseConfig::isOn("COVER.Input.CaveLibConfig.Yup", true);
        fprintf(stderr, " \n\n\n\n Unit: %s factor: %f\n\n\n\n\n", unit.c_str(), scaleFactor);
        CaveLibWandController = coCoviseConfig::getInt("COVER.Input.CaveLibConfig.Controller", 0);
        key = coCoviseConfig::getInt("COVER.Input.CaveLibConfig.WandSHMID", 4127);
#ifdef WIN32
        //HANDLE handle;
        HANDLE filemap;
        char tmp_str[512];
        /*     sprintf(tmp_str,"%s\\%d",getenv("tmp"), key);
   while((handle = CreateFile(tmp_str, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
      NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
   {
      cerr << "CreateFile file " << key << " does not exist\n";
      cerr << "waiting\n";
      Sleep(1000);
   } 
   filemap = CreateFileMapping(handle, NULL, PAGE_READWRITE, 0, sizeof(struct TRACKD_WAND), NULL);*/

        sprintf(tmp_str, "Global\\%d", key);

        filemap = OpenFileMapping(FILE_MAP_READ, FALSE, tmp_str);
        if (!(CaveLibWand = (TRACKD_WAND *)MapViewOfFile(filemap, FILE_MAP_READ, 0, 0, sizeof(struct TRACKD_WAND))))
        {
            fprintf(stderr, "Could not attach shared memory key %x = %d for Cavelib WAND\n", key, key);
            //exit(1);
            buttonSystem = B_NONE;
        }
#else
        tracker_shmid = shmget(key, sizeof(struct TRACKD_WAND),
                               PERMS | IPC_CREAT);
        if (tracker_shmid < 0)
        {
            fprintf(stderr, "Could access shared memory key %x = %d for Cavelib WAND\n", key, key);
            //exit(1);
            buttonSystem = B_NONE;
        }
        CaveLibWand = (struct TRACKD_WAND *)shmat(tracker_shmid, (char *)0, 0);

#endif
        if (CaveLibWand == (struct TRACKD_WAND *)-1)
        {
            fprintf(stderr, "Could attach shared memory key %x = %d for Cavelib WAND\n", key, key);
            //exit(1);
            buttonSystem = B_NONE;
        }
        long wandsize = (long)sizeof(struct TRACKD_WAND);
        printf("wand %ld\n", wandsize);
        //printf("CAVE_CONTROLLER_ST %d\n",sizeof(struct CAVE_CONTROLLER_ST));
        //printf("CAVE_TRACKDTRACKER_HEADER %d\n",sizeof(struct CAVE_TRACKDTRACKER_HEADER));
    }
#ifndef __APPLE__
    else if (buttonSystem == B_DIVISION)
    {
#ifdef _WIN32
        buttonData = new unsigned char[5];
        _beginthread(handleDIVISION, 0, this);
#else
        // get shared memory segment for button Data

        int shmid;
        key_t shmkey = SHMKEY;

        while ((shmid = shmget(shmkey, 10, PERMS | IPC_CREAT)) < 0)
        {
            cout << "shmget failed" << endl;
            shmkey++;
        }
        buttonData = (unsigned char *)shmat(shmid, (char *)0, 0);
        // fork button process
        int ret = fork();
        if (ret == -1)
        {
            cout << "fork failed" << endl;
        }
        else if (ret == 0) // child process
        {
            // read serial port and write data to shared memory

            //cout << "INFO: button server forked" << endl;
            if (!Init(buttonDevice, 9600))
            {
                fprintf(stderr, "error connecting to %s !!\n", buttonDevice);
                exit(0);
            }

            handleDIVISION(this);
        }
#endif
    }
#endif /* __APPLE__ */
    switch (trackingSystem)
    {
    case coVRTrackingUtil::T_DTRACK:
    {
        int port = coCoviseConfig::getInt("COVER.Input.DTrack.Port", 5000);
        std::string Startup = coCoviseConfig::getEntry("COVER.Input.DTrack.Startup"); // IP:5001
        dtrack = new DTrack(port, Startup.c_str());
    }
    break;
    case coVRTrackingUtil::T_CGVTRACK:
    {
        int port = coCoviseConfig::getInt("COVER.Input.CGVTrack.Port", 5000);
        std::string host = coCoviseConfig::getEntry("COVER.Input.CGVTrack.Host");
        cgvtrack = new CGVTrack(host.c_str(), port);
    }
    break;

    case coVRTrackingUtil::T_SEEREAL:
    {
#ifdef WIN32
        AWindow = CreateWindow("", "", WS_POPUP, 0, 0, 0, 0, 0, 0, 0, 0);
        //
        if (!initHeadFinder(AWindow)) //Check if someone else is using the head finder
        {
            cerr << "could not init SeeReal connection" << endl;
        }
        else
        {
            GetDC(0); //For convenience we use just the screen; I checked it with NT 4.0
        }

#endif
    }
    break;

    case coVRTrackingUtil::T_TARSUS:
    {
        int port = coCoviseConfig::getInt("port", "COVER.Input.Tarsus.Server", 800);
        string host = coCoviseConfig::getEntry("host", "COVER.Input.Tarsus.Server");
        string buttonDev = coCoviseConfig::getEntry("COVER.Input.Tarsus.ButtonDevice");
        if (mousebuttons == NULL)
        {
            cerr << "Tarsus: button device=" << buttonDev << endl;
            mousebuttons = new MouseButtons(buttonDev.c_str());
        }
#ifdef WIN32
        if (rawTarsusMouse == NULL)
            rawTarsusMouse = new coRawMouse(buttonDev.c_str());
#endif
        cerr << "Connecting to Tarsus tracking daemon on " << host << ":" << port << flush;
        tarsus = new Tarsus(port, host.c_str());
        cerr << "." << endl;
        for (int i = 0; i <= numStations; i++)
        {
            char key[1024];
            sprintf(key, "COVER.Input.Tarsus.StationName:%d", i);
            std::string name = coCoviseConfig::getEntry(key);
            if (!name.empty())
            {
                tarsus->setStationName(i, name.c_str());
            }
        }
    }
    break;
    case coVRTrackingUtil::T_SSD:
    {
        string host = coCoviseConfig::getEntry("COVER.Input.SSD.Host");
        ssd = new PvrSSD(host.c_str());
        string buttonDev = coCoviseConfig::getEntry("COVER.Input.SSD.ButtonDevice");
        if (mousebuttons == NULL)
        {
            if (!buttonDev.empty())
                cerr << "SSD: button device=" << buttonDev << endl;
            mousebuttons = new MouseButtons(buttonDev.c_str());
        }
#ifdef WIN32
        if (rawTarsusMouse == NULL)
            rawTarsusMouse = new coRawMouse(buttonDev.c_str());
#endif
    }
    break;
    case coVRTrackingUtil::T_VRPN:
    {
        string host = coCoviseConfig::getEntry("host", "COVER.Input.VRPN");
        string dev = coCoviseConfig::getEntry("device", "COVER.Input.VRPN");
        string buttonDev = coCoviseConfig::getEntry("COVER.Input.VRPN.ButtonDevice");
        vrpn = new VRPN(host, dev);

        if (mousebuttons == NULL)
        {
            if (!buttonDev.empty())
                cerr << "VRPN: button device=" << buttonDev << endl;
            mousebuttons = new MouseButtons(buttonDev.c_str());
        }
#ifdef WIN32
        if (rawTarsusMouse == NULL)
            rawTarsusMouse = new coRawMouse(buttonDev.c_str());
#endif
    }
    break;
    case coVRTrackingUtil::T_DYNASIGHT:
    {
        string serport = coCoviseConfig::getEntry("serial", "COVER.Input.DynaSight");
        if (serport.empty())
            serport = DEFAULTSERIAL;
        dynasight = new DynaSight(serport);

        string buttonDev = coCoviseConfig::getEntry("COVER.Input.VRPN.ButtonDevice");
        if (mousebuttons == NULL)
        {
            if (!buttonDev.empty())
                cerr << "DynaSight: button device=" << buttonDev << endl;
            mousebuttons = new MouseButtons(buttonDev.c_str());
        }
#ifdef WIN32
        if (rawTarsusMouse == NULL)
            rawTarsusMouse = new coRawMouse(buttonDev.c_str());
#endif
    }
    break;
#ifndef __APPLE__
    case coVRTrackingUtil::T_POLHEMUS:
    {
        int inputDev;
        std::string inputDevLine = coCoviseConfig::getEntry("COVER.Input.PolhemusConfig.InputDevice");
        if (!inputDevLine.empty())
        {
            if (strcasecmp(inputDevLine.c_str(), "stylus") == 0)
                inputDev = fastrak::BUTTONDEVICE_STYLUS;
            else if (strcasecmp(inputDevLine.c_str(), "wand") == 0)
                inputDev = fastrak::BUTTONDEVICE_WAND;
            else
            {
                fprintf(stderr, "WARNING: PolhemusConfig.INPUT_DEVICE [%s] not supported. Setting input device to stylus\n", inputDevLine.c_str());
                inputDev = fastrak::BUTTONDEVICE_STYLUS;
            }
        }
        else
        {
            fprintf(stderr, "WARNING: PolhemusConfig.INPUT_DEVICE missing in covise.config.\n");
            fprintf(stderr, "         Setting input device to stylus\n");
            inputDev = fastrak::BUTTONDEVICE_STYLUS;
        }
        fs = new fastrak(portname, baudrate, numStations, inputDev);
        break;
    }
    // Configure MotionStar tracker: extracted from birdTracker
    case coVRTrackingUtil::T_MOTIONSTAR:
    {
        std::string ip = coCoviseConfig::getEntry("value", "COVER.Input.MotionstarConfig.IPAddress", "141.58.8.125");
        string numR = coCoviseConfig::getEntry("COVER.Input.MotionstarConfig.NumReceivers");
        string bios = coCoviseConfig::getEntry("COVER.Input.MotionstarConfig.BIOS");
        bool debugOutput = coCoviseConfig::isOn("COVER.Input.MotionstarConfig.Debug", false);
        bool debugOutputAll = coCoviseConfig::isOn("COVER.Input.MotionstarConfig.DebugAll", false);
        int handAddr = coCoviseConfig::getInt("COVER.Input.HandAddress", 0);

        tracker = new birdTracker(ip.c_str(), handAddr, numR.c_str(), bios.c_str(), debugOutput, debugOutputAll);
    }
    break;

    case coVRTrackingUtil::T_FOB:
        fo = new fob(portname, baudrate, numStations, coCoviseConfig::isOn("COVER.Input.FobConfig.StreamMode", true));
        break;

    case coVRTrackingUtil::T_CAVELIB:
        key = coCoviseConfig::getInt("COVER.Input.CaveLibConfig.TrackerSHMID", 4126);
#ifdef WIN32
        //HANDLE handle;
        HANDLE filemap;
        char tmp_str[512];
        sprintf(tmp_str, "Global\\%d", key);
        /* while((handle = CreateFile(tmp_str, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
      NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
   {
      cerr << "CreateFile file " << key << " does not exist\n";
      cerr << "waiting\n";
      Sleep(1000);
   } 
   filemap = CreateFileMapping(handle, NULL, PAGE_READWRITE, 0, sizeof(struct TRACKD_TRACKING), NULL);*/

        filemap = OpenFileMapping(FILE_MAP_READ, FALSE, tmp_str);
        if (!(CaveLibTracker = (struct TRACKD_TRACKING *)MapViewOfFile(filemap, FILE_MAP_READ, 0, 0, sizeof(struct TRACKD_TRACKING))))
        {
            fprintf(stderr, "Could not attach shared memory key %x = %d for Cavelib TRACKER\n", key, key);
            //exit(1);
            buttonSystem = B_NONE;
        }
#else
        tracker_shmid = shmget(key, sizeof(struct TRACKD_TRACKING),
                               PERMS | IPC_CREAT);
        if (tracker_shmid < 0)
        {
            fprintf(stderr, "Could get shared memory key %x = %d for Cavelib TRACKER\n", key, key);

            exit(1);
        }
        CaveLibTracker = (struct TRACKD_TRACKING *)shmat(tracker_shmid, (char *)0, 0);
        if (CaveLibTracker == (struct TRACKD_TRACKING *)-1)
        {
            fprintf(stderr, "Could attach shared memory key %x = %d for Cavelib TRACKER\n", key, key);
            exit(1);
        }
#endif

        break;

    default:
        break;
#endif /* __APPLE__ */
    }
}

coVRTrackingSystems::~coVRTrackingSystems()
{
#ifndef __APPLE__
    delete fs;
#ifndef _WIN32
    //delete tracker;
    delete fo;
#endif
#endif /* __APPLE__ */
    delete tarsus;
    delete ssd;
    delete vrpn;
    delete dynasight;
    delete mousebuttons;
    delete dtrack;
    delete cgvtrack;
#ifdef WIN32
    delete rawTarsusMouse;
#endif

    if (x_coord != NULL)
    {
        delete[] x_coord;
    }
    if (y_coord != NULL)
    {
        delete[] y_coord;
    }
    if (z_coord != NULL)
    {
        delete[] z_coord;
    }

    if (n1 != NULL)
    {
        delete[] n1;
    }
    if (n2 != NULL)
    {
        delete[] n2;
    }
    if (n3 != NULL)
    {
        delete[] n3;
    }

    //for the tracker direction
    if (trans_basis != NULL)
    {
        for (int counter = 0; counter < nx * ny * nz; counter++)
        {
            for (int vector = 0; vector < 3; vector++)
            {
                delete[] trans_basis[counter][vector];
            }
            delete[] trans_basis[counter];
        }
        delete[] trans_basis;
    }
#ifdef _WIN32
    if (trackingSystem == coVRTrackingUtil::T_SEEREAL)
    {
        closeHeadFinder(AWindow);
    }
#endif

    /*if (calib_file_x)*/
    {
        calib_file_x.close();
    }
    /*if (calib_file_y)*/
    {
        calib_file_y.close();
    }
    /*if (calib_file_z)*/
    {
        calib_file_z.close();
    }
    /*if (calib_file_p)*/
    {
        calib_file_p.close();
    }
}

void
coVRTrackingSystems::reset()
{
    switch (trackingSystem)
    {
#ifndef __APPLE__
    case coVRTrackingUtil::T_POLHEMUS:
        fs->reset();
        break;
#endif
    case coVRTrackingUtil::T_TARSUS:
        tarsus->reset();
        break;
    case coVRTrackingUtil::T_SSD:
        ssd->reset();
        break;
    case coVRTrackingUtil::T_VRPN:
        vrpn->reset();
        break;
    case coVRTrackingUtil::T_DYNASIGHT:
        dynasight->reset();
        break;
    default:
        fprintf(stderr, "Reset not yet implemented for this Tracking system\n");
    }
}

//
// new: no calibration
//
void
coVRTrackingSystems::config()
{
#ifndef __APPLE__
    int r, d, num, i;
    std::string line;
#endif
    switch (trackingSystem)
    {
#ifndef __APPLE__
    case coVRTrackingUtil::T_POLHEMUS:
        // test
        if (!fs->testConnection())
        {
            printf("ERROR: polhemus tracker not connected to port %s\n", portname);
            trackingSystem = 0;
        }
        //TODO coConfig
        line = coCoviseConfig::getEntry("COVER.Input.PolhemusConfig.Hemisphere");
        if (!line.empty())
        {
            if (sscanf(line.c_str(), "%f %f %f", &hx, &hy, &hz) != 3)
            {
                cerr << "coVRTrackingSystems::config: sscanf1 failed" << endl;
            }
        }

        else
        {
            hx = 1;
            hy = 0;
            hz = 0;
        }
        // configure
        //fs->setPositionFilter(0.05, 0.2, 0.8, 0.8);
        //fs->setAttitudeFilter(0.05, 0.2, 0.8, 0.8);
        for (i = 0; i < numStations; i++)
            fs->setStation(i + 1);

        if (stylusStation > 0)
        {
            fs->setHemisphere(stylusStation, hx, hy, hz);
            fs->setStation(stylusStation);
            fs->setStylusMouseMode();
        }
        if (sensorStation > 0)
        {
            fs->setHemisphere(sensorStation, hx, hy, hz);
            fs->setStation(sensorStation);
        }

        // fork server process
        fs->start();
        break;
    case coVRTrackingUtil::T_MOTIONSTAR:
    {
        r = tracker->init();
        //fprintf( stderr, "init: %d\n", (r) );

        if (r == -1)
        {
            //delete tracker;
            fprintf(stderr, "setup failed\n");
            trackingSystem = 0;
            break;
        }

        if (coCoviseConfig::isOn("COVER.Input.MotionstarConfig.DualTransmitter", false))
            tracker->DualTransmitter(1);
        string mbs = coCoviseConfig::getEntry("COVER.Input.MotionstarConfig.MotionstarButtonSystem");

        int mButtonSystem = B_NONE;
        if (!mbs.empty())
        {
            if (!(strcmp(mbs.c_str(), "MIKE")))
            {
                mButtonSystem = B_MIKE;
            }
            else if (!(strcmp(mbs.c_str(), "HORNET")))
            {
                mButtonSystem = B_HORNET;
            }
            else if (!(strcmp(mbs.c_str(), "CYBER")))
            {
                mButtonSystem = B_CYBER;
            }
            else if (!(strcmp(mbs.c_str(), "VIRTUAL_PRESENCE")))
            {
                mButtonSystem = B_VP;
            }
            else if (!(strcmp(mbs.c_str(), "PINCH")))
            {
                mButtonSystem = B_PINCH;
            }
            else if (!(strcmp(mbs.c_str(), "CEREAL")))
            {
                mButtonSystem = B_CEREAL;
            }
            else if (!(strcmp(mbs.c_str(), "CAVELIB")))
            {
                mButtonSystem = B_CAVELIB;
            }
            else if (!(strcmp(mbs.c_str(), "DIVISION")))
            {
                mButtonSystem = B_DIVISION;
            }
        }
        tracker->setButtonSystem(mButtonSystem);
        birdTracker::hemisphere hemisphere = birdTracker::FRONT_HEMISPHERE;
        std::string hemch = coCoviseConfig::getEntry("COVER.Input.MotionstarConfig.Hemisphere");
        if (!hemch.empty())
        {
            if (strncasecmp(hemch.c_str(), "FRONT", 5) == 0)
            {
                hemisphere = birdTracker::FRONT_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "REAR", 4) == 0)
            {
                hemisphere = birdTracker::REAR_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "UPPER", 5) == 0)
            {
                hemisphere = birdTracker::UPPER_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "LOWER", 5) == 0)
            {
                hemisphere = birdTracker::LOWER_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "LEFT", 4) == 0)
            {
                hemisphere = birdTracker::LEFT_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "RIGHT", 5) == 0)
            {
                hemisphere = birdTracker::RIGHT_HEMISPHERE;
            }
        }

        birdTracker::dataformat angleMode = birdTracker::FLOCK_POSITIONMATRIX;
        std::string angch = coCoviseConfig::getEntry("COVER.Input.MotionstarConfig.AngleMode");
        if (!angch.empty())
        {
            if (strncasecmp(angch.c_str(), "Euler", 5) == 0)
            {
                angleMode = birdTracker::FLOCK_POSITIONANGLES;
            }
            fprintf(stderr, "AngleMode: %d\n", angleMode);
        }

        int samplingRate = coCoviseConfig::getInt("COVER.Input.MotionstarConfig.SamplingRate", 80);

        if (tracker->setup(hemisphere, angleMode, samplingRate) < 0)
        {
            //delete tracker;
            fprintf(stderr, "hemisphere setup failed\n");
            trackingSystem = 0;
            break;
        }

        num = tracker->getNumReceivers();

        fprintf(stderr, "number of receivers: %d\n", num);

        // setup filters and buttons
        for (d = 0; d < num; d++)
        {
            i = tracker->hasButtons(d);
            //fprintf(stderr, "hasButtons: %d\n", i);
            //#ifndef OLD_MOTIONSTAR_BIOS
            std::string bios_version = coCoviseConfig::getEntry("COVER.Input.MotionstarConfig.BIOS");
            if (strcasecmp(bios_version.c_str(), "OLD") == 0)
            {
            }
            else
            {
                if (i)
                    tracker->setFilter(d, 1, 1, 1, 1);
                else
                    tracker->setFilter(d, 1, 1, 1, 1);
            }
            // #endif
        }
        //tracker->singleShot();  // singleShot-Mode doesn't work !!!
        tracker->runContinuous();
    }
    break;

#ifndef _WIN32
    case coVRTrackingUtil::T_FOB:
    {
        if (!fo->testConnection())
        {
            fprintf(stderr, "serial connection to flock failed\n");
            return;
        }
        fo->autoconfig();
        //fo->stopStreaming();
        fo->printSystemStatus();
        fprintf(stderr, "found %d ERCs\n", fo->getNumERCs());
        fprintf(stderr, "found %d receivers\n", fo->getNumReceivers());
        int range = 0;
        if (coCoviseConfig::isOn("COVER.Input.FobConfig.Fullrange", false))
            range = 1;
        fob::hemisphere hemisphere = fob::FRONT_HEMISPHERE;
        std::string hemch = coCoviseConfig::getEntry("COVER.Input.FobConfig.Hemisphere");
        if (!hemch.empty())
        {
            if (strncasecmp(hemch.c_str(), "FRONT", 5) == 0)
            {
                hemisphere = fob::FRONT_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "REAR", 4) == 0)
            {
                hemisphere = fob::REAR_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "UPPER", 5) == 0)
            {
                hemisphere = fob::UPPER_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "LOWER", 5) == 0)
            {
                hemisphere = fob::LOWER_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "LEFT", 4) == 0)
            {
                hemisphere = fob::LEFT_HEMISPHERE;
            }
            else if (strncasecmp(hemch.c_str(), "RIGHT", 5) == 0)
            {
                hemisphere = fob::RIGHT_HEMISPHERE;
            }
        }

        // assumption: one ore zero ERCs
        for (i = fo->getNumERCs() + 1; i < numStations + 1; i++)
        {
            fo->setDataFormat(i, fob::FLOCK_POSITIONMATRIX);
            fo->setHemisphere(i, hemisphere);
            if (fo->getNumERCs() == 0)
            {
                fo->changeRange(i, range);
            }
        }

        if (coCoviseConfig::isOn("COVER.Input.FobConfig.StreamMode", true))
            fo->enableStreamMode();
        else
            //POINT mode - initially sends the group mode command only
            fo->sendGroupMode();

        fo->startServerProcess();
    }
    break;
#endif
#endif /* __APPLE__ */
    default:
        break;
    }
}

void
coVRTrackingSystems::getRotationMatrix(osg::Matrix &)
{
    //(rotMat);
}

void
coVRTrackingSystems::getTranslationMatrix(osg::Matrix &)
{

    //(transMat);
}

//
// new transmitter/sensor offset matrix instead of reference frame
//
void
coVRTrackingSystems::getMatrix(int station, osg::Matrix &mat)
{
#ifndef __APPLE__
    float w, q1, q2, q3;
    float h, p, r;
    float phi;
#endif
    float x = 0, y = 0, z = 0; // raw tracker pos
    //float                tp_off[3];                        // tracker pos with offset
    //float                 tp_interp[3];                // corrected tracker pos
    osg::Vec3 n, pos;
    //osg::Vec3                lmfc;                    // linear magnetic field correction
    static int firsttime = 1;
    static int debugPos = 0;
    static int debugStation = 0;
    const int DEBUG_RAW = 1;
    const int DEBUG_FILTER = 2;
    float m[9];
    osg::Matrix DeviceOffset;
    if (station == sensorStation)
    {
        DeviceOffset = VRTracker::instance()->trackingUtil->computeDeviceOffsetMat(coVRTrackingUtil::headDev);
    }
    else if (station == stylusStation)
    {
        DeviceOffset = VRTracker::instance()->trackingUtil->computeDeviceOffsetMat(coVRTrackingUtil::handDev);
    }
    else if (station == worldXFormStation)
    {
        DeviceOffset = VRTracker::instance()->trackingUtil->computeDeviceOffsetMat(coVRTrackingUtil::worldDev);
    }

    if (firsttime)
    {
        debugPos = 0;
        std::string entry = coCoviseConfig::getEntry("COVER.Input.DebugTracking");
        if (!entry.empty())
        {
            if (!(strcasecmp(entry.c_str(), "RAW")))
                debugPos = DEBUG_RAW;
            else if (!(strcasecmp(entry.c_str(), "FILTER")))
                debugPos = DEBUG_FILTER;
            // APP is handled in VRTracker
        }
        debugStation = coCoviseConfig::getInt("COVER.Input.DebugStation", 0);
        firsttime = 0;
    }

    switch (trackingSystem)
    {

    case coVRTrackingUtil::T_PLUGIN:
    {
        coVRTrackingUtil::instance()->getTrackingSystemPlugin()->getMatrix(station, mat);
        osg::Vec3 pos;
        pos = mat.getTrans();
        x = pos[0];
        y = pos[1];
        z = pos[2];
    }
    break;
    case coVRTrackingUtil::T_CGVTRACK:

        mat.makeIdentity();
        cgvtrack->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        mat.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
        if ((x == 0.0) && (y == 0.0) && (z == 0.0))
        {
            x = staticViewerX;
            y = staticViewerY;
            z = staticViewerZ;
        }
        // cerr << "T_CGVTRACK: Station=" << station << endl;
        break;
    case coVRTrackingUtil::T_DTRACK:
        mat.makeIdentity();
        if (dtrack->gotData())
        {
            dtrack->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
            mat.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
        }
        else
        {
            static double oldTime = 0.0;
            if ((cover->frameRealTime() - oldTime) > 1.0)
            {
                if (oldTime != 0.0)
                {
                    dtrack->sendStart();
                }
                oldTime = cover->frameRealTime();
            }
        }
        if ((x == 0.0) && (y == 0.0) && (z == 0.0))
        {
            x = staticViewerX;
            y = staticViewerY;
            z = staticViewerZ;
        }
        // cerr << "T_DTRACK: Station=" << station << endl;
        break;

#ifdef _WIN32
    case coVRTrackingUtil::T_SEEREAL:
    {
        mat.makeIdentity();
        int xi = 0, yi = 0, zi = 0;
        getViewerPosition(xi, yi, zi);

        x = xi / 20.0f; // now in mm
        y = yi / 20.0f; // now in mm
        z = zi / 20.0f; // now in mm
        if ((xi == 0) && (yi == 0) && (zi == 0))
        {
            x = staticViewerX;
            y = staticViewerY;
            z = staticViewerZ;
        }
    }
    break;
#endif
    case coVRTrackingUtil::T_TARSUS:
    case coVRTrackingUtil::T_SSD:
    case coVRTrackingUtil::T_VRPN:
    case coVRTrackingUtil::T_DYNASIGHT:

        mat.makeIdentity();
        if (tarsus)
        {
            tarsus->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        }
        else if (ssd)
        {
            ssd->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        }
        else if (vrpn)
        {
            vrpn->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        }
        else if (dynasight)
        {
            dynasight->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        }
        mat.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
        if ((x == 0.0) && (y == 0.0) && (z == 0.0))
        {
            x = staticViewerX;
            y = staticViewerY;
            z = staticViewerZ;
        }
        //cerr << "TARSUS: Station=" << station << endl;
        break;

#ifndef __APPLE__
    case coVRTrackingUtil::T_MOTIONSTAR:
        //tracker->getPositionEuler( station, &x, &y, &z, &h, &p, &r);
        //pfMakeEulerMat(mat, h,p,r);

        //phi = 2 * facos(w);
        //pfSetVec3(n, q1/fsin(phi/2), q2/fsin(phi/2), q3/fsin(phi/2));
        //pfMakeRotMat(mat, phi*180/M_PI, n[0], n[1], n[2]);

        mat.makeIdentity();
        tracker->getPositionMatrix(station, &x, &y, &z, m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        mat.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);

        // if((correct_y == 1.0) && (correct_x == 1.0))
        // { // uwes special RUS CUBE correction
        // z -= (fabs(x)+(fabs(y)*0.5))*(fabs(x)+(fabs(y)*0.5))*correct_z;
        // }

        break;

    case coVRTrackingUtil::T_POLHEMUS:
        fs->getQuaternions(station, &w, &q1, &q2, &q3);
#ifdef _WIN32
        phi = 2 * acos(w);
#else
        phi = 2 * facos(w);
#endif
        n.set(q1 / sin(phi / 2), q2 / sin(phi / 2), q3 / sin(phi / 2));
        mat.makeRotate(phi, n[0], n[1], n[2]);
        // get position data
        fs->getAbsPositions(station, &x, &y, &z);
        //fprintf(stderr,"-- in coVRTrackingSystems::getMatrix\n");
        //fprintf(stderr,"---- getAbsPositions: %d %f %f %f\n", station, x, y, z);
        // pos in cm convert it to mm
        x *= 10.0;
        y *= 10.0;
        z *= 10.0;

        break;

#ifndef _WIN32
    case coVRTrackingUtil::T_FOB:
        mat.makeIdentity();
        float m[9];
        fo->getPositionMatrix(station, &x, &y, &z, m + 0, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7, m + 8);
        mat.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
        break;
#endif

    case coVRTrackingUtil::T_CAVELIB:
    {
        if (CaveLibTracker)
        {
            x = CaveLibTracker->sensor[station].x;
            y = CaveLibTracker->sensor[station].y;
            z = CaveLibTracker->sensor[station].z;
            h = CaveLibTracker->sensor[station].azim;
            p = CaveLibTracker->sensor[station].elev;
            r = CaveLibTracker->sensor[station].roll;
            //fprintf(stderr, "H: %f   ",h);
            //fprintf(stderr, "P: %f   ",p);
            //fprintf(stderr, "R: %f\n",r);
            // you can't use this because hpr means something
            // different int Performer than OpenGL mat.makeEuler(h,p,r);
            osg::Matrix H, P, R;
            if (Yup)
            {
                H.makeRotate((h / 180.0f) * M_PI, 0, 1, 0); // H + rot y
                P.makeRotate((p / 180.0f) * M_PI, 1, 0, 0); // P + rot x
                R.makeRotate((r / 180.0f) * M_PI, 0, 0, 1); // R + rot z
            }
            else //Z-Up
            {
                H.makeRotate((h / 180.0f) * M_PI, 0, 0, 1); // H + rot z
                P.makeRotate((p / 180.0f) * M_PI, 1, 0, 0); // P + rot x
                R.makeRotate((r / 180.0f) * M_PI, 0, 1, 0); // R + rot y
            }
            // MAT = R*P*H
            mat = R;
            mat.postMult(P);
            mat.postMult(H);

            x *= scaleFactor; //  POSITION in some unit, default feet
            y *= scaleFactor;
            z *= scaleFactor; // now in mm
        }
        //y+=z*correct_z;
    }
    break;
#endif /* __APPLE__ */
    default:
        mat.makeIdentity();
        x = y = z = 0.0;
        break;
    }
    //if(station==0)
    //{
    // cerr << "Station: " << station << " q1: " << q1<< " q2: " << q2<< " q3: " << q3 << " w: " <<w << endl;
    //cerr << "Station: " << station << " X: " << x<< " Y: " << y<< " Z: " << z << endl;
    //}

    pos.set(x, y, z);
    if ((debugPos == DEBUG_RAW) && (debugStation == station))
        fprintf(stderr, "Station %d RAW [mm]: [%7.1f %7.1f %7.1f]\n", station, x, y, z);

    // use special filter
    mat.setTrans(pos);
    // correct the position data
    if (emFilterInfo.useFlag)
    {
        filterEMField(mat);

        //fprintf(stderr, ".");
        if (emFilterInfo.swapFlag)
        {
            mat(0, 0) *= -1.0;
            mat(0, 1) *= -1.0;

            mat(1, 0) *= -1.0;
            mat(1, 1) *= -1.0;

            mat(2, 0) *= -1.0;
            mat(2, 1) *= -1.0;

            mat(3, 0) *= -1.0;
            mat(3, 1) *= -1.0;
        }

        // and transform into performer coordinate-system
        if ((debugPos == DEBUG_FILTER) && (debugStation == station))
        {
            fprintf(stderr, "Station FILTER: %2d X: %3.3f Y: %3.3f Z: %3.3f \n", station, mat(3, 0), mat(3, 1), mat(3, 2));
        }

        // done
    }

    //interpolation
    if (interpolationFlag)
    {
        //mat[0][0..2] is the lateral vector of the maus
        //mat[1][0..2] is the vector parallel to the cabel and the long ray
        //mat[2][0..2] is the vector perpendicular to the big surfaces of the maus
        //mat[3][0..2] is the position of the tracker maus.
        //mat[0..3][3] is allways 0 and mat[3][3]=1.

        //osg::Matrix matrix_tr;
        float matrix_in[4][4];
        float matrix_tr[4][4];
        int vector, elem;
        static int write_counter = 0;

        // FOR TESTING ONLY
        int btest = 0;
        //static int write_counter;
        // uwe getButton(0, &btest);
        if (btest == 0)
        {
            write_counter = 0;
        }
        if ((btest > 0) && (write_counter == 0) && (station == 0))
        {

            //sprintf (interp_message,"btest=%d\t write_counter=%d station=%d",
            //                        btest,write_counter,station);
            //cout << interp_message << endl;
            //        fprintf(stderr,"mat[0][ ]= %f \t %f \t %f \t %f \t\n",  mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
            //        fprintf(stderr,"mat[1][ ]= %f \t %f \t %f \t %f \t\n",  mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
            //        fprintf(stderr,"mat[2][ ]= %f \t %f \t %f \t %f \t\n",  mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
            //        fprintf(stderr,"mat[3][ ]= %f \t %f \t %f \t %f \t\n",  mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
            //        fprintf(stderr,"\n");
            write_counter++;
        }

        for (vector = 0; vector < 4; vector++)
        {
            for (elem = 0; elem < 4; elem++)
            {
                matrix_in[vector][elem] = mat(vector, elem);
            }
        }

        interpolate(matrix_in, matrix_tr);

        for (vector = 0; vector < 4; vector++)
        {
            for (elem = 0; elem < 4; elem++)
            {
                mat(vector, elem) = matrix_tr[vector][elem];
            }
        }

        if ((btest > 0) && (write_counter == 1) && (station == 0))
        {

            //sprintf (interp_message,"interpolated:\n btest=%d\t write_counter=%d station=%d",
            //                        btest,write_counter,station);
            //cout << interp_message << endl;
            //        fprintf(stderr,"mat[0][ ]= %f \t %f \t %f \t %f \t\n",  mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
            //        fprintf(stderr,"mat[1][ ]= %f \t %f \t %f \t %f \t\n",  mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
            //        fprintf(stderr,"mat[2][ ]= %f \t %f \t %f \t %f \t\n",  mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
            //        fprintf(stderr,"mat[3][ ]= %f \t %f \t %f \t %f \t\n",  mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
            //        fprintf(stderr,"\n");
            write_counter++;
        }
    }

    if (write_calibration_flag == true)
    {
        unsigned int btest;
        char calib_output_line_x[500];
        char calib_output_line_y[500];
        char calib_output_line_z[500];
        char calib_output_line_p[500];
        static int w_counter = 0;
        static int w_line = 0;
        int calib_i, calib_j, calib_k;

        //static int write_counter;
        getButton(0, &btest);
        if (btest == 0)
        {
            w_counter = 0;
        }
        if ((btest & 1) && (w_counter < 1) && (station == 0))
        {
            calib_k = (int)fmod(floor((float)w_line / (float)(1)), w_nk);
            calib_j = (int)fmod(floor((float)w_line / (float)(w_nk)), w_nj);
            calib_i = (int)fmod(floor((float)w_line / (float)(w_nk * w_nj)), w_ni);

            sprintf(calib_output_line_x,
                    "%lf\t%lf\t%lf\t    %lf\t%lf\t%lf\t    %d\t",
                    mat(3, 0), mat(3, 1), mat(3, 2),
                    mat(0, 0), mat(0, 1), mat(0, 2),
                    w_line);

            sprintf(calib_output_line_y,
                    "%lf\t%lf\t%lf\t    %lf\t%lf\t%lf\t    %d\t",
                    mat(3, 0), mat(3, 1), mat(3, 2),
                    mat(1, 0), mat(1, 1), mat(1, 2),
                    w_line);

            sprintf(calib_output_line_z,
                    "%lf\t%lf\t%lf\t    %lf\t%lf\t%lf\t    %d\t",
                    mat(3, 0), mat(3, 1), mat(3, 2),
                    mat(2, 0), mat(2, 1), mat(2, 2),
                    w_line);

            sprintf(calib_output_line_p,
                    "%lf\t%lf\t%lf\t    %lf\t%lf\t%lf\t    %d\t",
                    mat(3, 0), mat(3, 1), mat(3, 2),
                    calib_pos_i[calib_i], calib_pos_j[calib_j], calib_pos_k[calib_k],
                    w_line);

            if (calib_k == 0)
            {
                calib_file_x << endl;
                calib_file_y << endl;
                calib_file_z << endl;
                calib_file_p << endl;

                //cout << endl;
            }

            calib_file_x << calib_output_line_x << endl;
            calib_file_y << calib_output_line_y << endl;
            calib_file_z << calib_output_line_z << endl;
            calib_file_p << calib_output_line_p << endl;

            cout << "i=" << calib_i
                 << "\tj=" << calib_j
                 << "\tk=" << calib_k
                 << "\t|\t" << calib_output_line_p << endl;

            w_line++;
            w_counter++;
        }
    }
}

void
coVRTrackingSystems::getWheel(int station, int *wheel)
{
    if (!wheel)
        return;

    *wheel = 0;
    if (buttonSystem == B_MOUSE
        || trackingSystem == coVRTrackingUtil::T_TARSUS
        || trackingSystem == coVRTrackingUtil::T_VRPN
        || trackingSystem == coVRTrackingUtil::T_DYNASIGHT
        || trackingSystem == coVRTrackingUtil::T_SSD)
    {
        *wheel = mousebuttons->getWheel(station);
    }
    else if (trackingSystem == coVRTrackingUtil::T_DTRACK)
    {
        *wheel = dtrackWheel;
        dtrackWheel = 0;
    }
}

void
coVRTrackingSystems::getButton(int station, unsigned int *button)
{
    static int firsttime = 1;
    static bool debugButton = false;

    if (firsttime)
    {
        debugButton = coCoviseConfig::isOn("COVER.Input.DebugButtons", false);

        firsttime = 0;
    }

    unsigned int rawButton = 0;

    if (buttonSystem == B_MIKE)
    {
        rawButton = buttonData[0] & (~0x80);
    }
    else if (buttonSystem == B_HORNET)
    {
        rawButton = buttonData[0];
    }
    else if (buttonSystem == B_PRESENTER)
    {
        rawButton = VRSceneGraph::instance()->KeyButton[0] | (VRSceneGraph::instance()->KeyButton[1] << 1) | (VRSceneGraph::instance()->KeyButton[2] << 2);
    }
    else if (buttonSystem == B_CYBER)
    {

        //int sobd = sizeof(buttonData);
        //cout << "size of unsigned char " << sizeof(unsigned char) << endl;
        //cout << "sizeof buttonData " << sobd << endl;
        //cout << "buttonData  " << (int)buttonData[2] << "  NOT  " << (~buttonData[2]) << "\n";
        rawButton = ~(buttonData[2]);

        //rawButton_micha = rawButton;

        //cout << "hoffentlich das wheel Byte: " << (int)buttonData[3] << endl;
        //cout << "but " << b1 << b2 << b3 << b4 << b5 << b6 << b7 << b8 << "\n";
    }
    else if (buttonSystem == B_PLUGIN && (buttonSystemPlugin != NULL))
    {
        rawButton = buttonSystemPlugin->button(station);
    }
    else if (buttonSystem == B_VP || buttonSystem == B_CEREAL)
    {
        rawButton = buttonData[0];
    }
    else if (buttonSystem == B_DIVISION)
    {
        rawButton = ~(buttonData[1]);
    }
    else if (buttonSystem == B_VRC)
    {
        rawButton = vrcTracker->getButton(station);
    }
    //vlad: disable VRTracker mouse hardware read

    else if (buttonSystem == B_MOUSE)
    {
#ifdef WIN32
        rawButton = rawTarsusMouse->getButtonBits();
#else
        mousebuttons->getButtons(station, &rawButton);
#endif
    }
    else if (buttonSystem == B_CAVELIB)
    {
        if (CaveLibWand->controller[station].button[0] && CaveLibWand->controller[station].button[1] && CaveLibWand->controller[station].button[2])
        {
            exit(0);
        }
        int bit = 1;
        int numCaveButtons = CaveLibWand->controller[station].num_buttons;
        if (numCaveButtons > 32)
            numCaveButtons = 32;
        for (int i = 0; i < numCaveButtons; i++)
        {
            if (CaveLibWand->controller[station].button[i])
            {
                rawButton |= bit;
            }
            bit = bit << 1;
        }
    }
    else
    {
        switch (trackingSystem)
        {

#ifndef __APPLE__
        case coVRTrackingUtil::T_POLHEMUS:

            fs->getStylusSwitchStatus(station, &rawButton);

        case coVRTrackingUtil::T_MOTIONSTAR:
        {
            tracker->getButtons(station, &rawButton);
        }
        break;
#endif /* __APPLE__ */

        case coVRTrackingUtil::T_DTRACK:
        {
            //fprintf(stderr,"\tbutton from tracking system DTRACK\n");
            rawButton = 0;
            static int lastButton = 0;
            dtrack->getButtons(station, &rawButton);
            if (rawButton & 32)
            {
                dtrackWheel = lastButton & 32 ? 0 : -1;
            }
            else if (rawButton & 16)
                dtrackWheel--;
            else if (rawButton & 128)
            {
                dtrackWheel = lastButton & 128 ? 0 : 1;
            }
            else if (rawButton & 64)
                dtrackWheel++;
            lastButton = rawButton;
        }
        break;
        case coVRTrackingUtil::T_CGVTRACK:
        {
            //fprintf(stderr,"\tbutton from tracking system DTRACK\n");
            cgvtrack->getButtons(station, &rawButton);
        }
        break;
        case coVRTrackingUtil::T_TARSUS:
        case coVRTrackingUtil::T_SSD:
        case coVRTrackingUtil::T_VRPN:
        case coVRTrackingUtil::T_DYNASIGHT:
        {
//fprintf(stderr,"\tbutton from tracking system TARSUS\n");
#ifdef WIN32
            rawButton = rawTarsusMouse->getButtonBits();
#else
            mousebuttons->getButtons(station, &rawButton);
#endif
        }
        break;

#if !defined(_WIN32) && !defined(__APPLE__)
        case coVRTrackingUtil::T_FOB:
        {
            unsigned int b;
            fo->getButtons(station, &b);
            //if(((b>0) && (b<56)) || (b >112 ))
            //   b=1;
            if (b == 16)
                rawButton = 1;
            else if (b == 48)
                rawButton = 2;
            else if (b == 112)
                rawButton = 4;
            else
                rawButton = 0;
        }
        break;

        default:
            rawButton = 0;
            break;
#endif
        }
    }

    //send rawButton. mapping in VRTracker.
    *button = rawButton;
    // cout << "buttonPointer  " << *button << "\n";

    if (debugButton)
    {
        static unsigned int oldButton = 0xffffffff;
        if (oldButton != rawButton)
        {
            fprintf(stderr, "rawButton 0x%04x ", rawButton);
            for (int i = 0; i < 32; i++)
            {
                if (rawButton & (1 << (31 - i)))
                    fprintf(stderr, "1");
                else
                    fprintf(stderr, "0");
            }
            fprintf(stderr, "\n");
            oldButton = rawButton;
        }
    }
}

void
coVRTrackingSystems::getAnalog(int /*station*/, float &x, float &y)
{

    static int init = 0;
    static float xCall = 0.0;
    static float yCall = 0.0;
    x = -10.0;
    y = -10.0;
    if (buttonSystem == B_CAVELIB)
    {
        if (XValuator >= 0)
        {
            x = CaveLibWand->controller[0].valuator[XValuator];
            y = CaveLibWand->controller[0].valuator[YValuator];
        }
        if (init > 10)
        {
            x -= xCall;
            y -= yCall;
            init++;
        }
        else if (init < 10)
        {
            xCall += x;
            yCall += y;
            init++;
        }
        else if (init == 10)
        {
            xCall /= 10.0;
            yCall /= 10.0;
            init++;
        }
    }
    if (buttonSystem == B_CYBER)
    {

        short int xi, yi;
        *((unsigned char *)&xi) = buttonData[5];
        *(((unsigned char *)&xi) + 1) = buttonData[4];
        *((unsigned char *)&yi) = buttonData[7];
        *(((unsigned char *)&yi) + 1) = buttonData[6];

        x = (float)xi;
        y = (float)yi;
        //x = ((float)xi - jsZeroPosX) / 10;
        //y = ((float)yi - jsZeroPosY) / 10;

        /*         static int coutCount = 0;
                 coutCount++;
                 if(coutCount%10 == 0)
                 {
                 fprintf(stderr,"Joystick xi = %d, yi = %d\n",(int)xi,(int)yi);
      //fprintf(stderr,"Joystick x = %f, y = %f\n",x,y);
      }
      if(coutCount > 10000000) coutCount = 0;
       */
        /*      if(init>10)
              {
              x -= xCall;
              y -= yCall;
              init++;
              }
              else if(init<10)
              {
              xCall+=x;
              yCall+=y;
              init++;
              }
              else if(init==10)
              {
              xCall/=10.0;
              yCall/=10.0;
              init++;
              }
       */
        //cout << "XValuator\n";
    }
}

//
// new
//

void
coVRTrackingSystems::getCyberWheel(int /*station*/, int &count)
{
    if (buttonSystem == B_CYBER)
    {
        count = buttonData[3];
    }
}

int
coVRTrackingSystems::readConfigFile()
{
    std::string line;

    trackingSystem = coVRTrackingUtil::instance()->getTrackingSystem();
    if (trackingSystem == coVRTrackingUtil::T_POLHEMUS)
    {

        line = coCoviseConfig::getEntry("COVER.Input.PolhemusConfig.SerialPort");
        if (!line.empty())
        {
            strcpy(portname, line.c_str());
        }
        else
        {
            fprintf(stderr, "WARNING: Entry PolhemusConfig.SerialPort not found in covise.config. Assuming %s\n", DEFAULTSERIAL);
            strcpy(portname, DEFAULTSERIAL);
        }
        baudrate = coCoviseConfig::getInt("COVER.Input.PolhemusConfig.Baudrate", 19200);
    }
    if (trackingSystem == coVRTrackingUtil::T_FOB)
    {
        line = coCoviseConfig::getEntry("COVER.Input.FobConfig.SerialPort");
        if (!line.empty())
        {
            strcpy(portname, line.c_str());
        }
        else
        {
            fprintf(stderr, "WARNING: Entry FobConfig.SERIAL_PORT not found in covise.config Assuming %s\n", DEFAULTSERIAL);
            strcpy(portname, DEFAULTSERIAL);
        }

        baudrate = coCoviseConfig::getInt("COVER.Input.FobConfig.Baudrate", 19200);
    }

    line = coCoviseConfig::getEntry("COVER.Input.ButtonSystem");
    buttonSystem = B_NONE;
    if (!line.empty())
    {
        if (!(strcmp(line.c_str(), "MIKE"))) // the iao or icido device
        {
            buttonSystem = B_MIKE;
        }
        else if (!(strcmp(line.c_str(), "HORNET")))
        {
            buttonSystem = B_HORNET;
        }
        else if (!(strcmp(line.c_str(), "CYBER")))
        {
            buttonSystem = B_CYBER;
        }
        else if (!(strcmp(line.c_str(), "VIRTUAL_PRESENCE")))
        {
            buttonSystem = B_VP;
        }
        else if (!(strcmp(line.c_str(), "PINCH")))
        {
            buttonSystem = B_PINCH;
        }
        else if (!(strcmp(line.c_str(), "CEREAL")))
        {
            buttonSystem = B_CEREAL;
        }
        else if (!(strcmp(line.c_str(), "CAVELIB")))
        {
            buttonSystem = B_CAVELIB;
        }
        else if (!(strcmp(line.c_str(), "DIVISION")))
        {
            buttonSystem = B_DIVISION;
        }
        else if (!(strcmp(line.c_str(), "MOUSE")))
        {
            buttonSystem = B_MOUSE;
        }
        else if (!(strcmp(line.c_str(), "PRESENTER")))
        {
            buttonSystem = B_PRESENTER;
        }
        else if (!(strcmp(line.c_str(), "VRC")))
        {
            buttonSystem = B_VRC;
        }
        else
        {
            buttonSystemPlugin = coVRPluginList::instance()->addPlugin(line.c_str());
            if (buttonSystemPlugin)
            {
                buttonSystem = B_PLUGIN;
            }
            else
                buttonSystem = B_NONE;
        }
    }

    line = coCoviseConfig::getEntry("COVER.Input.ButtonConfig.SerialPort");
    if (!line.empty())
    {
        buttonDevice = new char[strlen(line.c_str()) + 1];
        if (sscanf(line.c_str(), "%s", buttonDevice) != 1)
        {
            cerr << "coVRTrackingSystems::readConfigFile: sscanf4 failed" << endl;
        }
    }
    else
    {
        //fprintf(stderr," WARNING: Entry ButtonConfig.SERIAL_PORT not found in covise.config Assuming %s\n", DEFAULTSERIAL);
        buttonDevice = new char[50];
        strcpy(buttonDevice, DEFAULTSERIAL);
    }

    // read in the correction factors for the magnetic field
    line = coCoviseConfig::getEntry("COVER.Input.LinearMagneticFieldCorrection");
    if (!line.empty())
    {
        if (sscanf(line.c_str(), "%f %f %f", &correct_x, &correct_y, &correct_z) != 3)
        {
            cerr << "coVRTrackingSystems::readConfigFile: sscanf5 failed" << endl;
        }
    }
    else
    {
        correct_x = correct_y = correct_z = 0.0f;
    }

    // get the file name for the trilinear interpolation
    // for tracker correction
    line = coCoviseConfig::getEntry("COVER.Input.InterpolationFile");
    if (!line.empty())
    {
        interpolationFile = coVRFileManager::instance()->getName(line.c_str());
        //fprintf(stderr,"interpolation file name : %s\n", interpolationFile);
        interpolationFlag = true;
    }
    else
    {
        fprintf(stderr, "keyword InterpolationFile not found in covise.config\n");
        interpolationFlag = false;
    }

    orientInterpolationFlag = coCoviseConfig::isOn("COVER.Input.OrientInterpolation", false);

    if (orientInterpolationFlag)
    {
        line = coCoviseConfig::getEntry("COVER.Input.InterpolationFileX");
        ori_file_name_x = coVRFileManager::instance()->getName(line.c_str());
        fprintf(stderr, "ori_file_name_x : %s\n", ori_file_name_x);

        line = coCoviseConfig::getEntry("COVER.Input.InterpolationFileY");
        ori_file_name_y = coVRFileManager::instance()->getName(line.c_str());
        fprintf(stderr, "ori_file_name_y : %s\n", ori_file_name_y);

        line = coCoviseConfig::getEntry("COVER.Input.InterpolationFileZ");
        ori_file_name_z = coVRFileManager::instance()->getName(line.c_str());
        fprintf(stderr, "ori_file_name_z : %s\n", ori_file_name_z);
        orien_interp_files_flag = true;
    }
    else
    {
        fprintf(stderr, "keyword ORIENT_INTERPOLATION not found in covise.config\n");
    }

    write_calibration_flag = coCoviseConfig::isOn("COVER.Input.WriteCalibration", false);

    // EM-Filter settings
    emFilterInfo.useFlag = 0;
    emFilterInfo.swapFlag = 0;
    emFilterInfo.filterType = 1;
    if (coCoviseConfig::isOn("COVER.Input.TrackerCalibration.Status", false))
    {
        emFilterInfo.useFlag = 1;
        // disable former filters
    }
    line = coCoviseConfig::getEntry("COVER.Input.TrackerCalibration.Origin");
    if (!line.empty())
    {
        if (sscanf(line.c_str(), "%f %f %f", &emFilterInfo.origin[0], &emFilterInfo.origin[1], &emFilterInfo.origin[2]) != 3)
        {
            cerr << "coVRTrackingSystems::readConfigFile: sscanf12 failed" << endl;
        }
    }
    else
        emFilterInfo.origin[0] = emFilterInfo.origin[1] = emFilterInfo.origin[2] = 0.0;

    emFilterInfo.alpha = coCoviseConfig::getFloat("COVER.Input.TrackerCalibration.Alpha", 0.0);

    emFilterInfo.beta0 = coCoviseConfig::getFloat("COVER.Input.TrackerCalibration.Beta0", 0.0);
    emFilterInfo.beta1 = coCoviseConfig::getFloat("COVER.Input.TrackerCalibration.Beta1", 0.0);
    emFilterInfo.gamma = coCoviseConfig::getFloat("COVER.Input.TrackerCalibration.Gamma", 0.0);
    emFilterInfo.delta = coCoviseConfig::getFloat("COVER.Input.TrackerCalibration.Delta", 0.0);
    emFilterInfo.up_scale = coCoviseConfig::getFloat("COVER.Input.TrackerCalibration.UpScale", 0.0);
    emFilterInfo.swapFlag = coCoviseConfig::isOn("COVER.Input.TrackerCalibration.Swap", false);
    emFilterInfo.filterType = coCoviseConfig::getInt("COVER.Input.TrackerCalibration.FilterType", 0);

    return (0);
}

bool
coVRTrackingSystems::readInterpolationFile(const char *datapath)
{
    int line, column;
    float temp_energie;

    nx = ny = nz = 0;

    //___________________________________________________________________
    // open file with the deformed coords and their reall coords
    fstream datei;
    datei.open(datapath, ios::in);
    datei.seekg(0, ios::beg);
    /*if (datei)
     {
   //sprintf(interp_message,"File %s was opened to be read\n",datapath);
   //cout << interp_message << endl;
   }
   else
   {
   sprintf (interp_message,"file %s could not be opened for reading",datapath);
   cout << interp_message << endl;
   nx=ny=nz=0;
   return false;
   };*/

    // reading the header
    datei >> nx;
    datei >> ny;
    datei >> nz;
    datei >> ne;
    if (nx < 3 || ny < 3 || nz < 3)
    {
        cout << "Interpolation data file is too small,\n"
             << "it should be at least a 3*3*3 grid" << endl;
        return false;
    }

    //buid the arrays
    x_coord = new float[nx * ny * nz];
    y_coord = new float[nx * ny * nz];
    z_coord = new float[nx * ny * nz];

    n1 = new float[nx * ny * nz];
    n2 = new float[nx * ny * nz];
    n3 = new float[nx * ny * nz];

    //read data
    for (line = 0; line < (nx * ny * nz); line++)
    {
        datei >> x_coord[line];
        datei >> y_coord[line];
        datei >> z_coord[line];
        datei >> n1[line];
        datei >> n2[line];
        datei >> n3[line];

        for (column = 0; column < ne; column++) //only to be compatible
        //with read_itap modul files
        {
            datei >> temp_energie;
        }
    }

    datei.close();
    return true;
}

//void coVRTrackingSystems::interpolate(        const float tracker_p[3] , float p_interp[3] )
void
coVRTrackingSystems::interpolate(const float mat_in[4][4], float mat_tr[4][4])

{
    //        tracker_p :analized point                                                                                          /-----/-----q
    //  cl_ :closest grid point to tracker_p                                                                                        /     /  tp /
    //  q_  :q_ and cl are on the oposite corners of a 3d cuadrant (sector)                              /----cl-----/
    //        int cl_i=0 , cl_j=0 ,cl_k=0;        //,cl_index=0;

    float m_bas[3][3]; // interpolated middle tranformation basis for the exact pos. of tracker_p
    float mat_no[3][3]; // not orthogonal transformated matrix.
    float mat_o_nu[3][3]; // orthogonal but not unitary matrix
    float v_len[3];
    float comp_2in1;
    // diff_bas[direction][vector][comp]
    float tracker_p[3];
    int cl[3]; //  cl_ :closest grid point to tracker_p ( the 3 indexes )                                                                                                /     /  dp /
    int di, dj, dk;
    //float da_001,da_010,da_100;         //distances from 3 planes that have cl_ to tracker_p
    //float db_001,db_010,db_100;                //distances from the planes that are in front of the da_ to tracker_p
    float da[3]; //distances from 3 planes that have cl_ to tracker_p as ve
    float db[3]; //distances from the planes that are in front of the da_ to tracker_p
    float dr[3]; //relativ position of tracker_p in the cell
    bool quad_found;

    //8 points of a cell
    int index_000, index_001, index_010, index_011, index_100, index_101, index_110, index_111;
    float point_000[3];
    float point_001[3];
    float point_010[3];
    float point_011[3];
    float point_100[3];
    float point_101[3];
    float point_110[3];
    //float point_111[3];
    int vector, elem;

    tracker_p[0] = mat_in[3][0];
    tracker_p[1] = mat_in[3][1];
    tracker_p[2] = mat_in[3][2];

    //___________________________________________________________
    //find the closest point

    find_closest_g_point(tracker_p, cl);

    //__________________________________________________________________________________
    //        determinate in which of the 8 quadrants (sectors )
    //        around the closest grid point
    //        is the analized point
    //        the grid point has three neighbors in every cuadrant neig_i,neig_j,neig_k

    quad_found = false;
    for (di = -1; di <= 1; di = di + 2)
    {
        for (dj = -1; dj <= 1; dj = dj + 2)
        {
            for (dk = -1; dk <= 1; dk = dk + 2)
            {
                index_000 = (cl[0]) * ny * nz + (cl[1]) * nz + (cl[2]);
                index_001 = (cl[0]) * ny * nz + (cl[1]) * nz + (cl[2] + dk);
                index_010 = (cl[0]) * ny * nz + (cl[1] + dj) * nz + (cl[2]);
                index_011 = (cl[0]) * ny * nz + (cl[1] + dj) * nz + (cl[2] + dk);
                index_100 = (cl[0] + di) * ny * nz + (cl[1]) * nz + (cl[2]);
                index_101 = (cl[0] + di) * ny * nz + (cl[1]) * nz + (cl[2] + dk);
                index_110 = (cl[0] + di) * ny * nz + (cl[1] + dj) * nz + (cl[2]);
                index_111 = (cl[0] + di) * ny * nz + (cl[1] + dj) * nz + (cl[2] + dk);

                point_000[0] = x_coord[index_000];
                point_001[0] = x_coord[index_001];
                point_010[0] = x_coord[index_010];
                point_011[0] = x_coord[index_011];
                point_100[0] = x_coord[index_100];
                point_101[0] = x_coord[index_101];
                point_110[0] = x_coord[index_110];
                //        point_111[0] = x_coord[index_111] ;

                point_000[1] = y_coord[index_000];
                point_001[1] = y_coord[index_001];
                point_010[1] = y_coord[index_010];
                point_011[1] = y_coord[index_011];
                point_100[1] = y_coord[index_100];
                point_101[1] = y_coord[index_101];
                point_110[1] = y_coord[index_110];
                //        point_111[1] = y_coord[index_111] ;

                point_000[2] = z_coord[index_000];
                point_001[2] = z_coord[index_001];
                point_010[2] = z_coord[index_010];
                point_011[2] = z_coord[index_011];
                point_100[2] = z_coord[index_100];
                point_101[2] = z_coord[index_101];
                point_110[2] = z_coord[index_110];
                //        point_111[2] = z_coord[index_111] ;

                /*                                da_001= dis_pn_pt (         point_000, point_100, point_010, tracker_p   );
                                              da_010= dis_pn_pt (         point_000, point_001, point_100, tracker_p   );
                                              da_100= dis_pn_pt (         point_000, point_010, point_001, tracker_p   );

                                              db_001= dis_pn_pt (         point_001, point_101, point_011, tracker_p   );
                                              db_010= dis_pn_pt (         point_010, point_011, point_110, tracker_p   );
                                              db_100= dis_pn_pt (         point_100, point_110, point_101, tracker_p   );

                                              if (   (da_001*db_001<=0) && (da_010*db_010<=0) && (da_100*db_100<=0)   )
                                              {
            //the point tracker_p is in this cell
            quad_found = true;
            break;
            }
             */
                da[2] = dis_pn_pt(point_000, point_100, point_010, tracker_p);
                da[1] = dis_pn_pt(point_000, point_001, point_100, tracker_p);
                da[0] = dis_pn_pt(point_000, point_010, point_001, tracker_p);

                db[2] = dis_pn_pt(point_001, point_101, point_011, tracker_p);
                db[1] = dis_pn_pt(point_010, point_011, point_110, tracker_p);
                db[0] = dis_pn_pt(point_100, point_110, point_101, tracker_p);

                if ((da[0] * db[0] <= 0) && (da[1] * db[1] <= 0) && (da[2] * db[2] <= 0))
                {
                    //the point tracker_p is in this cell
                    quad_found = true;
                    break;
                }
            }
            if (quad_found == true)
            {
                break;
            }
        }
        if (quad_found == true)
        {
            break;
        }
    }

    dr[0] = da[0] / (da[0] - db[0]); //reativ pos of P_tracker in the cell
    dr[1] = da[1] / (da[1] - db[1]);
    dr[2] = da[2] / (da[2] - db[2]);

    //only copy
    for (vector = 0; vector < 4; vector++)
    {
        for (elem = 0; elem < 4; elem++)
        {
            mat_tr[vector][elem] = mat_in[vector][elem];
        }
    }

    //changes
    mat_tr[3][0] = n1[index_000] + (n1[index_100] - n1[index_000]) * dr[0];
    mat_tr[3][1] = n2[index_000] + (n2[index_010] - n2[index_000]) * dr[1];
    mat_tr[3][2] = n3[index_000] + (n3[index_001] - n3[index_000]) * dr[2];

    if (orientInterpolationFlag == true)
    {
        // find a intermediary basis for the tracker pos
        for (vector = 0; vector < 3; vector++)
        {
            for (elem = 0; elem < 3; elem++)
            {
                val8[0][0][0] = trans_basis[index_000][vector][elem];
                val8[0][0][1] = trans_basis[index_001][vector][elem];
                val8[0][1][0] = trans_basis[index_010][vector][elem];
                val8[0][1][1] = trans_basis[index_011][vector][elem];
                val8[1][0][0] = trans_basis[index_100][vector][elem];
                val8[1][0][1] = trans_basis[index_101][vector][elem];
                val8[1][1][0] = trans_basis[index_110][vector][elem];
                val8[1][1][1] = trans_basis[index_111][vector][elem];

                val4[0][0] = val8[0][0][0] + (val8[0][0][1] - val8[0][0][0]) * dr[2];
                val4[0][1] = val8[0][1][0] + (val8[0][1][1] - val8[0][1][0]) * dr[2];
                val4[1][0] = val8[1][0][0] + (val8[1][0][1] - val8[1][0][0]) * dr[2];
                val4[1][1] = val8[1][1][0] + (val8[1][1][1] - val8[1][1][0]) * dr[2];

                val2[0] = val4[0][0] + (val4[0][1] - val4[0][0]) * dr[1];
                val2[1] = val4[1][0] + (val4[1][1] - val4[1][0]) * dr[1];

                val1 = val2[0] + (val2[1] - val2[0]) * dr[0];
                m_bas[vector][elem] = val1;
            }
        }

        //matrix multiplication
        for (vector = 0; vector < 3; vector++)
        {
            for (elem = 0; elem < 3; elem++)
            {

                mat_no[vector][elem] = (m_bas[0][elem] * mat_in[vector][0] + m_bas[1][elem] * mat_in[vector][1] + m_bas[2][elem] * mat_in[vector][2]);

                /*
               mat_tr[vector][elem]=        (
               m_bas[0][elem]*mat_in[vector][0]+
               m_bas[1][elem]*mat_in[vector][1]+
               m_bas[2][elem]*mat_in[vector][2]   );

             */
            }
        }

        //make the vectors orthogonal
        // vector[1] derection will be keeped
        for (elem = 0; elem < 3; elem++)
        {
            mat_o_nu[1][elem] = mat_no[1][elem];
        }
        v_len[1] = sqrt(mat_no[1][0] * mat_no[1][0] + mat_no[1][1] * mat_no[1][1] + mat_no[1][2] * mat_no[1][2]);

        // vector[2]
        comp_2in1 = (mat_no[1][0] * mat_no[2][0] + mat_no[1][1] * mat_no[2][1] + mat_no[1][2] * mat_no[2][2]) / v_len[1];

        mat_o_nu[2][0] = mat_no[2][0] - comp_2in1 * mat_no[1][0] / v_len[1];
        mat_o_nu[2][1] = mat_no[2][1] - comp_2in1 * mat_no[1][1] / v_len[1];
        mat_o_nu[2][2] = mat_no[2][2] - comp_2in1 * mat_no[1][2] / v_len[1];
        v_len[2] = sqrt(mat_o_nu[2][0] * mat_o_nu[2][0] + mat_o_nu[2][1] * mat_o_nu[2][1] + mat_o_nu[2][2] * mat_o_nu[2][2]);

        mat_o_nu[0][0] = +mat_no[1][1] * mat_no[2][2] - mat_no[1][2] * mat_no[2][1];
        mat_o_nu[0][1] = -mat_no[1][0] * mat_no[2][2] + mat_no[1][2] * mat_no[2][0];
        mat_o_nu[0][2] = +mat_no[1][0] * mat_no[2][1] - mat_no[1][1] * mat_no[2][0];

        v_len[0] = sqrt(mat_o_nu[0][0] * mat_o_nu[0][0] + mat_o_nu[0][1] * mat_o_nu[0][1] + mat_o_nu[0][2] * mat_o_nu[0][2]);

        for (vector = 0; vector < 3; vector++)
        {
            for (elem = 0; elem < 3; elem++)
            {
                mat_tr[vector][elem] = mat_o_nu[vector][elem] / v_len[vector];
            }
        }
    }
}

//_____________________________________________________________________
void
coVRTrackingSystems::find_closest_g_point(const float tracker_p[3], int cl[3])
/*
   void
   coVRTrackingSystems::find_closest_g_point (float dp_x, float dp_y,        float dp_z,
   int *cl_i,         int *cl_j,        int *cl_k        )
 */
{

    // find the closest grid point
    // the points on the borders are excluded
    int closest_i = 0, closest_j = 0, closest_k = 0;
    bool first;
    int iii, jjj, kkk, index;
    float vgp_x, vgp_y, vgp_z; // vector from a grid point to the tracker point
    float dis_min = 0, distance = 0;

    first = true;
    for (iii = 1; iii <= (nx - 2); iii++)
    {
        for (jjj = 1; jjj <= (ny - 2); jjj++)
        {
            for (kkk = 1; kkk <= (nz - 2); kkk++)
            {
                index = (iii * ny * nz + jjj * nz + kkk);
                /*                                cout << "iii =" << iii << "\tjjj =" << jjj << "\tkkk =" << kkk << "\tindex=" << index << endl;
             */
                vgp_x = tracker_p[0] - x_coord[index];
                vgp_y = tracker_p[1] - y_coord[index];
                vgp_z = tracker_p[2] - z_coord[index];
                distance = sqrt(vgp_x * vgp_x + vgp_y * vgp_y + vgp_z * vgp_z);

                if ((first == true) || (distance < dis_min))
                {
                    dis_min = distance;
                    closest_i = iii;
                    closest_j = jjj;
                    closest_k = kkk;
                }
                first = false;
            }
        }
    }

    cl[0] = closest_i;
    cl[1] = closest_j;
    cl[2] = closest_k;
}

//___________________________________________________________________________
float
coVRTrackingSystems::dis_pn_pt(const float a[3],
                               const float b[3],
                               const float c[3],
                               const float p[3])
/*
   float
   coVRTrackingSystems::dis_pn_pt (         float ax , float ay , float az ,
   float bx , float by , float bz ,
   float cx , float cy , float cz ,
   float px , float py , float pz    )
 */
{
    // Calculate the distance from a plane (a_ b_ c_) to a point (p_)
    float abx, aby, abz;
    float acx, acy, acz;
    float apx, apy, apz;
    float tx, ty, tz;
    float area, volumen, dist;

    abx = b[0] - a[0];
    aby = b[1] - a[1];
    abz = b[2] - a[2];

    acx = c[0] - a[0];
    acy = c[1] - a[1];
    acz = c[2] - a[2];

    apx = p[0] - a[0];
    apy = p[1] - a[1];
    apz = p[2] - a[2];

    tx = aby * acz - abz * acy;
    ty = -abx * acz + abz * acx;
    tz = abx * acy - aby * acx;

    area = sqrt(tx * tx + ty * ty + tz * tz);
    volumen = (abx * acy * apz + aby * acz * apx + abz * acx * apy - apx * acy * abz - apy * acz * abx - apz * acx * aby);

    if (area != 0)
    {
        dist = volumen / area;
    }
    else
    {
        dist = 0;
        cout << "division by 0 in funktion dis_pn_pt"
             << "three poins of the plane are in the same line " << endl;
    }
    return dist;
}

//_________________________________________________________________________
int
coVRTrackingSystems::find_xyz_velocity(void)
//find in which order are changing x,y,z of the real values of the
// interpolation file,for example:
// find_xyz_velocity = 123 means that z chance faster, than y and than x
// find_xyz_velocity = 213 means that z chance faster, than x and than y (cave)
//        in case of 213 the data will be reorganized to 123 (standard)
{
    int n1_velocity = 0, n2_velocity = 0, n3_velocity = 0;

    //n1 change with i
    if (n1[ny * nz] != n1[0])
    {
        n1_velocity = 1;
    }
    else if (n1[nz] != n1[0]) //n1 change with j
    {
        n1_velocity = 2;
    }
    //n1 change with k
    else if (n1[1] != n1[0])
    {
        n1_velocity = 3;
    }

    //n2 change with i
    if (n2[ny * nz] != n2[0])
    {
        n2_velocity = 1;
    }
    //n2 change with j
    else if (n2[nz] != n2[0])
    {
        n2_velocity = 2;
    }
    //n2 change with k
    else if (n2[1] != n2[0])
    {
        n2_velocity = 3;
    }

    //n3 change with i
    if (n3[ny * nz] != n3[0])
    {
        n3_velocity = 1;
    }
    //n3 change with j
    else if (n3[nz] != n3[0])
    {
        n3_velocity = 2;
    }
    //n3 change with k
    else if (n3[1] != n3[0])
    {
        n3_velocity = 3;
    }

    return n1_velocity * 100 + n2_velocity * 10 + n3_velocity;
}

//_________________________________________________________________________
void
coVRTrackingSystems::reorganize_data(int xyz_vel)
//reorganize interpolation data file
// from xyz_velocity=213 to xyz_velocity=123
// so that z chance faster, than y and than x
{
    int t_nx, t_ny, t_nz;
    int i, j, k, m;
    int index, t_index;

    // t_ for a temporary copy
    float *t_x_coord = 0, *t_y_coord = 0, *t_z_coord = 0;
    float *t_n1 = 0, *t_n2 = 0, *t_n3 = 0;

    t_nx = nx;
    t_ny = ny;
    t_nz = nz;

    t_x_coord = new float[nx * ny * nz];
    t_y_coord = new float[nx * ny * nz];
    t_z_coord = new float[nx * ny * nz];

    t_n1 = new float[nx * ny * nz];
    t_n2 = new float[nx * ny * nz];
    t_n3 = new float[nx * ny * nz];

    for (m = 0; m < nx * ny * nz; m++)
    {

        t_x_coord[m] = x_coord[m];
        t_y_coord[m] = y_coord[m];
        t_z_coord[m] = z_coord[m];

        t_n1[m] = n1[m];
        t_n2[m] = n2[m];
        t_n3[m] = n3[m];
    }

    //vertauchen
    if (xyz_vel == 213)
    {
        nx = t_ny;
        ny = t_nx;
        nz = t_nz;

        for (i = 0; i < nx; i++)
        {
            for (j = 0; j < ny; j++)
            {
                for (k = 0; k < nz; k++)
                {
                    index = j * t_ny * t_nz + i * t_nz + k;
                    t_index = i * ny * nz + j * nz + k;

                    x_coord[t_index] = t_x_coord[index];
                    y_coord[t_index] = t_y_coord[index];
                    z_coord[t_index] = t_z_coord[index];

                    n1[t_index] = t_n1[index];
                    n2[t_index] = t_n2[index];
                    n3[t_index] = t_n3[index];
                }
            }
        }
    }
    /*
      for (index=0 ; index < nx*ny*nz ; index++)
      {
      sprintf (interp_message,"%15f %15f %15f %15f %15f %15f",
      x_coord[index],y_coord[index],z_coord[index],
      n1[index],n2[index],n3[index]);
      cout << interp_message << endl;
      }
    */

    if (t_x_coord != 0)
    {
        delete[] t_x_coord;
    }
    if (t_y_coord != 0)
    {
        delete[] t_y_coord;
    }
    if (t_z_coord != 0)
    {
        delete[] t_z_coord;
    }

    if (t_n1 != 0)
    {
        delete[] t_n1;
    }
    if (t_n2 != 0)
    {
        delete[] t_n2;
    }
    if (t_n3 != 0)
    {
        delete[] t_n3;
    }
}

//        functions for the orientation

//__________________________________________________________________________
void
coVRTrackingSystems::create_trans_basis(void)
{
    int counter, axis, vector, comp;
    int iii, jjj, kkk, index, index_p, index_m;

    int di[3];
    int v_index[3], v_max[3];
    float unit_mat[3][3] = {
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 }
    };

    float pp[3][3]; // 3 points around the knot on the pos axis
    float pm[3][3]; // on the negativ axis
    float vt[3][3]; // 3 vectors tangent to the knot
    float vt_len;

    //the same but with the real values
    float ppr[3][3]; // 3 points around the knot on the pos axis
    float pmr[3][3]; // on the negativ axis
    float vtr, vtr_sg;

    if (trans_basis == NULL)
    {
        trans_basis = new float **[nx * ny * nz];
        for (counter = 0; counter < nx * ny * nz; counter++)
        {
            trans_basis[counter] = new float *[3];
            for (vector = 0; vector < 3; vector++)
            {
                trans_basis[counter][vector] = new float[3];
            }
        }
    }

    v_max[0] = nx - 1;
    v_max[1] = ny - 1;
    v_max[2] = nz - 1;
    for (iii = 0; iii < (nx); iii++)
    {
        for (jjj = 0; jjj < (ny); jjj++)
        {
            for (kkk = 0; kkk < (nz); kkk++)
            {
                index = iii * ny * nz + jjj * nz + kkk;
                v_index[0] = iii;
                v_index[1] = jjj;
                v_index[2] = kkk;

                for (axis = 0; axis < 3; axis++)
                {
                    di[0] = 0;
                    di[1] = 0;
                    di[2] = 0;
                    di[axis] = 1;

                    index_p = (iii + di[0]) * ny * nz + (jjj + di[1]) * nz + (kkk + di[2]);
                    index_m = (iii - di[0]) * ny * nz + (jjj - di[1]) * nz + (kkk - di[2]);

                    if (v_index[axis] == 0)
                    {
                        index_m = index;
                    }

                    if (v_index[axis] == v_max[axis])
                    {
                        index_p = index;
                    }

                    pp[axis][0] = x_coord[index_p];
                    pp[axis][1] = y_coord[index_p];
                    pp[axis][2] = z_coord[index_p];
                    pm[axis][0] = x_coord[index_m];
                    pm[axis][1] = y_coord[index_m];
                    pm[axis][2] = z_coord[index_m];

                    ppr[axis][0] = n1[index_p];
                    ppr[axis][1] = n2[index_p];
                    ppr[axis][2] = n3[index_p];
                    pmr[axis][0] = n1[index_m];
                    pmr[axis][1] = n2[index_m];
                    pmr[axis][2] = n3[index_m];

                    vtr = ppr[axis][axis] - pmr[axis][axis];
                    vtr_sg = vtr / fabs(vtr);

                    for (comp = 0; comp < 3; comp++)
                    {
                        vt[axis][comp] = (pp[axis][comp] - pm[axis][comp]);
                    }
                    vt_len = sqrt(vt[axis][0] * vt[axis][0] + vt[axis][1] * vt[axis][1] + vt[axis][2] * vt[axis][2]);

                    for (comp = 0; comp < 3; comp++)
                    {
                        vt[axis][comp] = vt[axis][comp] / vt_len * vtr_sg;
                    }
                }
                for (axis = 0; axis < 3; axis++)
                {

                    linear_equations_sys(vt[0], vt[1], vt[2],
                                         unit_mat[axis], trans_basis[index][axis]);
                };
            }
        }
    }
}

void
coVRTrackingSystems::read_trans_basis(void)
{
    int index;
    int vector, elem;
    float temp;
    int r_nx, r_ny, r_nz, r_ne;
    char file_name[3][200];
    fstream ori_file[3];
    float unit_mat[3][3] = {
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 }
    };
    float vt[3][3]; // 3 vectors tangent to the knot

    r_nx = r_ny = r_nz = 0;
    strcpy(file_name[0], ori_file_name_x);
    strcpy(file_name[1], ori_file_name_y);
    strcpy(file_name[2], ori_file_name_z);

    // open files with the orientations
    for (vector = 0; vector < 3; vector++)
    {
        ori_file[vector].open(file_name[vector], ios::in);
        ori_file[vector].seekg(0, ios::beg);

        /*if (ori_file[vector])
        {
        sprintf (interp_message,"file %s was opened to read",file_name[vector]);
        cout << interp_message << endl;
        }
        else
        {
        sprintf (interp_message,"file %s could not be opened to read\n",file_name[vector]);
        cout << interp_message << endl;
        orien_interp_files_flag=false;
        return;
        }*/

        ori_file[vector] >> r_nx >> r_ny >> r_nz >> r_ne;

        if (r_nx != nx || r_ny != ny || r_nz != nz)
        {
            sprintf(interp_message,
                    "ERROR:interpolation file size and %s size are different", file_name[vector]);
            cout << interp_message << endl;
            orien_interp_files_flag = false;
            return;
        }
    }

    if (trans_basis == NULL)
    {
        trans_basis = new float **[nx * ny * nz];
        for (index = 0; index < nx * ny * nz; index++)
        {
            trans_basis[index] = new float *[3];
            for (vector = 0; vector < 3; vector++)
            {
                trans_basis[index][vector] = new float[3];
            }
        }
    }

    for (index = 0; index < nx * ny * nz; index++)
    {
        for (vector = 0; vector < 3; vector++)
        {
            ori_file[vector] >> temp >> temp >> temp
                >> vt[vector][0]
                >> vt[vector][1]
                >> vt[vector][2]
                >> temp;
        }
        for (vector = 0; vector < 3; vector++)
        {
            if (vt[vector][vector] < 0)
            {
                for (elem = 0; elem < 3; elem++)
                {
                    vt[vector][elem] = -vt[vector][elem];
                }
            }
        }

        for (vector = 0; vector < 3; vector++)
        {
            linear_equations_sys(vt[0], vt[1], vt[2],
                                 unit_mat[vector], trans_basis[index][vector]);
        };
    }

    for (vector = 0; vector < 3; vector++)
    {
        ori_file[vector].close();
    }
}

void
coVRTrackingSystems::linear_equations_sys(const float c0[3], const float c1[3], const float c2[3],
                                          const float b[3], float a[3])
{
    // c_ are the columns of the matrix M, Ma=b,only b is known.
    float divisor;

    divisor = determinante(c0, c1, c2);
    if (divisor == 0)
    {
        sprintf(interp_message, "M.ERROR:linear_equations_sys:Determinante = 0");
        cout << interp_message << endl;
        a[0] = 1;
        a[1] = 0;
        a[2] = 0;
    }
    else
    {
        a[0] = determinante(b, c1, c2) / divisor;
        a[1] = determinante(c0, b, c2) / divisor;
        a[2] = determinante(c0, c1, b) / divisor;
    }
}

float
coVRTrackingSystems::determinante(const float c0[3], const float c1[3], const float c2[3])
{
    float det;
    det = ((c0[0]) * (c1[1]) * (c2[2])
           + (c0[1]) * (c1[2]) * (c2[0])
           + (c0[2]) * (c1[0]) * (c2[1])
           - (c2[0]) * (c1[1]) * (c0[2])
           - (c2[1]) * (c1[2]) * (c0[0])
           - (c2[2]) * (c1[0]) * (c0[1]));

    if (det == 0)
    {
        cout << "coVRTrackingSystems::determinante: det = " << det << endl;
    }

    return det;
}

//end of functions for interpolation
//__________________________________________________________________________

/////
/////
/////

void coVRTrackingSystems::filterEMField(osg::Matrix &mat)
{
    float p[3], v[3];

    float pos[3], ex[3], ey[3]; //, ez[3];
    //float n;
    //float v0[3], v1[3], v2[3];

    // get information from damn matrix
    pos[0] = mat(3, 0);
    pos[1] = mat(3, 1);
    pos[2] = mat(3, 2);

    ex[0] = mat(0, 0);
    ex[1] = mat(0, 1);
    ex[2] = mat(0, 2);

    ey[0] = mat(1, 0);
    ey[1] = mat(1, 1);
    ey[2] = mat(1, 2);

    //ez[0] = mat(2][0];
    //ez[1] = mat[2][1];
    //ez[2] = mat[2][2];

    // here we go

    // position first
    p[0] = pos[0];
    p[1] = pos[1];
    p[2] = pos[2];
    filterEMPoint(p[0], p[1], p[2], emFilterInfo.filterType);

    // now the "orientations"

    // x
    v[0] = pos[0] + ex[0];
    v[1] = pos[1] + ex[1];
    v[2] = pos[2] + ex[2];
    filterEMPoint(v[0], v[1], v[2], emFilterInfo.filterType);
    ex[0] = v[0] - p[0];
    ex[1] = v[1] - p[1];
    ex[2] = v[2] - p[2];

    // y
    v[0] = pos[0] + ey[0];
    v[1] = pos[1] + ey[1];
    v[2] = pos[2] + ey[2];
    filterEMPoint(v[0], v[1], v[2], emFilterInfo.filterType);
    ey[0] = v[0] - p[0];
    ey[1] = v[1] - p[1];
    ey[2] = v[2] - p[2];

    // z
    /*
      v[0] = pos[0]+ez[0];
      v[1] = pos[1]+ez[1];
      v[2] = pos[2]+ez[2];
      filterEMPoint( v[0], v[1], v[2] );
      ez[0] = v[0]-p[0];
      ez[1] = v[1]-p[1];
      ez[2] = v[2]-p[2];
    */

    osg::Vec3 vec0, vec1, vec2;
    vec0.set(ex[0], ex[1], ex[2]);
    vec1.set(ey[0], ey[1], ey[2]);
    //   pfSetVec3( vec2, ez[0], ez[1], ez[2] );
    vec0.normalize();
    vec1.normalize();
    //   pfNormalizeVec3( vec2 );

    vec2 = vec0 ^ vec1;
    vec0 = vec1 ^ vec2;

    ex[0] = vec0[0];
    ex[1] = vec0[1];
    ex[2] = vec0[2];

    ey[0] = vec1[0];
    ey[1] = vec1[1];
    ey[2] = vec1[2];

    //ez[0] = vec2[0];
    //ez[1] = vec2[1];
    //ez[2] = vec2[2];

    // build new (filtered) matrix
    mat(3, 0) = p[0];
    mat(3, 1) = p[1];
    mat(3, 2) = p[2];

    mat(0, 0) = vec0[0];
    mat(0, 1) = vec0[1];
    mat(0, 2) = vec0[2];

    mat(1, 0) = vec1[0];
    mat(1, 1) = vec1[1];
    mat(1, 2) = vec1[2];

    mat(2, 0) = vec2[0];
    mat(2, 1) = vec2[1];
    mat(2, 2) = vec2[2];

    /*
      n = -1.0 / sqrt(ex[0]*ex[0] + ex[1]*ex[1] + ex[2]*ex[2]);
      mat[1][0] = ex[0]*n;
      mat[1][1] = ex[1]*n;
      mat[1][2] = ex[2]*n;

      n = -1.0 / sqrt(ey[0]*ey[0] + ey[1]*ey[1] + ey[2]*ey[2]);
      mat[0][0] = ey[0]*n;
      mat[0][1] = ey[1]*n;
      mat[0][2] = ey[2]*n;

      n = -1.0 / sqrt(ez[0]*ez[0] + ez[1]*ez[1] + ez[2]*ez[2]);
      mat[2][0] = ez[0]*n;
      mat[2][1] = ez[1]*n;
      mat[2][2] = ez[2]*n;
    */

    // done
    return;
}

void coVRTrackingSystems::filterEMPoint(float &x, float &y, float &z, float, float, float)
{
    float origin[3];
    float alpha, beta0, beta1, gamma, delta, up_scale;

    float v[3];
    float c, s, a, b;
    float rX, rY, rZ;

    //   rX = x;
    //   rY = y;
    //   rZ = z;

    rX = -y;
    rY = -z;
    rZ = x;

    // the following parameters may be retrieved using ms and TrackerCalibration
    origin[0] = -22.0;
    origin[1] = 29.0;
    origin[2] = 202.0;
    alpha = -0.55;
    beta0 = 0.5;
    beta1 = 0.4;
    gamma = 30.0;
    delta = 35.0;
    up_scale = 0.4;

    /*

      origin[0] = emFilterInfo.origin[0];
      origin[1] = emFilterInfo.origin[1];
      origin[2] = emFilterInfo.origin[2];
      alpha = emFilterInfo.alpha;
      beta0 = emFilterInfo.beta0;
      beta1 = emFilterInfo.beta1;
      gamma = emFilterInfo.gamma;
      delta = emFilterInfo.delta;
      up_scale = emFilterInfo.up_scale;
    */

    // translate
    x = rX - origin[0];
    y = rY - origin[1];
    z = rZ - origin[2];

    // rotate
    c = cos(alpha);
    s = sin(alpha);
    v[0] = x;
    v[1] = c * y - s * z;
    v[2] = s * y + c * z;

    // adjust z according to y
    // use linear interpolation with v[2] !
    a = (beta0 + beta1) / 2.0;
    b = (beta0 - beta1) / 280.0;
    v[2] += v[1] * (a - v[2] * b);

    // use a cosinus to flatten the whole thing
    a = (v[0] / 140) * (M_PI / 2.0);
    v[1] -= gamma * cos(a);

    // and now we might have to adjust v[1] again
    v[1] += delta;

    // even do it linear
    v[1] *= (1.0 + up_scale);

    // we got it
    x = v[0];
    y = v[1];
    z = v[2];

    y = -v[2];
    z = v[1];

    //   x = -x;
    //   y = -y;
    //   z = -z;

    // done
    return;
}

void coVRTrackingSystems::filterEMPoint(float &x, float &y, float &z, int filterType)
{
    float origin[3];
    float alpha, beta0, beta1, gamma, delta, up_scale;

    float v[3];
    float c, s, a, b;
    float rX, rY, rZ;
    float par1[10], par2[10], par3[10];

    switch (filterType)
    {
    case 2: // use hype

        rX = -y;
        rY = x;
        rZ = z;

        // set parameters
        par1[3] = 20.46865338;
        par1[4] = -3.785633507;
        par1[5] = 0.030199765;
        par1[6] = 0.001999956;
        par1[7] = 4861.914259;
        par1[8] = -1275458.475;
        par1[9] = -71.12116027;

        /*
            par2[1] = -155.1609653;
            par2[2] = 1.796722109;
            par2[3] = -0.005526336;
            par2[4] = 0.018665718;
            par2[5] = 0.002429597;
            par2[6] = 0.000000538687;
            par2[7] = -0.010180046;
            par2[8] = 0.00237541;
            par2[9] = -0.0000021188;
            y = (par2[1] + par2[2]*rY + par2[3]*(rY*rY)) + (par2[4]*rX + par2[5]*(rX*rX) + par2[6]*(rX*rX*rX)) + (par2[7]*rZ + par2[8]*(rZ*rZ) + par2[9]*(rZ*rZ*rZ));
          */

        par2[1] = -154.6761989;
        par2[2] = 0.024596565;
        par2[3] = 2.048112357;
        par2[4] = -0.053116184;
        par2[5] = 0.002264969;
        par2[6] = -0.007501853;
        par2[7] = 0.002480045;
        par2[8] = 0.000274596;

        par3[3] = 2.571399774;
        par3[4] = -0.00756446;
        par3[5] = -0.03468848;
        par3[6] = 6.01449908;
        par3[7] = 0.169742954;
        par3[8] = -0.00171691;
        par3[9] = -0.00038588;

        a = (rX * rX) + (rZ * rZ);

        // x
        x = par1[3] + par1[4] * rX + par1[5] * rZ + rX * ((a + par1[6]) * par1[7] + powf(a + par1[6], 2.0) * par1[8] + powf(a + par1[6], 3.0) * par1[9]);

        // y
        y = par2[1] + rX * par2[2] + rY * par2[3] + rZ * par2[4] + rX * rX * par2[5] + rY * rY * par2[6] + rZ * rZ * par2[7] + rX * rZ * par2[8];

        // z
        z = par3[3] + par3[4] * rX + par3[5] * rZ + rX * ((a + par3[6]) * par3[7] + powf(a + par3[6], 2.0) * par3[8] + powf(a + par3[6], 3.0) * par3[9]);

        // done

        rX = x;
        rY = y;
        rZ = z;

        //x = rY;
        // y = -rX;
        //z = rZ;

        //x = rY;
        //y = rX;
        //z = -rZ;

        //x = -rY;

        // turn into performer-coordinate-system, which should be x to the right, y into the
        // screen, z up

        //x = rX;
        //y = -rZ;
        //z = rY;

        x = -rZ;
        y = rX;
        z = -rY;

        //x = -rX;
        //y = -rZ;
        //z = -rY;

        break;

    default: // use standard (poor quality)

        //   rX = x;
        //   rY = y;
        //   rZ = z;

        rX = -y;
        rY = -z;
        rZ = x;

        //   rX = x;rX;

        //   rY = y;
        //   rZ = z;

        // the following parameters may be retrieved using ms and TrackerCalibration
        origin[0] = -22.0;
        origin[1] = 29.0;
        origin[2] = 202.0;
        alpha = -0.55;
        beta0 = 0.5;
        beta1 = 0.4;
        gamma = 30.0;
        delta = 35.0;
        up_scale = 0.4;

        /*

            origin[0] = emFilterInfo.origin[0];
            origin[1] = emFilterInfo.origin[1];
            origin[2] = emFilterInfo.origin[2];
            alpha = emFilterInfo.alpha;
            beta0 = emFilterInfo.beta0;
            beta1 = emFilterInfo.beta1;
            gamma = emFilterInfo.gamma;
            delta = emFilterInfo.delta;
            up_scale = emFilterInfo.up_scale;
          */

        // translate
        x = rX - origin[0];
        y = rY - origin[1];
        z = rZ - origin[2];

        // rotate
        c = cos(alpha);
        s = sin(alpha);
        v[0] = x;
        v[1] = c * y - s * z;
        v[2] = s * y + c * z;

        // adjust z according to y
        // use linear interpolation with v[2] !
        a = (beta0 + beta1) / 2.0;
        b = (beta0 - beta1) / 280.0;
        v[2] += v[1] * (a - v[2] * b);

        // use a cosinus to flatten the whole thing
        a = (v[0] / 140) * (M_PI / 2.0);
        v[1] -= gamma * cos(a);

        // and now we might have to adjust v[1] again
        v[1] += delta;

        // even do it linear
        v[1] *= (1.0 + up_scale);

        // we got it
        x = v[0];
        y = v[1];
        z = v[2];

        //y = -v[2];
        //z = v[1];

        //   x = -x;
        //   y = -y;
        //   z = -z;

        x = v[2];
        y = -v[0];
        z = -v[1];
        break;
    }

    // done
    return;
}

void
coVRTrackingSystems::getCerealAnalog(int station, float **value)
{
    if (buttonSystem == B_CEREAL)
    {
        cerr << "setting CerealAnalog addresses\n";
        *value = &analogData[station];
    }
}

int coVRTrackingSystems::getNumMarkers()
{
    if (tarsus)
        return tarsus->getNumMarkers();
    else
        return 0;
}

bool coVRTrackingSystems::getMarker(int index, float *pos)
{
    if (tarsus)
    {
        static osg::Matrix offsetMat;
        static bool offsetsRead = false;
        if (!offsetsRead)
        {
            offsetMat = VRTracker::instance()->trackingUtil->computeDeviceOffsetMat(coVRTrackingUtil::trackingSys);
            offsetsRead = true;
        }

        bool visible = tarsus->getMarker(index, pos);
        osg::Vec4 v(pos[0], pos[1], pos[2], 1.0);
        v = offsetMat.preMult(v);
        for (int i = 0; i < 3; ++i)
            pos[i] = v[i] / v[3];

        return visible;
    }
    else
        return false;
}

const DTrack::FingerData *coVRTrackingSystems::getFingerData(coVRTrackingUtil::IDOfDevice hand)
{
    if (hand != coVRTrackingUtil::handDev && hand != coVRTrackingUtil::secondHandDev)
        return 0;

    if (dtrack != 0)
    {
        return dtrack->getFingerData(VRTracker::instance()->trackingUtil->getDeviceAddress(hand));
    }
    else
        return 0;
}
