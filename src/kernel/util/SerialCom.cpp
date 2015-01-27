/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS SerialCom
//
// This class is a primitive Interface to serial connections
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.1 non-blocking I/O with select
//

#include "SerialCom.h"

#include "common.h"
///////#include <sys/types.h>
//#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef WIN32
#include <unistd.h>

#include <sys/ioctl.h>
#endif

#ifdef __linux__
#include <termios.h> /* tcsetattr */
#endif

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/** Initialisiert die serielle Schnittstelle so, dass eine 	**/
/** Zeichenweise E/A stattfindet ( raw Modus )		   	**/
/**  Datenbits : 8						**/
/**  Baudrate  : baud_rate					**/
/**  Stopbits  : 1						**/
/**  Startbits : 1	  ( wird automatisch gesendet )		**/
/**  Paritaet  : keine						**/
/**  Handshake : HW  						**/

namespace covise
{

SerialCom::SerialCom(const char *device, int baudrate, int Parity, int DataBits, int StopBits)
{
    // no errors yey
    d_error[0] = '\0';

#ifdef _WIN32

    DCB dcb;
    BOOL fSuccess;
    //device ==  "COM2";

    d_channel = CreateFile(device,
                           GENERIC_READ | GENERIC_WRITE,
                           0, // must be opened with exclusive-access
                           NULL, // no security attributes
                           OPEN_EXISTING, // must use OPEN_EXISTING
                           0, // not overlapped I/O
                           NULL // hTemplate must be NULL for comm devices
                           );

    if (d_channel == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", device, GetLastError());
        d_channel = 0;
    }

    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    fSuccess = GetCommState(d_channel, (LPDCB)&dcb);
    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
    }

    /* First, get the parameters which can be set in the Preferences */
    switch (baudrate)
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

    fSuccess = SetCommState(d_channel, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
    }
#else
#ifndef __APPLE__
    bool NoDelayFlag = false;
    int BaudRate = baudrate;
    int SerialSpeed = 0;
    int SerialBits = 0;
    struct termios TermPar;

    //cerr<<"[CoviseInterfaceSerial::Open] Opening Termio chan on "<<device
    //       <<" with baudrate" <<BaudRate<<"\n";

    if (NoDelayFlag)
    {
        if ((d_channel = open(device, O_RDWR | O_NDELAY)) == -1)
        {
            sprintf(d_error, "Error opening serial device: %s", strerror(errno));
            return;
        }
    }
    else
    {
        if ((d_channel = open(device, O_RDWR)) == -1)
        {
            sprintf(d_error, "Error opening serial device: %s", strerror(errno));
            return;
        }
    }
    /* Set the parameters via ioctl. */

    /* First, get the parameters which can be set in the Preferences */
    switch (BaudRate)
    {
    case 1200:
        SerialSpeed |= B1200;
        break;
    case 2400:
        SerialSpeed |= B2400;
        break;
    case 4800:
        SerialSpeed |= B4800;
        break;
    case 9600:
        SerialSpeed |= B9600;
        break;
    case 19200:
        SerialSpeed |= B19200;
        break;
    case 38400:
        SerialSpeed |= B38400;
        break;
    default:
        sprintf(d_error, "Serial Speed %s not valid", strerror(errno));
        return;
    }

    if (DataBits == 7)
        SerialBits = CS7;
    else
        SerialBits = CS8;

    if (ioctl(d_channel, TCGETA, &TermPar) == -1)
    {
        sprintf(d_error, "Error serial ioctl: %s", strerror(errno));
        return;
    }

    /* change values and flags in term_par struct */
    TermPar.c_oflag &= ~OPOST; /* no character conversion */
    TermPar.c_cc[VMIN] = 1; /* 0 for any setup purposes,> 0 for streaming  */
    TermPar.c_cc[VTIME] = (char)1; /* 1 normally */

    TermPar.c_lflag = 0;
    TermPar.c_oflag = 0;

    TermPar.c_iflag = 0;
    TermPar.c_iflag &= ~IXOFF;

#if defined(__linux__) || defined(__hpux)
    TermPar.c_line = 0;
#else
    TermPar.c_line = LDISC1;
#endif

#ifdef _OLD_TERMIOS
    TermPar.c_cflag = (SerialSpeed | SerialBits | CREAD | CLOCAL);
#else
    TermPar.c_cflag = (SerialBits | CREAD | CLOCAL);
    TermPar.c_ispeed = SerialSpeed;
    TermPar.c_ospeed = SerialSpeed;
#endif

    switch (StopBits)
    {
    case 1:
        TermPar.c_cflag &= ~CSTOPB;
        break;
    default:
        TermPar.c_cflag |= CSTOPB;
        break; /* Default: set CSTOPB */
    };

    switch (Parity)
    {
    case 'N':
        TermPar.c_cflag &= ~PARENB;
        break;
    case 'O':
        TermPar.c_cflag |= (PARENB | PARODD);
        break;
    case 'E':
    {
        TermPar.c_cflag |= PARENB;
        TermPar.c_cflag &= ~PARODD;
    };
    };

    /* Put back values */
    if ((ioctl(d_channel, TCSETAF, &TermPar)) == -1)
    {
        sprintf(d_error, "Error serial ioctl: %s", strerror(errno));
        return;
    }
#else
    (void)device;
    (void)baudrate;
    (void)Parity;
    (void)DataBits;
    (void)StopBits;
#endif
#endif
    return;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SerialCom::~SerialCom()
{
#ifdef WIN32
    if (d_channel)
        CloseHandle(d_channel);
#else
    if (d_channel > 9)
        close(d_channel);
#endif
}

int SerialCom::write(void *data, int bufLen)
{
#ifdef WIN32
    DWORD numWrite = 0;
    if (WriteFile(d_channel, data, bufLen, &numWrite, NULL))
        return numWrite;
    else
        return -1;
#else
    char *charData = (char *)data;
    int retVal = ::write(d_channel, charData, bufLen);
    return retVal;
#endif
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// read a maximum of bufLen bytes into buffer,
//    return number of read bytes
int SerialCom::read(void *data, int bufLen, struct timeval &timeout)

{
    char *charData = (char *)data;
    int bytesRead = 0;
#ifdef WIN32
    DWORD bytesReceived;
    BOOL bResult;
    int bytesToRead;
    long startTime, endTime;

    bytesToRead = bufLen;
    bytesReceived = 0;
    startTime = (long)time(NULL);
    endTime = startTime + 3;
    while (bytesToRead)
    {
        bResult = ReadFile(d_channel, (charData + (bufLen - bytesToRead)), bytesToRead, &bytesReceived, NULL);
        if (bytesReceived)
            bytesToRead -= bytesReceived;
    }

    if (bytesToRead > 0)
    {
        fprintf(stderr, "SERVER: lost bytes (%d read from %d, %d lost)\n",
                bufLen - bytesToRead, bufLen, bytesToRead);
        return (0);
    }
    else
        return (1);

#else

    // wait in loop if nothing there
    size_t bytes;
    //Sind schon bytes da
    int res = ioctl(d_channel, FIONREAD, &bytes);
    if (res == 0 && bytes == 0)
    {
        struct timeval to = timeout; // return empty after timeout
        fd_set fileSet;
        FD_ZERO(&fileSet);
        FD_SET(d_channel, &fileSet);
        select(d_channel + 1, &fileSet, NULL, NULL, &to);
        res = ioctl(d_channel, FIONREAD, &bytes); //wieviel bytes im Buffer
    }
    if (bytes)
    {
        int len = ::read(d_channel, &charData[bytesRead], bufLen - bytesRead);
        if (len > 0)
        {
            bytesRead += len;
        }
    }
#endif
    return bytesRead;
}

int SerialCom::read(void *data, int bufLen, int max_time)
{

    char *charData = (char *)data;
    int bytesRead = 0;
#ifdef WIN32
    DWORD bytesReceived;
    BOOL bResult;
    int bytesToRead;
    long startTime, endTime;

    bytesToRead = bufLen;
    bytesReceived = 0;
    startTime = (long)time(NULL);
    endTime = startTime + 3;
    while (bytesToRead)
    {
        bResult = ReadFile(d_channel, (charData + (bufLen - bytesToRead)), bytesToRead, &bytesReceived, NULL);
        if (bytesReceived)
            bytesToRead -= bytesReceived;
    }

    if (bytesToRead > 0)
    {
        fprintf(stderr, "SERVER: lost bytes (%d read from %d, %d lost)\n",
                bufLen - bytesToRead, bufLen, bytesToRead);
        return (0);
    }
    else
        return (1);

#else
    struct timeval timeout = // return empty after 1 sec
        {
          1, 0
        };
    time_t start_time = time(NULL);

    while (bytesRead < bufLen)
    {

        // wait in loop if nothing there
        size_t bytes;
        //Sind schon bytes da
        int res = ioctl(d_channel, FIONREAD, &bytes);
        if (res == 0 && bytes == 0)
        {
            struct timeval to = timeout; // return empty after 1 sec
            fd_set fileSet;
            FD_ZERO(&fileSet);
            FD_SET(d_channel, &fileSet);
            select(d_channel + 1, &fileSet, NULL, NULL, &to);
            res = ioctl(d_channel, FIONREAD, &bytes); //wieviel bytes im Buffer
        }
        if (bytes)
        {
            int len = ::read(d_channel, &charData[bytesRead], bufLen - bytesRead);
            if (len > 0)
            {
                bytesRead += len;
            }
        }

        if (time(NULL) > start_time + max_time)
        {
            //fprintf(stderr, "\n Timeout : Keine Antwort erhalten !\n");
            return bytesRead;
        }
    }
#endif
    return bytesRead;
}

/// write a maximum of bufLen bytes into buffer,
//    return number of written bytes
//int SerialCom::write(void *data, int bufLen)
//{
//}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// return true if an error occurred
bool SerialCom::isBad() const
{
    return (d_error[0] != '\0');
}

// return error message
const char *SerialCom::errorMessage() const
{
    return d_error;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}
