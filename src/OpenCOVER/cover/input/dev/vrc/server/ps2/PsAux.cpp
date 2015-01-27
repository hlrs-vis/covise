/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS PsAux
//
// This class is a primitive Interface to serial connections
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.1 non-blocking I/O with select
//

#include "PsAux.h"

#include <covise/covise.h>
#include <sys/types.h>
//#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <unistd.h>

#include <sys/ioctl.h>

#ifdef __linux__
#ifndef _OLD_TERMIOS
#define _OLD_TERMIOS
#endif
#include <termios.h> /* tcsetattr */
#include <termio.h> /* tcsetattr */
#endif

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PsAux::PsAux(const char *device)
{
    // no errors yey
    d_error[0] = '\0';
    if ((d_channel = open(device, O_RDWR)) == -1)
    {
        sprintf(d_error, "Error opening PS2 device: %s", strerror(errno));
        return;
    }
    return;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PsAux::~PsAux()
{
    if (d_channel > 9)
        close(d_channel);
}

int PsAux::write(void *data, int bufLen)
{
    char *charData = (char *)data;
    int retVal = ::write(d_channel, charData, bufLen);
    return retVal;
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// read a maximum of bufLen bytes into buffer,
//    return number of read bytes
int PsAux::read(void *data, int bufLen, struct timeval &timeout)
{

    char *charData = (char *)data;
    int bytesRead = 0;

    cerr << "1IN read" << endl;
    // wait in loop if nothing there
    size_t bytes;
    //Sind schon bytes da
    int res = ioctl(d_channel, FIONREAD, &bytes);
    if (res == 0 && bytes == 0)
    {
        cerr << "2IN read" << endl;
        struct timeval to = timeout; // return empty after timeout
        fd_set fileSet;
        FD_ZERO(&fileSet);
        FD_SET(d_channel, &fileSet);
        select(d_channel + 1, &fileSet, NULL, NULL, &to);
        res = ioctl(d_channel, FIONREAD, &bytes); //wieviel bytes im Buffer
        cerr << "BYTES is" << bytes << endl;
    }
    if (bytes)
    {
        cerr << "3IN read" << endl;
        int len = ::read(d_channel, &charData[bytesRead], bufLen - bytesRead);
        if (len > 0)
        {
            bytesRead += len;
        }
    }
    cerr << "4IN read; bytesRead is" << bytesRead << endl;
    return bytesRead;
}

int PsAux::read(void *data, int bufLen, int max_time)
{
    struct timeval timeout = { 1, 0 }; // return empty after 1 sec
    char *charData = (char *)data;
    int bytesRead = 0;
    time_t start_time = time(NULL);
    ///TRY
    bytesRead = ::read(d_channel, charData, bufLen);

    return bufLen;
    ///TRY
    while (bytesRead < bufLen)
    {

        // wait in loop if nothing there
        size_t bytes = 0;
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
            // fprintf(stderr, "\n Timeout : Keine Antwort erhalten !\n");
            return bytesRead;
        }
    }
    return bytesRead;
}

/// write a maximum of bufLen bytes into buffer,
//    return number of written bytes
//int PsAux::write(void *data, int bufLen)
//{
//}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// return true if an error occurred
bool PsAux::isBad() const
{
    return (d_error[0] != '\0');
}

// return error message
const char *PsAux::errorMessage() const
{
    return d_error;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
PsAux::PsAux(const PsAux &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
PsAux &PsAux::operator=(const PsAux &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
PsAux::PsAux()
{
    assert(0);
}
