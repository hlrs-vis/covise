/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SERIAL_COM_H_
#define __SERIAL_COM_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS SerialCom
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#include "common.h"
#ifdef WIN32
#elif defined(__hpux)
#include <termio.h>
#else
#include <termios.h>
#endif

/**
 * Class for opening and reading/writing serial devices
 *
 */

namespace covise
{

class UTILEXPORT SerialCom
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    SerialCom(const char *device, int baudrate, int DataBits = 8, int Parity = 'N', int StopBits = 1);

    /// Destructor : virtual in case we derive objects
    virtual ~SerialCom();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// read a maximum of bufLen bytes into buffer,
    //    return number of read bytes
    int read(void *data, int bufLen, int max_time = 5);

    int read(void *data, int bufLen, struct timeval &timeout);

    /// write a maximum of bufLen bytes into buffer,
    //    return number of written bytes
    int write(void *data, int bufLen);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // return true if an error occurred
    bool isBad() const;

    // return error message
    const char *errorMessage() const;

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // Error code from errno if something went bad, 0 otherwise
    char d_error[1024];

// file descriptor for serial channel
#ifdef _WIN32
    HANDLE d_channel;
#else
    int d_channel;
#endif

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    SerialCom(const SerialCom &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    SerialCom &operator=(const SerialCom &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    SerialCom();
};
}
#endif
