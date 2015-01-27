/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SERIAL_COM_H_
#define __SERIAL_COM_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS PsAux
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#include <termios.h>

/**
 * Class for opening and reading/writing serial devices
 * 
 */
class PsAux
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    PsAux(const char *device);

    /// Destructor : virtual in case we derive objects
    virtual ~PsAux();

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
    int d_channel;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    PsAux(const PsAux &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    PsAux &operator=(const PsAux &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    PsAux();
};

#endif
