/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _UDP__SENDER_H_
#define _UDP__SENDER_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS UDP_Sender
//
// This class @@@
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#include "common.h"
#ifndef _WIN32
#include <netinet/in.h>
#else
#include <winsock2.h>
#endif
#ifndef NULL
#define NULL 0
#endif
/**
 * Class for a very simple UDP server
 *
 */
namespace covise
{

class UTILEXPORT UDP_Sender
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *  Prepare sender to hostname / port
       */
    UDP_Sender(const char *hostname, int port,
               const char *mcastif = NULL, int mcastttl = -1);

    /** Constructor
       *  Prepare sender to hostname/port given as "host:port"
       */
    UDP_Sender(const char *hostPort,
               const char *mcastif = NULL, int mcastttl = -1);

    /// Destructor : virtual in case we derive objects
    virtual ~UDP_Sender();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // send a UDP packet: return #bytes sent, error ret=-1, error string set
    int send(const void *buffer, int numsendBytes);

    // send \0-terminated string
    int send(const char *string);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // return true if bad
    bool isBad();

    // get error string of last error
    const char *errorMessage();

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // address info to send to
    struct sockaddr_in d_address;

    // error buffer: contains non-empty string if bad things happen
    char d_error[1024];

    // Socket number
    int d_socket;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // Try to get IP, use no nameserver if dot-notation
    unsigned long getIP(const char *hostname);

    // set up server data (used by constructors)
    void setup(const char *hostname, int port,
               const char *mcastif = NULL, int mcastttl = -1);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    UDP_Sender(const UDP_Sender &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    UDP_Sender &operator=(const UDP_Sender &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    UDP_Sender();
};
}
#endif
