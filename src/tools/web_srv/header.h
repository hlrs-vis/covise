/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <unistd.h>

/***********************************************************************\ 
 **                                                                     **
 **   Header classes Routines                     Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic message structure as well as ways to     **
 **                  initialize messages easily are provided.           **
 **                  Subclasses for special types of messages           **
 **                  can be introduced.                                 **
 **                                                                     **
 **   Classes      : Header, UnknownH                                   **
 **                                                                     **
 **   Copyright (C) 2001     by                  **
 **                                              **
 **                                              **
 **                                              **
 **                                                                     **
 **                                                                     **
 **   Author       :                                   **
 **                                                                     **
 **   History      :                                                    **
 **                                                  **
 **                    **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

// IDs for all header types that go between processes are fixed here

enum header_type
{
    EMPTY = 0, //  0
    CACHE_CONTROL, //  1  // "Cache-Control" --- general-headers
    CONNECTION, //  2  // "Connection"
    DATE, //  3  // "Date"
    PRAGMA, //  4  // "Pragma"
    TRAILER, //  5  // "Trailer"
    TRANSFER_ENCODING, //  6  // "Transfer-Encoding"
    UPGRADE, //  7  // "Upgrade"
    VIA, //  8  // "Via"
    WARNING, //  9  // "Warning"
    ACCEPT, // 10  // "Accept"        --- request-headers
    ACCEPT_CHARSET, // 11  // "Accept-Charset"
    ACCEPT_ENCODING, // 12  // "Accept-Encoding"
    ACCEPT_LANGUAGE, // 13  // "Accept-Language"
    AUTHORIZATION, // 14  // "Authorization"
    EXPECT, // 15  // "Expect"
    FROM, // 16  // "From"
    HOST, // 17  // "Host"
    IF_MATCH, // 18  // "If-Match"
    IF_MODIFIED_SINCE, // 19  // "If-Modified-Since"
    IF_NONE_MATCH, // 20  // "If-None-Match"
    IF_RANGE, // 21  // "If-Range"
    IF_UNMODIFIED_SINCE, // 22  // "If-Unmodified-Since"
    MAX_FORWARDS, // 23  // "Max-Forwards"
    PROXY_AUTHORIZATION, // 24  // "Proxy-Authorization"
    RANGE, // 25  // "Range"
    REFERER, // 26  // "Referer"
    TE, // 27  // "TE"
    USER_AGENT, // 28  // "User-Agent"
    ACCEPT_RANGES, // 29  // "Accept-Ranges" --- response headers
    AGE, // 30  // "Age"
    ETAG, // 31  // "ETag"
    LOCATION, // 32  // "Location"
    PROXY_AUTHENTICATE, // 33  // "Proxy-Authenticate"
    RETRY_AFTER, // 34  // "Retry-After"
    SERVER, // 35  // "Server"
    VARY, // 36  // "Vary"
    WWW_AUTHENTICATE, // 37  // "WWW-Authenticate"
    ALLOW, // 38  // "Allow"         --- entity headers
    CONTENT_ENCODING, // 39  // "Content-Encoding"
    CONTENT_LANGUAGE, // 40  // "Content-Language"
    CONTENT_LENGTH, // 41  // "Content-Length"
    CONTENT_LOCATION, // 42  // "Content-Location"
    CONTENT_MD5, // 43  // "Content-MD5"
    CONTENT_RANGE, // 44  // "Content-Range"
    CONTENT_TYPE, // 45  // "Content-Type"
    EXPIRES, // 46  // "Expires"
    LAST_MODIFIED, // 47  // "Last-Modified"
    EXTENSION, // 48  // EXTENSION HEADER
    UNKNOWN, // 49  // UNKNOWN HEADER
    MAX_HEADERS // 50
};

#ifdef DEFINE_HTTP_HEADERS
char *header_array[] = {
    "EMPTY", //  0
    "Cache-Control", //  1  // "Cache-Control" --- general-headers
    "Connection", //  2  // "Connection"
    "Date", //  3  // "Date"
    "Pragma", //  4  // "Pragma"
    "Trailer", //  5  // "Trailer"
    "Transfer-Encoding", //  6  // "Transfer-Encoding"
    "Upgrade", //  7  // "Upgrade"
    "Via", //  8  // "Via"
    "Warning", //  9  // "Warning"
    "Accept", // 10  // "Accept"        --- request-headers
    "Accept-Charset", // 11  // "Accept-Charset"
    "Accept-Encoding", // 12  // "Accept-Encoding"
    "Accept-Language", // 13  // "Accept-Language"
    "Authorization", // 14  // "Authorization"
    "Expect", // 15  // "Expect"
    "From", // 16  // "From"
    "Host", // 17  // "Host"
    "If-Match", // 18  // "If-Match"
    "If-Modified-Since", // 19  // "If-Modified-Since"
    "If-None-Match", // 20  // "If-None-Match"
    "If-Range", // 21  // "If-Range"
    "If-Unmodified-Since", // 22  // "If-Unmodified-Since"
    "Max-Forwards", // 23  // "Max-Forwards"
    "Proxy-Authorization", // 24  // "Proxy-Authorization"
    "Range", // 25  // "Range"
    "Referer", // 26  // "Referer"
    "TE", // 27  // "TE"
    "User-Agent", // 28  // "User-Agent"
    "Accept-Ranges", // 29  // "Accept-Ranges" --- response headers
    "Age", // 30  // "Age"
    "ETag", // 31  // "ETag"
    "Location", // 32  // "Location"
    "Proxy-Authenticate", // 33  // "Proxy-Authenticate"
    "Retry-After", // 34  // "Retry-After"
    "Server", // 35  // "Server"
    "Vary", // 36  // "Vary"
    "WWW-Authenticate", // 37  // "WWW-Authenticate"
    "Allow", // 38  // "Allow"         --- entity headers
    "Content-Encoding", // 39  // "Content-Encoding"
    "Content-Language", // 40  // "Content-Language"
    "Content-Length", // 41  // "Content-Length"
    "Content-Location", // 42  // "Content-Location"
    "Content-MD5", // 43  // "Content-MD5
    "Content-Range", // 44  // "Content-Range"
    "Content-Type", // 45  // "Content-Type"
    "Expires", // 46  // "Expires"
    "Last-Modified", // 47  // "Last-Modified"
    "EXTENSION", // 48  // EXTENSION HEADER
    "UNKNOWN", // 49  // UNKNOWN HEADER
    "MAX_HEADERS" // 50
};
#else
extern char *header_array[];
#endif

class Header // generic class for headers
{

public:
    char *m_name;
    char *m_value;

    Header()
    {
        m_name = NULL;
        m_value = NULL;
    };
    Header(char *name);
    Header(char *name, char *value);
    ~Header();
    void Set(char *name, char *value);

    void print(void);
};
#endif
