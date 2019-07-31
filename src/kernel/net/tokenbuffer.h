/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TokenBuffer_H
#define TokenBuffer_H

#include "dataHandle.h"

#include <string.h>
#include <stdio.h>
#include <iostream>
#include <string>


#include <util/coExport.h>
#include <util/byteswap.h>

#ifdef _WIN64
#define __WORDSIZE 64
#endif

/*
 $Log:  $
 * Revision 1.4  1994/03/23  18:07:03  zrf30125
 * Modifications for multiple Shared Memory segments have been finished
 * (not yet for Cray)
 *
 * Revision 1.3  93/10/11  09:22:19  zrhk0125
 * new types DM_CONTACT_DM and APP_CONTACT_DM included
 *
 * Revision 1.2  93/10/08  19:18:06  zrhk0125
 * data type sizes introduced
* some fixed type sizes with sizeof calls replaced
 *
 * Revision 1.1  93/09/25  20:47:03  zrhk0125
 * Initial revision
 *
 */

/***********************************************************************\
 **                                                                     **
 **   Message classes                              Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic message structure as well as ways to     **
 **                  initialize messages easily are provided.           **
 **                  Subclasses for special types of messages           **
 **                  can be introduced.                                 **
 **                                                                     **
 **   Classes      : Message, ShmMessage                                **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                        HOSTID                                             **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  15.04.93  Ver 1.1 new Messages and type added      **
 **                                    sender and send_type added       **
 **                                                                     **
 **                                                                     **
\***********************************************************************/
namespace vrb
{
	class UdpMessage;
}
namespace covise
{

class MessageBase;
class DataHandle;

class NETEXPORT TokenBuffer// class for tokens
{
private:
    enum Types
    {
        TbBool = 7,
        TbInt64,
        TbInt32,
        TbFloat,
        TbDouble,
        TbString,
        TbChar,
        TbTB, //TokenBuffer or DataHandle
        TbBinary,
    };
    DataHandle data;
    void puttype(Types t);
    bool checktype(Types t);

    bool debug = false;
    int buflen = 0; // number of allocated bytes

    char *currdata = nullptr; // pointer to the tokens
    bool networkByteOrder = false;

    void incbuf(int size = 100);

public:
    TokenBuffer();
    explicit TokenBuffer(bool nbo);
    //creates a TokenBuffer with allocated memory
    TokenBuffer(int al, bool nbo = false);

    TokenBuffer(const MessageBase *msg, bool nbo = false);
    TokenBuffer(const DataHandle& dh, bool nbo = false);
    TokenBuffer(const char *dat, int len, bool nbo = false);
    virtual ~TokenBuffer();
    TokenBuffer &operator=(const TokenBuffer &other);

    const DataHandle& getData();
    const char *getBinary(int n);
    void addBinary(const char *buf, int n);
    const char *allocBinary(int n);

    TokenBuffer &operator<<(const bool b);
    TokenBuffer &operator<<(const uint64_t i);
#ifndef WIN32 // it does not work on win32 as size_t == int
    //TokenBuffer& operator << (const size_t s){return (*this<<(uint64_t)s);}
#endif
    TokenBuffer &operator<<(const uint32_t i);
    TokenBuffer &operator<<(const int i);
    TokenBuffer &operator<<(const std::string &s);
    TokenBuffer &operator<<(const char c);
    TokenBuffer &operator<<(const float f);
    TokenBuffer &operator<<(const double f);
    TokenBuffer &operator<<(const char *c);
    TokenBuffer& operator<<(const DataHandle& d);
    TokenBuffer &operator<<(const TokenBuffer &t);

    TokenBuffer &operator>>(bool &b);
    TokenBuffer &operator>>(uint64_t &i);
#ifndef WIN32 // it does not work on win32 as size_t == int
    //TokenBuffer& operator >> (size_t &s){uint64_t i; *this>>i; s=i; return *this; }
#endif
    TokenBuffer &operator>>(uint32_t &i);
    TokenBuffer &operator>>(int &i);
    TokenBuffer &operator>>(char &c);
    TokenBuffer &operator>>(unsigned char &c);
    TokenBuffer &operator>>(float &f);
    TokenBuffer &operator>>(double &f);
    TokenBuffer &operator>>(std::string &s);
    TokenBuffer &operator>>(char *&c);
    TokenBuffer& operator>>(DataHandle& d);
    TokenBuffer &operator>>(TokenBuffer &tb);
    uint32_t get_int_token();
    char get_char_token();;
    float get_float_token();
    char *get_charp_token();;

    void reset();
    void rewind();
};
}

#endif
