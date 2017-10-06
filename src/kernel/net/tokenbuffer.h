/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TokenBuffer_H
#define TokenBuffer_H

#include <string.h>
#include <stdio.h>
#include <iostream>

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

namespace covise
{

class Message;

class NETEXPORT TokenBuffer // class for tokens
{
private:
    TokenBuffer(const TokenBuffer &other) = delete;
    TokenBuffer &operator=(const TokenBuffer &other) = delete;

    int buflen; // number of allocated bytes
    int length; // number of used bytes
    char *data; // pointer to the tokens
    char *currdata; // pointer to the tokens
    bool networkByteOrder;

    void incbuf(int size = 100);

public:
    TokenBuffer(bool nbo = false)
    {
        buflen = length = 0;
        data = currdata = NULL;
        networkByteOrder = nbo;
    }
    TokenBuffer(int al, bool nbo = false)
    {
        buflen = al;
        length = 0;
        data = currdata = new char[al];
        networkByteOrder = nbo;
    }
    virtual ~TokenBuffer();
    void delete_data();
    TokenBuffer(Message *msg, bool nbo = false);
    TokenBuffer(const char *dat, int len, bool nbo = false);

    const char *getBinary(int n)
    {
        const char *c = currdata;
        currdata += n;
        return c;
    }

    void addBinary(const char *buf, int n)
    {
        if (buflen < length + n + 1)
            incbuf(n + 40);
        memcpy(currdata, buf, n);
        currdata += n;
        length += n;
    }
    const char *allocBinary(int n)
    {
        if (buflen < length + n + 1)
            incbuf(n + 40);
        const char *buf = currdata;
        currdata += n;
        length += n;
        return buf;
    }

    TokenBuffer &operator<<(const uint64_t i);
#ifndef WIN32 // it does not work on win32 as size_t == int
//TokenBuffer& operator << (const size_t s){return (*this<<(uint64_t)s);}
#endif
    TokenBuffer &operator<<(const uint32_t i);
    TokenBuffer &operator<<(const int i)
    {
        return (*this << (uint32_t)i);
    }
    TokenBuffer &operator<<(const std::string &s);
    TokenBuffer &operator<<(const char c)
    {
        if (buflen < length + 2)
            incbuf();
        *currdata = c;
        currdata++;
        length++;
        return (*this);
    }
    TokenBuffer &operator<<(const float f);
    TokenBuffer &operator<<(const double f);
    TokenBuffer &operator<<(const char *c)
    {
        int l = int(strlen(c) + 1);
        if (buflen < length + l + 1)
            incbuf(l * 10);
        strcpy(currdata, c);
        currdata += l;
        length += l;
        return (*this);
    }
    TokenBuffer &operator<<(const TokenBuffer *t);
    TokenBuffer &operator<<(const TokenBuffer &t)
    {
        if (buflen < length + t.get_length() + 1)
            incbuf(t.get_length() * 4);
        memcpy(currdata, t.get_data(), t.get_length());
        currdata += t.get_length();
        length += t.get_length();
        return (*this);
    }
    TokenBuffer &operator>>(uint64_t &i);
#ifndef WIN32 // it does not work on win32 as size_t == int
//TokenBuffer& operator >> (size_t &s){uint64_t i; *this>>i; s=i; return *this; }
#endif
    TokenBuffer &operator>>(uint32_t &i);
    TokenBuffer &operator>>(int &i)
    {
        return (*this >> *((uint32_t *)&i));
    }
    TokenBuffer &operator>>(char &c)
    {
        c = *(char *)currdata;
        currdata++;
        return (*this);
    }
    TokenBuffer &operator>>(unsigned char &c)
    {
        c = *(unsigned char *)currdata;
        currdata++;
        return (*this);
    }
    TokenBuffer &operator>>(float &f);
    TokenBuffer &operator>>(double &f);
    TokenBuffer &operator>>(std::string &s);
    TokenBuffer &operator>>(char *&c)
    {
        char *end = data + length - 1;
        c = currdata;
        while (*currdata)
        {
            currdata++;
            if (currdata > end)
            {
                std::cerr << "string not terminated within range" << std::endl;
                *end = '\0';
                return (*this);
            }
        }
        currdata++;
        return (*this);
    }

    uint32_t get_int_token();
    char get_char_token()
    {
        char ret = *(char *)currdata;
        currdata++;
        return (ret);
    };
    float get_float_token();
    char *get_charp_token()
    {
        char *ret = currdata;
        while (*currdata)
            currdata++;
        currdata++;
        return (ret);
    };
    int get_length() const
    {
        return (length);
    };
    const char *get_data() const
    {
        return (data);
    };
    void reset()
    {
        currdata = data;
        length = 0;
        if (data)
            data[0] = 0;
    };

    void rewind();
};
}
#endif
