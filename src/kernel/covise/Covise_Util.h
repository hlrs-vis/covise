/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_UTIL_H
#define COVISE_UTIL_H

#include "covise.h"

namespace covise
{

class CharNum
{
    char *buf;
    int num;

public:
    CharNum()
    {
        num = 0;
        buf = NULL;
    };
    CharNum(int n)
    {
        buf = new char[50];
        sprintf(buf, "%d", n);
        num = n;
    };
    CharNum(float n)
    {
        buf = new char[50];
        sprintf(buf, "%f", n);
        num = (int)n;
    };
    ~CharNum()
    {
        delete[] buf;
    };
    operator char *()
    {
        return (buf);
    };
    operator int()
    {
        return (num);
    };
};

class CharBuffer
{
    char *buf;
    int len;

public:
    int cur_len;
    CharBuffer()
    {
        cur_len = 0;
        len = 0;
        buf = NULL;
    };
    CharBuffer(CharBuffer *obuf)
    {
        cur_len = obuf->cur_len;
        len = cur_len + 1;
        buf = new char[len];
        strcpy(buf, obuf->getbuf());
    };
    CharBuffer(int def)
    {
        cur_len = 0;
        len = def;
        buf = new char[len];
        if (len)
            *buf = '\0';
    };
    ~CharBuffer()
    {
        delete[] buf;
    };
    char *return_data()
    {
        char *tmp = buf;
        buf = nullptr;
        cur_len = 0;
        len = 0;
        return (tmp);
    };
    int strlen()
    {
        return (cur_len);
    };
    void operator+=(const char *const s)
    {
        int l = (int)::strlen(s);
        if (cur_len + l >= len)
        {
            len += l * 10;
            char *nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        strcpy(buf + cur_len, s);
        cur_len += l;
    };
    void operator+=(char c)
    {
        if (cur_len + 1 >= len)
        {
            len += 100;
            char *nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        buf[cur_len] = c;
        cur_len++;
        buf[cur_len] = 0;
    };
    void operator+=(int n)
    {
        CharNum s(n);
        int l = (int)::strlen(s);
        if (cur_len + l >= len)
        {
            len += l * 10;
            char *nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        strcpy(buf + cur_len, s);
        cur_len += l;
    };
    void operator+=(float n)
    {
        CharNum s(n);
        int l = (int)::strlen(s);
        if (cur_len + l >= len)
        {
            len += l * 10;
            char *nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        strcpy(buf + cur_len, s);
        cur_len += l;
    };
    operator const char *() const
    {
        return (buf);
    };
    const char *getbuf()
    {
        return (buf);
    };
};
}
#endif
