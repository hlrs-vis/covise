/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Token.h"
#include <string.h>

using namespace covise;

Token::~Token()
{
    delete[] string;
}

Token::Token(const char *s)
{
    size_t len = strlen(s);
    string = new char[1 + len];
    strcpy(string, s);
    position = string;
}

char *Token::next()
{
    char *curpos = position;
    while (*curpos != '\n' && *curpos != '\0')
    {
        curpos++;
    }
    size_t len = curpos - position;
    char *retVal = new char[1 + len];
    strncpy(retVal, position, len);
    retVal[len] = '\0';
    position = curpos;
    if (*position != '\0')
    {
        ++position;
    }
    return retVal;
}

char *Token::nextSpace()
{
    char *curpos = position;
    while (*curpos != '\n' && *curpos != ' ' && *curpos != '\0')
    {
        curpos++;
    }
    size_t len = curpos - position;
    char *retVal = new char[1 + len];
    strncpy(retVal, position, len);
    retVal[len] = '\0';
    position = curpos;
    if (*position != '\0')
    {
        ++position;
    }
    return retVal;
}
