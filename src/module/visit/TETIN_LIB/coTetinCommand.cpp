/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetinCommand.h"
#include <ctype.h>
#include <string.h>
#include <stdio.h>

/////////// read a line from a stream

istream &coTetinCommand::getLine(char *line, int length, istream &str)
{
    *line = '\0';
    if (str.getline(line, length)) // get the line
    {
        line[length - 1] = '\0'; // make sure buffer is terminated

        int actLen = strlen(line) - 1; // remove all trailing blanks
        while ((actLen >= 0) && isspace(line[actLen]))
        {
            line[actLen] = '\0';
            actLen--;
        }
        actLen++;

        // continuation line ?
        if (line[actLen - 1] == '\\')
        {
            getLine(line + actLen - 1, length - actLen, str);
        }

        if (*line == 0 && !str.eof())
            return getLine(line, length, str);
    }
    else
        *line = '\0';

    return str;
}

// read a number of floats as fool-proof as possible from a stream
// requires to have nothing except numbers and delimiters ' ' or ','
// or '\t' or '\n' in the stream ;-)
istream &coTetinCommand::getFloats(float *data, int num, istream &str)
{
    char buffer[4096];
    buffer[0] = '\0';
    char *bPtr = buffer;
    while (num)
    {
        if (!(*bPtr))
        {
            str.getline(buffer, 4096);
            //cout << "--->" << buffer << "<---" << endl;
            if (!str)
                return str;
            bPtr = buffer;
        }

        sscanf(bPtr, "%f", data);
        data++;
        num--;
        while ((*bPtr) && !(((*bPtr) != ',') || isspace(*bPtr)))
            bPtr++;
        while ((*bPtr) && ((*bPtr) != ',') && !isspace(*bPtr))
            bPtr++;
        if (*bPtr)
            bPtr++;
    }

    return str;
}

/// read a 0-terminated string from a concatanation in a char field
char *coTetinCommand::getString(char *&chPtr)
{
    char *res;
    int length = strlen(chPtr) + 1; // first string length incl. 0
    res = strcpy(new char[length], chPtr);
    chPtr += length; // increment pointer behind 0
    return res;
}

/// read a float from a line buffer and advance pointer
float coTetinCommand::readFloat(char *&line)
{
    // skip everythin which cannot be a number...
    while (*line
           && !(isdigit(*line) || (*line == '-') || (*line == '+')))
        line++;

    float val;
    sscanf(line, "%f", &val);

    // skip to first possible separation char
    while (*line && !isspace(*line) && (*line != ',') && (*line != ';'))
        line++;

    // if this was a space, skip all following space
    while (*line && isspace(*line))
        line++;

    // now if this is a separation char, skip it
    if (*line && ((*line == ',') || (*line == ';')))
        line++;

    return val;
}

/// read a int from a line buffer and advance pointer
int coTetinCommand::readInt(char *&line)
{
    // skip everythin which cannot be a number...
    while (*line
           && !(isdigit(*line) || (*line != '-') && (*line != '+')))
        line++;

    int val;
    sscanf(line, "%d", &val);

    // skip to first possible separation char
    while (*line && !isspace(*line) && (*line != ',') && (*line != ';'))
        line++;

    // if this was a space, skip all following space
    while (*line && isspace(*line))
        line++;

    // now if this is a separation char, skip it
    if (*line && ((*line == ',') || (*line == ';')))
        line++;

    return val;
}

/////////// read an option within a line

int coTetinCommand::getOption(const char *line, const char *name, float &value) const
{
    if (!line)
        return 0;
    char *start = strstr(line, name);
    if (start)
    {
        start += strlen(name);
        sscanf(start, "%f", &value);
        return 1;
    }
    return 0;
}

int coTetinCommand::getOption(const char *line, const char *name, int &value) const
{
    if (!line)
        return 0;
    char *start = strstr(line, name);
    if (start)
    {
        start += strlen(name);
        sscanf(start, "%d", &value);
        return 1;
    }
    return 0;
}

int coTetinCommand::getOption(const char *line, const char *name, char *&value) const
{
    if (!line)
        return 0;
    char *start = strstr(line, name);
    if (start)
    {
        start += strlen(name);
        value = new char[strlen(start) + 1]; // this is always too large, but ok..
        sscanf(start, "%s", value);
        return 1;
    }
    else
        return 0;
}

void coTetinCommand::append(coTetinCommand *newNext)
{
    if (!newNext)
        return;
    coTetinCommand *ptrNext;
    coTetinCommand *chain;
    if (!d_next)
    {
        chain = newNext;
        ptrNext = d_next;
        d_next = newNext;
    }
    else
    {
        chain = d_next;
        ptrNext = newNext;
    }
    while (chain->d_next)
        chain = chain->d_next;
    chain->d_next = ptrNext;
}

int coTetinCommand::is(coTetin::Command comm) const
{
    return (d_comm == comm);
}
