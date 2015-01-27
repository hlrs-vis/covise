/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __PORSCHE_H
#define __PORSCHE_H

#include <iostream>

#include <util/common.h>
#include <util/coExport.h>

#define MAXFLOATS 100

#ifndef WIN32
#include <termios.h>
#include <sys/stat.h> /* open */
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <termio.h> /* tcsetattr */
#include <limits.h> /* sginap */
#endif

class PLUGINEXPORT Porsche
{
private:
#ifdef WIN32
    HANDLE serialDev;
    HWND AWindow;
#else
    int serialDev;
    struct termio oldconfig;
#endif
    bool isOpen;
    char *devName;

public:
    Porsche(const char *devName, int baudrate);

    ~Porsche();
    bool deviceOpen()
    {
        return isOpen;
    };

    bool readBlock();
    int readChars(unsigned char *buf, int n);
    int writeChars(const unsigned char *buf, int n);
    unsigned char numFloats;
    float floats[MAXFLOATS];
    bool bufferempty;
};
#endif
