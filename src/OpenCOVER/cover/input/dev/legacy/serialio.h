/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SERIALIO_H
#define SERIALIO_H

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#endif

#ifdef __APPLE__
#define TCGETA TIOCGETA
#define TCSETA TIOCSETA
#define TCSETAF TIOCSETAF
#endif

#endif
