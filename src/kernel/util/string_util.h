/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STRIP_H
#define STRIP_H

#include <string>
#include <vector>
#include "coExport.h"

UTILEXPORT std::string strip(const std::string &str);
UTILEXPORT std::string strip(const char *ch);
UTILEXPORT std::string replace(const std::string &where, const char *what, const char *with, int times = 1);
UTILEXPORT int isIntNumber(const std::string &str);

UTILEXPORT std::vector<std::string> split(const std::string &str, char delimiter = ' ');
UTILEXPORT std::string toLower(const std::string &str);

#endif
