/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_REGEXP_H
#define COVISE_REGEXP_H

#if defined(_MSC_VER) || defined(MINGW)
#include "regex/regex.h"
#else
#include <regex.h>
#endif

#include "coExport.h"

namespace covise
{

class UTILEXPORT CoviseRegexp
{
    static const int maxmatches_ = 30;
    regmatch_t matches_[maxmatches_];
    regex_t preg_;
    char *line_;

public:
    CoviseRegexp(const char *regexp);
    ~CoviseRegexp();
    bool isMatching(const char *line);
    char *getMatchString(int position);
    int getMatchInt(int position);
    double getMatchFloat(int position);
};
}
#endif
