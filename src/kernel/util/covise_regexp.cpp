/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise_regexp.h"
#include <cstdlib>
#include <cstring>

using namespace covise;

CoviseRegexp::CoviseRegexp(const char *expression)
{
    regcomp(&preg_, expression, REG_EXTENDED);
    int i;
    for (i = 0; i < maxmatches_; i++)
    {
        matches_[i].rm_so = -1;
        matches_[i].rm_eo = -1;
    }
    line_ = NULL;
}

CoviseRegexp::~CoviseRegexp()
{
    if (NULL != line_)
    {
        delete[] line_;
    }
}

bool CoviseRegexp::isMatching(const char *line)
{
    if (NULL != line_)
    {
        delete[] line_;
    }
    size_t len = strlen(line);
    line_ = new char[1 + len];
    strcpy(line_, line);
    int m = regexec(&preg_, line_, maxmatches_, matches_, 0);
    return (0 == m);
}

// get a substring matched by regexec
char *CoviseRegexp::getMatchString(int position)
{
    int len = matches_[position].rm_eo - matches_[position].rm_so;
    char *retVal = new char[1 + len];
    strncpy(retVal, line_ + matches_[position].rm_so, len);
    retVal[len] = '\0';
    return retVal;
}

int CoviseRegexp::getMatchInt(int position)
{
    char *s;
    s = getMatchString(position);
    int retVal = atoi(s);
    delete[] s;
    return retVal;
}

double CoviseRegexp::getMatchFloat(int position)
{
    char *s;
    s = getMatchString(position);
    double retVal = atof(s);
    delete[] s;
    return retVal;
}
