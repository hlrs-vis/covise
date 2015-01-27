/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SCANNER_H_INCLUDED
#define _SCANNER_H_INCLUDED

#include <FlexLexer.h> // provides yyFlexLexer interface

#include <util/coviseCompat.h>

class Scanner : public yyFlexLexer
{
private:
    char *hostName;

public:
    int getLineNo();
    Scanner(istream *arg_yyin = 0, ostream *arg_yyout = 0)
        : yyFlexLexer(arg_yyin, arg_yyout)
    {
        hostName = (char *)"default";
    }

    const char *getHostname()
    {
        return hostName;
    }
    void setHostName(const char *name);
};
#endif
