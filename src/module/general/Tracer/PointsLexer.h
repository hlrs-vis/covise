/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POINT_FLEXER_H_
#define _POINT_FLEXER_H_

// include standard Flex header file, but use First as prefix, not yy.
#ifndef yyFlexLexer
#define yyFlexLexer PointsFlexLexer
#include <FlexLexer.h>
#undef yyFlexLexer
#endif

#define register

#include "PointsParser.h"

// define new class
class PointsLexer : public PointsFlexLexer
{
public:
    // new constructor
    PointsLexer(istream *pDesc)
        : PointsFlexLexer(pDesc)
    {
    }

    // new scanner method
    int scan(TokenType *token);
};
#endif
