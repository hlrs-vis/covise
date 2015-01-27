/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*******************************************************
 * FILE 'utils.h'
 *******************************************************
 * Utility functions for anidysim
 */

#ifndef __UTILS_H
#define __UTILS_H

#include <cstdio>

#include "primitive.h"

void TextFormat(std::FILE *stream, const int width, const char linebegin[],
                char text[]);

int CountLines(std::FILE *stream, const char c);
int CountColumns(std::FILE *stream, const char c);
int NoComment(std::FILE *stream, const char c);

void glMakeSimpleFont();
void glPrintString(const char s[]);

#endif // __UTILS_H
