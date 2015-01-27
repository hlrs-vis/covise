/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          error.h  -  error handling
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __ERROR_H_

#define __ERROR_H_

#define ERR_GENERAL -1
#define ERR_ABORTED -2

void ErrorFunction(char *, ...);
void WarningFunction(char *, ...);

// ***************************************************************************
// error class
// ***************************************************************************

class TException
{
public:
    TException(char *pstr = "Unbekannter Fehler", ...);

    void Display();

    char Description[100];
};
#endif
