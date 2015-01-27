/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          fortran.h  -  fortran like I/O
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef _FORTRAN_H_

#define _FORTRAN _H_

#include <fstream>
#include "arrays.h"

int unitopen(int, const char *, int imode = ios::in, bool bApp = false);
int unitwrite(int, const char *, ...);
int unitwrite(int, int, const char *pstr = 0);
int unitwrite(int, bool, const char *pstr = 0);
int unitwrite(int, prec, const char *pstr = 0);
int unitwriteln(int, const char *, ...);
int unitfortranwrite(int, const char *, ...);
char *unitreadln(int, char *pstr = 0);
char readchar(int);
int readint(int, bool bFortran = true);
prec readreal(int, bool bFortran = true);
bool readbool(int, bool bFortran = true);
void unitclose(int);
void unitcloseall(void);
void unitread(int, int &);
void unitread(int, bool &);
void unitread(int, prec &);
void init_stream();

extern char AppPath[];
extern char FilePath[];
extern fstream *fp[];
#endif
