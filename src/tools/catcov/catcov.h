/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CATCOV_H
#define _CATCOV_H
/**************************************************************************\ 
 **                                                      C)2001 Vircinity  **
 **                                                                        **
 ** Description:                                                           **
 **           Catenate input static covise files into a single dynamic one **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                        Sergio Leseduarte                               **
 **                        Vircinity IT GmbH                               **
 **                         Nobel Str. 15                                  **
 **                        70550 Stuttgart                                 **
 **                                                                        **
 ** Date:  03.05.2001                                                      **
\**************************************************************************/

#include <iostream>

#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

// #include "ApplInterface.h"
// #include "coString.h"

class CatCov
{
    int fileOutN_; // file descriptor for output
    int noTimeSteps_;
    int TimeStep_;
    int BuffSize_;
    char *Buffer_;
    int not_ok_;
    //   coString progname_;
    char **pathname_;
    int Magic_;
    enum
    {
        IGNORANCE,
        NO_INFO,
        IS_BIG_ENDIAN,
        IS_LITTLE_ENDIAN,
        NO_INFO_IS_BIG_ENDIAN,
        NO_INFO_IS_LITTLE_ENDIAN,
        IS_BIG_ENDIAN_IS_LITTLE_ENDIAN // This latter state implies a contradiction
    };
    enum
    {
        INPUT_IS_LITTLE_ENDIAN,
        INPUT_IS_BIG_ENDIAN
    } InputType_;

    void WriteToOutFile(int);
    void swap_int(int &);

public:
    CatCov(int, char *argv[]);
    ~CatCov()
    {
        delete[] Buffer_;
        delete[] pathname_;
    }
    int Diagnose(char *argv[]); // {return 0;}  // Correct when handling options
    void WriteMagic();
    void WriteSetHeader();
    int how_many()
    {
        return noTimeSteps_;
    }
    void DumpFile(int);
    void WriteTimeAttrib();
};
#endif // _CATCOV_H
