/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FORTRAN_H_
#define FORTRAN_H_
class FortranData
{
    static FILE *uffFile;

public:
    static void setFile(FILE *file);

    static unsigned int ReadFortranDataFormat(const char *format, ...); //read a line of a dataset in given format
    static void WriteFortranDataFormat(const char *format, ...);

    FortranData(void);
    ~FortranData(void);
};
#endif
