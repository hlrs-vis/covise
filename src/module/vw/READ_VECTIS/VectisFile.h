/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VECTISFILE_H
#define VECTISFILE_H

#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

class VectisFile
{
    char *name;
    int hdl;

public:
    VectisFile(char *n);
    char *get_filename()
    {
        return name;
    };
    int read_record(int &len, char **data);
    int read_record(int len, char *data);
    int read_record(int &data);
    int read_record(float &data);
    int read_textrecord(char **data);
    int skip_record();
    int set_lseek()
    {
        return lseek(hdl, 0, SEEK_CUR);
    };
    int goto_lseek(int ls)
    {
        return lseek(hdl, ls, SEEK_SET);
    };
};
#endif
