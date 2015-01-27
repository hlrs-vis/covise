/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coStepFile.h"
#include "coModule.h"
using namespace covise;

coStepFile::~coStepFile()
{
    delete[] prefix;
    delete[] suffix;
    delete[] base;
}

coStepFile::coStepFile(const char *filepath)
    : skip(0)
    , delta(MAX_DELTA)
    , finished(0)
{
    int i, j, found = 0;
    int index_nb, index_suffix;
    int len_file;
    char nb_string[32];
    char c;
    singleFile = false;
    int lenStr = 1024 + (int)strlen(filepath);
    base = new char[lenStr];
    prefix = new char[lenStr];
    suffix = new char[lenStr];

    strcpy(base, filepath);
    len_file = (int)strlen(filepath);
    i = len_file;
    while (i >= 0 && !found)
    {
        c = filepath[i];
        if (isdigit(c))
        {
            found = 1;
            index_suffix = i + 1;
            for (j = index_suffix; j <= len_file; j++)
                suffix[j - index_suffix] = filepath[j];
            index_nb = index_suffix;
            while (isdigit(c))
                c = base[(index_nb--) - 2];
            for (j = index_nb; j < index_suffix; j++)
                nb_string[j - index_nb] = filepath[j];
            nb_string[j - index_nb] = '\0';
            len_nb = (int)strlen(nb_string);
            for (j = 0; j < index_nb; j++)
                prefix[j] = filepath[j];
            prefix[j] = '\0';
        }
        i--;
    }
    if (!found)
    {
        singleFile = true;
    }
    else
    {
        len_nb = (int)strlen(nb_string);
        size_t retval;
        retval = sscanf(nb_string, "%d", &base_number);
        if (retval != 1)
        {
            std::cerr << "coStepFile::coStepFile: sscanf failed" << std::endl;
            return;
        }
        file_index = base_number;
    }
}

void coStepFile::get_nextpath(char **resultpath)
{
    int len_file_index, found = 0;
    int i, j, k;
    FILE *fd;
    char file_index_string[32], buffer[128];

    *resultpath = NULL;
    if (finished)
        return;
    if (singleFile)
    {
        *resultpath = new char[strlen(base) + 1];
        strcpy(*resultpath, base);
        // we're finished now!
        finished = 1;
    }
    else if (file_index == base_number)
    {
        *resultpath = new char[strlen(base) + 1];
        strcpy(*resultpath, base);
        file_index = base_number + 1;
    }
    else
    {
        i = 0;
        while (i <= skip && !finished)
        {
            j = 0;
            found = 0;
            while (j < delta && !found)
            {
                strcpy(buffer, prefix);
                sprintf(file_index_string, "%d", file_index++);
                len_file_index = (int)strlen(file_index_string);
                for (k = 0; k < len_nb - len_file_index; k++)
                    strcat(buffer, "0");
                strcat(buffer, file_index_string);
                strcat(buffer, suffix);
                fd = fopen(buffer, "r");
                if (fd)
                {
                    fclose(fd);
                    found = 1;
                }
                j++;
            }
            if (found)
                i++;
            else
                finished = 1;
        }

        if (!finished)
        {
            *resultpath = new char[strlen(buffer) + 1];
            strcpy(*resultpath, buffer);
        }
    }
}

void coStepFile::set_delta(int delta_value)
{
    delta = delta_value;
}

void coStepFile::getDelta(int *delta_value)
{
    *delta_value = delta;
}

void coStepFile::set_skip_value(int skip_value)
{
    skip = skip_value;
}

void coStepFile::get_skip_value(int *skip_value)
{
    *skip_value = skip;
}
