/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StepFile.h"
#include <api/coModule.h>
#include <util/coviseCompat.h>

using namespace covise;

StepFile::StepFile(const coModule *mod, const char *filepath)
    : skip(0)
    , delta(MAX_DELTA)
    , finished(0)
    , module(mod)
{
    size_t i, j, found = 0;
    size_t index_nb, index_suffix;
    size_t len_file;
    char nb_string[32];
    char c;

    strcpy(base, filepath);
    len_file = strlen(filepath);
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
            len_nb = strlen(nb_string);
            for (j = 0; j < index_nb; j++)
                preffix[j] = filepath[j];
            preffix[j] = '\0';
        }
        i--;
    }
    if (!found)
    {
        module->sendError("ERROR: The name of the timestep file must contain digits!");
        finished = 1;
    }
    else
    {
        len_nb = strlen(nb_string);
        if (sscanf(nb_string, "%d", &base_number) != 1)
        {
            fprintf(stderr, "StepFile::StepFile: sscanf failed\n");
        }
        file_index = base_number;
    }
}

void StepFile::get_nextpath(char **resultpath)
{
	size_t len_file_index;
	size_t found = 0;
	size_t i, j, k;
    FILE *fd;
    char file_index_string[32], buffer[128];

    *resultpath = NULL;

    if (file_index == base_number)
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
                strcpy(buffer, preffix);
                sprintf(file_index_string, "%d", file_index++);
                len_file_index = strlen(file_index_string);
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

void StepFile::set_delta(int delta_value)
{
    delta = delta_value;
}

void StepFile::getDelta(int *delta_value)
{
    *delta_value = delta;
}

void StepFile::set_skip_value(int skip_value)
{
    skip = skip_value;
}

void StepFile::get_skip_value(int *skip_value)
{
    *skip_value = skip;
}
