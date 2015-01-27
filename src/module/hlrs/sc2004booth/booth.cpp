/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include "include/booth.h"

struct sc_booth *AllocBooth(void)
{
    struct sc_booth *booth;

    if ((booth = (struct sc_booth *)calloc(1, sizeof(struct sc_booth))) == NULL)
    {
        fprintf(stderr, "not enough memory for struct sc_booth!");
        return NULL;
    }

    return booth;
}

void FreeBooth(struct sc_booth *booth)
{
    int i;

    for (i = 0; i < booth->nobjects; i++)
    {
        free(booth->cubes[i]);
    }
    free(booth->cubes);

    return;
}

int ReadStartfile(const char *fn, struct sc_booth *booth)
{
    FILE *stream;
    char buf[200];
    char *buf2;
    int i;

    const char *separator = "->";

    if ((stream = fopen(&fn[0], "r")) == NULL)
    {
        fprintf(stderr, "Cannot open '%s'! Does file exist?\n", fn);
        return -1;
    }
    fprintf(stderr, "Reading parameter file %s\n", fn);

    // number of objects
    fgets(buf, 200, stream);
    read_int(buf, &booth->nobjects, separator);

    // alloc cubes
    if ((booth->cubes = (struct cubus **)calloc(booth->nobjects, sizeof(struct cubus *))) == NULL)
    {
        fprintf(stderr, "%s\n", "Not enough space!");
        return -1;
    }
    for (i = 0; i < booth->nobjects; i++)
    {
        if ((booth->cubes[i] = (struct cubus *)calloc(1, sizeof(struct cubus))) == NULL)
        {
            fprintf(stderr, "%s\n", "Not enough space!");
            return -1;
        }
    }

    // totalsize
    fgets(buf, 200, stream);
    read_string(buf, &buf2, separator);
    sscanf(buf2, "%f %f %f", &booth->size[0],
           &booth->size[1],
           &booth->size[2]);

    fgets(buf, 200, stream);
    read_string(buf, &buf2, separator);
    sscanf(buf2, "%f %f %f %f %f %f", &booth->cubes[0]->pos[0],
           &booth->cubes[0]->pos[1],
           &booth->cubes[0]->pos[2],
           &booth->cubes[0]->size[0],
           &booth->cubes[0]->size[1],
           &booth->cubes[0]->size[2]);

    if (booth->nobjects > 1)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &booth->cubes[1]->pos[0],
               &booth->cubes[1]->pos[1],
               &booth->cubes[1]->pos[2],
               &booth->cubes[1]->size[0],
               &booth->cubes[1]->size[1],
               &booth->cubes[1]->size[2]);
    }
    if (booth->nobjects > 2)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &booth->cubes[2]->pos[0],
               &booth->cubes[2]->pos[1],
               &booth->cubes[2]->pos[2],
               &booth->cubes[2]->size[0],
               &booth->cubes[2]->size[1],
               &booth->cubes[2]->size[2]);
    }
    if (booth->nobjects > 3)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &booth->cubes[3]->pos[0],
               &booth->cubes[3]->pos[1],
               &booth->cubes[3]->pos[2],
               &booth->cubes[3]->size[0],
               &booth->cubes[3]->size[1],
               &booth->cubes[3]->size[2]);
    }
    if (booth->nobjects > 4)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &booth->cubes[4]->pos[0],
               &booth->cubes[4]->pos[1],
               &booth->cubes[4]->pos[2],
               &booth->cubes[4]->size[0],
               &booth->cubes[4]->size[1],
               &booth->cubes[4]->size[2]);
    }

    fclose(stream);

    return (0);
}
