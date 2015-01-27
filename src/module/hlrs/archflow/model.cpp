/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"

struct arch_model *AllocModel(void)
{
    struct arch_model *model;

    if ((model = (struct arch_model *)calloc(1, sizeof(struct arch_model))) == NULL)
    {
        fprintf(stderr, "not enough memory for struct arch_model!");
        return NULL;
    }

    return model;
}

void FreeModel(struct arch_model *model)
{
    int i;

    for (i = 0; i < model->nobjects; i++)
    {
        free(model->cubes[i]);
    }
    free(model->cubes);

    return;
}

int ReadStartfile(const char *fn, struct arch_model *model)
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
    read_int(buf, &model->nobjects, separator);

    // alloc cubes
    if ((model->cubes = (struct cubus **)calloc(model->nobjects, sizeof(struct cubus *))) == NULL)
    {
        fprintf(stderr, "%s\n", "Not enough space!");
        return -1;
    }
    for (i = 0; i < model->nobjects; i++)
    {
        if ((model->cubes[i] = (struct cubus *)calloc(1, sizeof(struct cubus))) == NULL)
        {
            fprintf(stderr, "%s\n", "Not enough space!");
            return -1;
        }
    }

    // totalsize
    fgets(buf, 200, stream);
    read_string(buf, &buf2, separator);
    sscanf(buf2, "%f %f %f", &model->size[0],
           &model->size[1],
           &model->size[2]);

    fgets(buf, 200, stream);
    read_string(buf, &buf2, separator);
    sscanf(buf2, "%f %f %f %f %f %f", &model->cubes[0]->pos[0],
           &model->cubes[0]->pos[1],
           &model->cubes[0]->pos[2],
           &model->cubes[0]->size[0],
           &model->cubes[0]->size[1],
           &model->cubes[0]->size[2]);

    if (model->nobjects > 1)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &model->cubes[1]->pos[0],
               &model->cubes[1]->pos[1],
               &model->cubes[1]->pos[2],
               &model->cubes[1]->size[0],
               &model->cubes[1]->size[1],
               &model->cubes[1]->size[2]);
    }
    if (model->nobjects > 2)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &model->cubes[2]->pos[0],
               &model->cubes[2]->pos[1],
               &model->cubes[2]->pos[2],
               &model->cubes[2]->size[0],
               &model->cubes[2]->size[1],
               &model->cubes[2]->size[2]);
    }
    if (model->nobjects > 3)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &model->cubes[3]->pos[0],
               &model->cubes[3]->pos[1],
               &model->cubes[3]->pos[2],
               &model->cubes[3]->size[0],
               &model->cubes[3]->size[1],
               &model->cubes[3]->size[2]);
    }
    if (model->nobjects > 4)
    {
        fgets(buf, 200, stream);
        read_string(buf, &buf2, separator);
        sscanf(buf2, "%f %f %f %f %f %f", &model->cubes[4]->pos[0],
               &model->cubes[4]->pos[1],
               &model->cubes[4]->pos[2],
               &model->cubes[4]->size[0],
               &model->cubes[4]->size[1],
               &model->cubes[4]->size[2]);
    }

    fclose(stream);

    return (0);
}
