/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BcfFile.h"
#include "ReadTascflowTDI.h"
#include <util/coviseCompat.h>

#ifdef CO_hp1020

#define dims_ dims
#define gnum_ gnum
#endif
#ifdef CO_hp

#define dims_ dims
#define gnum_ gnum
#endif
extern "C" {

void dims_(int *, int *, int *, int *, char *, int *, int);
void gnum_(char *, int *, int *, int);
}

BcfFile::BcfFile(char *bcf_path)
    : nb_grids(0)
{
    FILE *fd;
    char buffer[MAXBCFCOL];
    char *tmp_buf, *x_buf, *y_buf, *z_buf, *name;
    char *x_buf2, *y_buf2, *z_buf2, *name2;

    int x[2], y[2], z[2];
    int x2[2], y2[2], z2[2];
    int dimx, dimy, dimz;
    int dimx2, dimy2, dimz2;

    nb_tot_patches = 0;

    fd = fopen(bcf_path, "r");

    grid = new Grid *[MAX_GRIDS];

    if (fd > 0)
    {
        rewind(fd);
        fgets(buffer, MAXBCFCOL, fd);
        while (!feof(fd))
        {
            //fgets(buffer, MAXBCFCOL,fd);
            // if (feof(fd))
            //   return
            if (strstr(buffer, "$$$ILS") != NULL)
            {
                if (!feof(fd))
                    fgets(buffer, MAXBCFCOL, fd);
                if (!feof(fd))
                    fgets(buffer, MAXBCFCOL, fd);
                if (!feof(fd))
                    fgets(buffer, MAXBCFCOL, fd);
                while (!feof(fd) && strstr(buffer, "$$$ILE") == NULL)
                {
                    tmp_buf = strtok(buffer, "[");
                    x_buf = strtok(NULL, ",");
                    y_buf = strtok(NULL, ",");
                    z_buf = strtok(NULL, "]");
                    name = strtok(NULL, " ");
                    int i;
                    for (i = 0; i < strlen(name); i++)
                        name[i] = name[i + 1];

                    tmp_buf = strtok(NULL, "[");
                    x_buf2 = strtok(NULL, ",");
                    y_buf2 = strtok(NULL, ",");
                    z_buf2 = strtok(NULL, "]");
                    name2 = strtok(NULL, " .");
                    for (i = 0; i < strlen(name2); i++)
                        name2[i] = name2[i + 1];

                    tmp_buf = strtok(x_buf, ":");
                    sscanf(tmp_buf, "%d", &x[0]);
                    tmp_buf = strtok(NULL, ":");
                    if (tmp_buf != NULL)
                    {
                        sscanf(tmp_buf, "%d", &x[1]);
                        dimx = 2;
                    }
                    else
                        dimx = 1;

                    tmp_buf = strtok(y_buf, ":");
                    sscanf(tmp_buf, "%d", &y[0]);
                    tmp_buf = strtok(NULL, ":");
                    if (tmp_buf != NULL)
                    {
                        sscanf(tmp_buf, "%d", &y[1]);
                        dimy = 2;
                    }
                    else
                        dimy = 1;

                    tmp_buf = strtok(z_buf, ":");
                    sscanf(tmp_buf, "%d", &z[0]);
                    tmp_buf = strtok(NULL, ":");
                    if (tmp_buf != NULL)
                    {
                        sscanf(tmp_buf, "%d", &z[1]);
                        dimz = 2;
                    }
                    else
                        dimz = 1;
                    tmp_buf = strtok(x_buf2, ":");
                    sscanf(tmp_buf, "%d", &x2[0]);
                    tmp_buf = strtok(NULL, ":");
                    if (tmp_buf != NULL)
                    {
                        sscanf(tmp_buf, "%d", &x2[1]);
                        dimx2 = 2;
                    }
                    else
                        dimx2 = 1;

                    tmp_buf = strtok(y_buf2, ":");
                    sscanf(tmp_buf, "%d", &y2[0]);
                    tmp_buf = strtok(NULL, ":");
                    if (tmp_buf != NULL)
                    {
                        sscanf(tmp_buf, "%d", &y2[1]);
                        dimy2 = 2;
                    }
                    else
                        dimy2 = 1;

                    tmp_buf = strtok(z_buf2, ":");
                    sscanf(tmp_buf, "%d", &z2[0]);
                    tmp_buf = strtok(NULL, ":");
                    if (tmp_buf != NULL)
                    {
                        sscanf(tmp_buf, "%d", &z2[1]);
                        dimz2 = 2;
                    }
                    else
                        dimz2 = 1;

                    // if (dimx == dimx2 && dimy == dimy2 && dimz == dimz2&&
                    if (dimx == 1 || dimy == 1 || dimz == 1)
                    {
                        int aux;
                        if (dimx == 1)
                        {
                            if (y[0] > y[1])
                            {
                                aux = y[0];
                                y[0] = y[1];
                                y[1] = aux;
                            }
                            if (z[0] > z[1])
                            {
                                aux = z[0];
                                z[0] = z[1];
                                z[1] = aux;
                            }
                            nb_tot_patches += (y[1] - y[0]) * (z[1] - z[0]);
                        }
                        else if (dimy == 1)
                        {
                            if (x[0] > x[1])
                            {
                                aux = x[0];
                                x[0] = x[1];
                                x[1] = aux;
                            }
                            if (z[0] > z[1])
                            {
                                aux = z[0];
                                z[0] = z[1];
                                z[1] = aux;
                            }
                            nb_tot_patches += (x[1] - x[0]) * (z[1] - z[0]);
                        }
                        else if (dimz == 1)
                        {
                            if (x[0] > x[1])
                            {
                                aux = x[0];
                                x[0] = x[1];
                                x[1] = aux;
                            }
                            if (y[0] > y[1])
                            {
                                aux = y[0];
                                y[0] = y[1];
                                y[1] = aux;
                            }
                            nb_tot_patches += (x[1] - x[0]) * (y[1] - y[0]);
                        }

                        int found = 0;
                        i = 0;

                        while (i < nb_grids && !found)
                        {
                            if (!strcmp(grid[i]->name, name))
                                found = 1;
                            i++;
                        }

                        if (found)
                        {
                            i--;
                            grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        }
                        else
                        {
                            grid[i] = new Grid;
                            grid[i]->name = new char[strlen(name) + 1];
                            strcpy(grid[i]->name, name);
                            grid[i]->nb_patches = 0;
                            grid[i]->patch = new int *[MAX_PATCHES];
                            grid[i]->patch[grid[i]->nb_patches] = new int[8];
                            nb_grids++;
                        }

                        grid[i]->patch[grid[i]->nb_patches][0] = dimx;
                        grid[i]->patch[grid[i]->nb_patches][1] = dimy;
                        grid[i]->patch[grid[i]->nb_patches][2] = dimz;

                        int index = 3;
                        int j;
                        for (j = 0; j < dimx; j++)
                            grid[i]->patch[grid[i]->nb_patches][index++] = x[j] - 1;
                        for (j = 0; j < dimy; j++)
                            grid[i]->patch[grid[i]->nb_patches][index++] = y[j] - 1;
                        for (j = 0; j < dimz; j++)
                            grid[i]->patch[grid[i]->nb_patches][index++] = z[j] - 1;
                        grid[i]->nb_patches++;
                    }
                    //if (dimx == dimx2 && dimy == dimy2 && dimz == dimz2&&
                    if (dimx2 == 1 || dimy2 == 1 || dimz2 == 1)
                    {
                        int aux;
                        if (dimx2 == 1)
                        {
                            if (y2[0] > y2[1])
                            {
                                aux = y2[0];
                                y2[0] = y2[1];
                                y2[1] = aux;
                            }
                            if (z2[0] > z2[1])
                            {
                                aux = z2[0];
                                z2[0] = z2[1];
                                z2[1] = aux;
                            }
                            nb_tot_patches += (y2[1] - y2[0]) * (z2[1] - z2[0]);
                        }
                        else if (dimy2 == 1)
                        {
                            if (x2[0] > x2[1])
                            {
                                aux = x2[0];
                                x2[0] = x2[1];
                                x2[1] = aux;
                            }
                            if (z2[0] > z2[1])
                            {
                                aux = z2[0];
                                z2[0] = z2[1];
                                z2[1] = aux;
                            }
                            nb_tot_patches += (x2[1] - x2[0]) * (z2[1] - z2[0]);
                        }
                        else if (dimz2 == 1)
                        {
                            if (x2[0] > x2[1])
                            {
                                aux = x2[0];
                                x2[0] = x2[1];
                                x2[1] = aux;
                            }
                            if (y2[0] > y2[1])
                            {
                                aux = y2[0];
                                y2[0] = y2[1];
                                y2[1] = aux;
                            }
                            nb_tot_patches += (x2[1] - x2[0]) * (y2[1] - y2[0]);
                        }

                        int found = 0;
                        i = 0;
                        while (i < nb_grids && !found)
                        {
                            if (!strcmp(grid[i]->name, name2))
                                found = 1;
                            i++;
                        }

                        if (found)
                        {
                            i--;
                            grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        }
                        else
                        {
                            grid[i] = new Grid;
                            grid[i]->name = new char[strlen(name2) + 1];
                            strcpy(grid[i]->name, name2);
                            grid[i]->nb_patches = 0;
                            grid[i]->patch = new int *[MAX_PATCHES];
                            grid[i]->patch[grid[i]->nb_patches] = new int[8];
                            nb_grids++;
                        }

                        grid[i]->patch[grid[i]->nb_patches][0] = dimx2;
                        grid[i]->patch[grid[i]->nb_patches][1] = dimy2;
                        grid[i]->patch[grid[i]->nb_patches][2] = dimz2;

                        int j, index = 3;
                        for (j = 0; j < dimx2; j++)
                            grid[i]->patch[grid[i]->nb_patches][index++] = x2[j] - 1;
                        for (j = 0; j < dimy2; j++)
                            grid[i]->patch[grid[i]->nb_patches][index++] = y2[j] - 1;
                        for (j = 0; j < dimz2; j++)
                            grid[i]->patch[grid[i]->nb_patches][index++] = z2[j] - 1;
                        grid[i]->nb_patches++;
                    }
                    else
                    //embedded grids
                    {
                        int aux;

                        //first grid (the embedded one)
                        if (x[0] > x[1])
                        {
                            aux = x[0];
                            x[0] = x[1];
                            x[1] = aux;
                        }
                        if (y[0] > y[1])
                        {
                            aux = y[0];
                            y[0] = y[1];
                            y[1] = aux;
                        }
                        if (z[0] > z[1])
                        {
                            aux = z[0];
                            z[0] = z[1];
                            z[1] = aux;
                        }

                        //compare if there are some identical faces of the entire second grid and this region of the second grid; eliminate the coreponding polygons of the first grid

                        char buf[MAX_GRIDNAME_LENGHT];
                        int grid_index, tascflowError, x_dim = 0, y_dim = 0, z_dim = 0;
                        int name_lenght = strlen(name2);
                        gnum_(name2, &grid_index, &tascflowError, name_lenght);
                        dims_(&x_dim, &y_dim, &z_dim, &grid_index, buf, &tascflowError, MAX_GRIDNAME_LENGHT);
                        if (tascflowError > 0)
                            sendError("ERROR: An error was occured while executing the routine: TRGNUM");

                        if (x2[0] > 1 || x2[1] < x_dim || y2[0] > 1 || y2[1] < y_dim || z2[0] > 1 || z2[1] < z_dim)
                        {
                            int found = 0;
                            i = 0;

                            while (i < nb_grids && !found)
                            {
                                if (!strcmp(grid[i]->name, name))
                                    found = 1;
                                i++;
                            }

                            if (found)
                                i--;
                            else
                            {
                                grid[i] = new Grid;
                                grid[i]->name = new char[strlen(name) + 1];
                                strcpy(grid[i]->name, name);
                                grid[i]->nb_patches = 0;
                                grid[i]->patch = new int *[MAX_PATCHES];
                                nb_grids++;
                            }
                            if (x2[0] > 1)
                            {
                                nb_tot_patches += (y[1] - y[0]) * (z[1] - z[0]);
                                grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                grid[i]->patch[grid[i]->nb_patches][0] = 1;
                                grid[i]->patch[grid[i]->nb_patches][1] = 2;
                                grid[i]->patch[grid[i]->nb_patches][2] = 2;
                                grid[i]->patch[grid[i]->nb_patches][3] = x[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][4] = y[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][5] = y[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][6] = z[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][7] = z[1] - 1;
                                grid[i]->nb_patches++;
                            }
                            if (x2[1] < x_dim)
                            {
                                nb_tot_patches += (y[1] - y[0]) * (z[1] - z[0]);
                                grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                grid[i]->patch[grid[i]->nb_patches][0] = 1;
                                grid[i]->patch[grid[i]->nb_patches][1] = 2;
                                grid[i]->patch[grid[i]->nb_patches][2] = 2;
                                grid[i]->patch[grid[i]->nb_patches][3] = x[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][4] = y[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][5] = y[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][6] = z[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][7] = z[1] - 1;
                                grid[i]->nb_patches++;
                            }
                            if (y2[0] > 1)
                            {
                                nb_tot_patches += (x[1] - x[0]) * (z[1] - z[0]);
                                grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                grid[i]->patch[grid[i]->nb_patches][0] = 2;
                                grid[i]->patch[grid[i]->nb_patches][1] = 1;
                                grid[i]->patch[grid[i]->nb_patches][2] = 2;
                                grid[i]->patch[grid[i]->nb_patches][3] = x[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][4] = x[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][5] = y[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][6] = z[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][7] = z[1] - 1;
                                grid[i]->nb_patches++;
                            }
                            if (y2[1] < y_dim)
                            {
                                nb_tot_patches += (x[1] - x[0]) * (z[1] - z[0]);
                                grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                grid[i]->patch[grid[i]->nb_patches][0] = 2;
                                grid[i]->patch[grid[i]->nb_patches][1] = 1;
                                grid[i]->patch[grid[i]->nb_patches][2] = 2;
                                grid[i]->patch[grid[i]->nb_patches][3] = x[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][4] = x[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][5] = y[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][6] = z[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][7] = z[1] - 1;
                                grid[i]->nb_patches++;
                            }
                            if (z2[0] > 1)
                            {
                                nb_tot_patches += (x[1] - x[0]) * (y[1] - y[0]);
                                grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                grid[i]->patch[grid[i]->nb_patches][0] = 2;
                                grid[i]->patch[grid[i]->nb_patches][1] = 2;
                                grid[i]->patch[grid[i]->nb_patches][2] = 1;
                                grid[i]->patch[grid[i]->nb_patches][3] = x[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][4] = x[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][5] = y[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][6] = y[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][7] = z[0] - 1;
                                grid[i]->nb_patches++;
                            }
                            if (z2[1] < z_dim)
                            {
                                nb_tot_patches += (x[1] - x[0]) * (y[1] - y[0]);
                                grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                grid[i]->patch[grid[i]->nb_patches][0] = 2;
                                grid[i]->patch[grid[i]->nb_patches][1] = 2;
                                grid[i]->patch[grid[i]->nb_patches][2] = 1;
                                grid[i]->patch[grid[i]->nb_patches][3] = x[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][4] = x[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][5] = y[0] - 1;
                                grid[i]->patch[grid[i]->nb_patches][6] = y[1] - 1;
                                grid[i]->patch[grid[i]->nb_patches][7] = z[1] - 1;
                                grid[i]->nb_patches++;
                            }
                        }

                        //second grid

                        if (x2[0] > x2[1])
                        {
                            aux = x2[0];
                            x2[0] = x2[1];
                            x2[1] = aux;
                        }
                        if (y2[0] > y2[1])
                        {
                            aux = y2[0];
                            y2[0] = y2[1];
                            y2[1] = aux;
                        }
                        if (z2[0] > z2[1])
                        {
                            aux = z2[0];
                            z2[0] = z2[1];
                            z2[1] = aux;
                        }

                        int found = 0;
                        i = 0;

                        while (i < nb_grids && !found)
                        {
                            if (!strcmp(grid[i]->name, name2))
                                found = 1;
                            i++;
                        }

                        if (found)
                        {
                            i--;
                            grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        }
                        else
                        {
                            grid[i] = new Grid;
                            grid[i]->name = new char[strlen(name2) + 1];
                            strcpy(grid[i]->name, name2);
                            grid[i]->nb_patches = 0;
                            grid[i]->patch = new int *[MAX_PATCHES];
                            grid[i]->patch[grid[i]->nb_patches] = new int[8];
                            nb_grids++;
                        }

                        nb_tot_patches += (y2[1] - y2[0]) * (z2[1] - z2[0]);
                        grid[i]->patch[grid[i]->nb_patches][0] = 1;
                        grid[i]->patch[grid[i]->nb_patches][1] = 2;
                        grid[i]->patch[grid[i]->nb_patches][2] = 2;
                        grid[i]->patch[grid[i]->nb_patches][3] = x2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][4] = y2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][5] = y2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][6] = z2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][7] = z2[1] - 1;
                        grid[i]->nb_patches++;

                        nb_tot_patches += (y2[1] - y2[0]) * (z2[1] - z2[0]);
                        grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        grid[i]->patch[grid[i]->nb_patches][0] = 1;
                        grid[i]->patch[grid[i]->nb_patches][1] = 2;
                        grid[i]->patch[grid[i]->nb_patches][2] = 2;
                        grid[i]->patch[grid[i]->nb_patches][3] = x2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][4] = y2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][5] = y2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][6] = z2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][7] = z2[1] - 1;
                        grid[i]->nb_patches++;

                        nb_tot_patches += (x2[1] - x2[0]) * (z2[1] - z2[0]);
                        grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        grid[i]->patch[grid[i]->nb_patches][0] = 2;
                        grid[i]->patch[grid[i]->nb_patches][1] = 1;
                        grid[i]->patch[grid[i]->nb_patches][2] = 2;
                        grid[i]->patch[grid[i]->nb_patches][3] = x2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][4] = x2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][5] = y2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][6] = z2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][7] = z2[1] - 1;
                        grid[i]->nb_patches++;

                        nb_tot_patches += (x2[1] - x2[0]) * (z2[1] - z2[0]);
                        grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        grid[i]->patch[grid[i]->nb_patches][0] = 2;
                        grid[i]->patch[grid[i]->nb_patches][1] = 1;
                        grid[i]->patch[grid[i]->nb_patches][2] = 2;
                        grid[i]->patch[grid[i]->nb_patches][3] = x2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][4] = x2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][5] = y2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][6] = z2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][7] = z2[1] - 1;
                        grid[i]->nb_patches++;

                        nb_tot_patches += (x2[1] - x2[0]) * (y2[1] - y2[0]);
                        grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        grid[i]->patch[grid[i]->nb_patches][0] = 2;
                        grid[i]->patch[grid[i]->nb_patches][1] = 2;
                        grid[i]->patch[grid[i]->nb_patches][2] = 1;
                        grid[i]->patch[grid[i]->nb_patches][3] = x2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][4] = x2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][5] = y2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][6] = y2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][7] = z2[0] - 1;
                        grid[i]->nb_patches++;

                        nb_tot_patches += (x2[1] - x2[0]) * (y2[1] - y2[0]);
                        grid[i]->patch[grid[i]->nb_patches] = new int[8];
                        grid[i]->patch[grid[i]->nb_patches][0] = 2;
                        grid[i]->patch[grid[i]->nb_patches][1] = 2;
                        grid[i]->patch[grid[i]->nb_patches][2] = 1;
                        grid[i]->patch[grid[i]->nb_patches][3] = x2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][4] = x2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][5] = y2[0] - 1;
                        grid[i]->patch[grid[i]->nb_patches][6] = y2[1] - 1;
                        grid[i]->patch[grid[i]->nb_patches][7] = z2[1] - 1;
                        grid[i]->nb_patches++;
                    }
                    if (!feof(fd))
                        fgets(buffer, MAXBCFCOL, fd);
                } //while ILE
            } //if ILS
            else if (strstr(buffer, "$$$BLS") != NULL)
            {
                while (!feof(fd) && strstr(buffer, "$$$BLE") == NULL)
                {
                    if (strstr(buffer, "PERIODIC BOUNDARY") != NULL)
                    //    ||strstr(buffer,"GENERAL GRID INTERFACE") != NULL)
                    {
                        while (!feof(fd) && strstr(buffer, "These regions currently include the following faces") == NULL)
                            fgets(buffer, MAXBCFCOL, fd);
                        if (!feof(fd))
                            fgets(buffer, MAXBCFCOL, fd);

                        while (!feof(fd) && (strstr(buffer, "Primary") != NULL))
                        //||strstr(buffer,"Side")!= NULL))
                        {
                            tmp_buf = strtok(buffer, "[");
                            x_buf = strtok(NULL, ",");
                            y_buf = strtok(NULL, ",");
                            z_buf = strtok(NULL, "]");
                            name = strtok(NULL, "\n ");
                            int i;
                            for (i = 0; i < strlen(name); i++)
                                name[i] = name[i + 1];

                            tmp_buf = strtok(x_buf, ":");
                            sscanf(tmp_buf, "%d", &x[0]);
                            tmp_buf = strtok(NULL, ":");
                            if (tmp_buf != NULL)
                            {
                                sscanf(tmp_buf, "%d", &x[1]);
                                dimx = 2;
                            }
                            else
                                dimx = 1;

                            tmp_buf = strtok(y_buf, ":");
                            sscanf(tmp_buf, "%d", &y[0]);
                            tmp_buf = strtok(NULL, ":");
                            if (tmp_buf != NULL)
                            {
                                sscanf(tmp_buf, "%d", &y[1]);
                                dimy = 2;
                            }
                            else
                                dimy = 1;

                            tmp_buf = strtok(z_buf, ":");
                            sscanf(tmp_buf, "%d", &z[0]);
                            tmp_buf = strtok(NULL, ":");
                            if (tmp_buf != NULL)
                            {
                                sscanf(tmp_buf, "%d", &z[1]);
                                dimz = 2;
                            }
                            else
                                dimz = 1;

                            if (dimx == 1 || dimy == 1 || dimz == 1)
                            {
                                int aux;
                                if (dimx == 1)
                                {
                                    if (y[0] > y[1])
                                    {
                                        aux = y[0];
                                        y[0] = y[1];
                                        y[1] = aux;
                                    }
                                    if (z[0] > z[1])
                                    {
                                        aux = z[0];
                                        z[0] = z[1];
                                        z[1] = aux;
                                    }
                                    nb_tot_patches += (y[1] - y[0]) * (z[1] - z[0]);
                                }
                                else if (dimy == 1)
                                {
                                    if (x[0] > x[1])
                                    {
                                        aux = x[0];
                                        x[0] = x[1];
                                        x[1] = aux;
                                    }
                                    if (z[0] > z[1])
                                    {
                                        aux = z[0];
                                        z[0] = z[1];
                                        z[1] = aux;
                                    }
                                    nb_tot_patches += (x[1] - x[0]) * (z[1] - z[0]);
                                }
                                else if (dimz == 1)
                                {
                                    if (x[0] > x[1])
                                    {
                                        aux = x[0];
                                        x[0] = x[1];
                                        x[1] = aux;
                                    }
                                    if (y[0] > y[1])
                                    {
                                        aux = y[0];
                                        y[0] = y[1];
                                        y[1] = aux;
                                    }
                                    nb_tot_patches += (x[1] - x[0]) * (y[1] - y[0]);
                                }

                                int found = 0;
                                i = 0;

                                while (i < nb_grids && !found)
                                {
                                    if (!strcmp(grid[i]->name, name))
                                        found = 1;
                                    i++;
                                }

                                if (found)
                                {
                                    i--;
                                    grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                }
                                else
                                {
                                    grid[i] = new Grid;
                                    grid[i]->name = new char[strlen(name) + 1];
                                    strcpy(grid[i]->name, name);
                                    grid[i]->nb_patches = 0;
                                    grid[i]->patch = new int *[MAX_PATCHES];
                                    grid[i]->patch[grid[i]->nb_patches] = new int[8];
                                    nb_grids++;
                                }

                                grid[i]->patch[grid[i]->nb_patches][0] = dimx;
                                grid[i]->patch[grid[i]->nb_patches][1] = dimy;
                                grid[i]->patch[grid[i]->nb_patches][2] = dimz;

                                int index = 3;
                                int j;
                                for (j = 0; j < dimx; j++)
                                    grid[i]->patch[grid[i]->nb_patches][index++] = x[j] - 1;
                                for (j = 0; j < dimy; j++)
                                    grid[i]->patch[grid[i]->nb_patches][index++] = y[j] - 1;
                                for (j = 0; j < dimz; j++)
                                    grid[i]->patch[grid[i]->nb_patches][index++] = z[j] - 1;
                                grid[i]->nb_patches++;

                                fgets(buffer, MAXBCFCOL, fd);
                            }
                        }
                    }
                    if (!feof(fd))
                        fgets(buffer, MAXBCFCOL, fd);
                } //while BLE

            } //if ILS, BLS
            if (!feof(fd))
                fgets(buffer, MAXBCFCOL, fd);
        } //while feof
    } //if fd
    fclose(fd);
}

BcfFile::~BcfFile()
{

    for (int i = 0; i < nb_grids; i++)
    {
        //cerr << "Name of the deleted grid: "<<grid[i]->name<<endl;

        delete[] grid[i]->name;
        //cerr << "Number of deleted patches:"<<grid[i]->nb_patches<<endl<<endl;
        for (int j = 0; j < grid[i]->nb_patches; j++)
            delete[] grid[i]->patch[j];
        delete[] grid[i]->patch;
    }
    delete[] grid;
}

void BcfFile::get_nb_polygons(int *nb_pol)
{
    *nb_pol = nb_tot_patches;
}

void BcfFile::get_patches(char *grid_name, int dimx, int dimy, int dimz, int *nb_patches, int ***patch_list)
{
    int found = 0, i = 0;
    *nb_patches = 0;
    *patch_list = NULL;
    int xmin, xmax, ymin, ymax, zmin, zmax;

    while (i < nb_grids && !found)
    {
        if (!strcmp(grid[i]->name, grid_name))
            found = 1;
        i++;
    }
    if (found)
    {
        i--;

        *patch_list = new int *[nb_tot_patches + 1];
        int j;
        for (j = 0; j < grid[i]->nb_patches; j++)
        {
            if (grid[i]->patch[j][0] == 1)
            {
                xmin = grid[i]->patch[j][3];
                ymin = grid[i]->patch[j][4];
                ymax = grid[i]->patch[j][5];
                zmin = grid[i]->patch[j][6];
                zmax = grid[i]->patch[j][7];
                //cerr<<"xmin"<<xmin<<"ymin: "<<ymin<<"ymax: "<<ymax<<"zmin: "<<zmin<<"zmax: "<<zmax<<endl;
                for (int k = ymin; k < ymax; k++)
                    for (int l = zmin; l < zmax; l++)
                    {
                        (*patch_list)[*nb_patches] = new int[4];

                        (*patch_list)[*nb_patches][0] = xmin * dimy * dimz + k * dimz + l;
                        (*patch_list)[*nb_patches][1] = xmin * dimy * dimz + k * dimz + l + 1;
                        (*patch_list)[*nb_patches][2] = xmin * dimy * dimz + (k + 1) * dimz + l + 1;
                        (*patch_list)[(*nb_patches)++][3] = xmin * dimy * dimz + (k + 1) * dimz + l;
                        //j*(ymax-ymin)*(zmax-zmin)+(k-ymin)*(zmax-zmin)+l-zmin
                    }
            }
            else if (grid[i]->patch[j][1] == 1)
            {
                xmin = grid[i]->patch[j][3];
                xmax = grid[i]->patch[j][4];
                ymin = grid[i]->patch[j][5];
                zmin = grid[i]->patch[j][6];
                zmax = grid[i]->patch[j][7];

                for (int k = xmin; k < xmax; k++)
                    for (int l = zmin; l < zmax; l++)
                    {
                        (*patch_list)[*nb_patches] = new int[4];

                        (*patch_list)[*nb_patches][0] = k * dimy * dimz + ymin * dimz + l;
                        (*patch_list)[*nb_patches][1] = k * dimy * dimz + ymin * dimz + l + 1;
                        (*patch_list)[*nb_patches][2] = (k + 1) * dimy * dimz + ymin * dimz + l + 1;
                        (*patch_list)[(*nb_patches)++][3] = (k + 1) * dimy * dimz + ymin * dimz + l;
                    }
            }
            else if (grid[i]->patch[j][2] == 1)
            {
                xmin = grid[i]->patch[j][3];
                xmax = grid[i]->patch[j][4];
                ymin = grid[i]->patch[j][5];
                ymax = grid[i]->patch[j][6];
                zmin = grid[i]->patch[j][7];
                //cerr<<"xmin"<<xmin<<"xmax: "<<xmax<<"ymin: "<<ymin<<"ymax: "<<ymax<<"zmin: "<<zmin<<endl;

                for (int k = xmin; k < xmax; k++)
                    for (int l = ymin; l < ymax; l++)
                    {
                        (*patch_list)[*nb_patches] = new int[4];

                        (*patch_list)[*nb_patches][0] = k * dimy * dimz + l * dimz + zmin;
                        (*patch_list)[*nb_patches][1] = k * dimy * dimz + (l + 1) * dimz + zmin;
                        (*patch_list)[*nb_patches][2] = (k + 1) * dimy * dimz + (l + 1) * dimz + zmin;
                        (*patch_list)[(*nb_patches)++][3] = (k + 1) * dimy * dimz + l * dimz + zmin;
                    }
            }
        }

        /*      cerr << " nb_patches= "<<*nb_patches<<endl;
      for (j=0;j<*nb_patches;j++)
      {
        if ((*patch_list)[j][0]>dimx*dimy*dimz)
      cerr<<" Error: ("<<j<<") -0-"<<(*patch_list)[j][0]<<endl;
        if ((*patch_list)[j][1]>dimx*dimy*dimz)
          cerr<<" Error: ("<<j<<") -1- "<<(*patch_list)[j][1]<<endl;
        if ((*patch_list)[j][2]>dimx*dimy*dimz)
          cerr<<" Error: ("<<j<<") -2- "<<(*patch_list)[j][2]<<endl;
        if ((*patch_list)[j][3]>dimx*dimy*dimz)
          cerr<<" Error: ("<<j<<") -3- "<<(*patch_list)[j][2]<<endl;
      }
      cerr<<"living get_patches"<<endl;
      */
    }
}
