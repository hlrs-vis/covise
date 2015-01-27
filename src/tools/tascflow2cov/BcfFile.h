/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BCF_FILE_H_
#define _BCF_FILE_H_

#define MAX_PATCHES 100
#define MAX_GRIDS 500
#define MAXBCFCOL 300

class BcfFile
{

private:
    int nb_tot_patches;
    int nb_grids;
    struct Grid
    {
        char *name;
        int nb_patches;
        int **patch;
    } **grid;

public:
    // Member functions
    BcfFile(char *);
    ~BcfFile();
    void get_patches(char *, int, int, int, int *, int ***);
    void get_nb_polygons(int *);
};
#endif
