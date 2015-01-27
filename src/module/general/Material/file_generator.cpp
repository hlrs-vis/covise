/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: File generator for Covise API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Gabor Duroska                            ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.11.2000  V1.0                                             ++
// ++**********************************************************************/

// include headers
#include <limits.h>
#include <ftw.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <cstdio>

long int num_groups = 0;
long int num_materials = 0;
int fd;
FILE *datafile;
FILE *matfile;
int first = 0;
int pos = 0;
const char *slash = "/";

static int auswert(const char *path, const struct stat *statptr, int filetype)
{
    (void)path;
    (void)statptr;

    if (filetype == FTW_D)
        num_groups++;

    if (filetype == FTW_F)
        num_materials++;

    return (0);
}

static int auswert_grp(const char *path, const struct stat *statptr, int filetype)
{
    (void)statptr;

    if (filetype == FTW_D)
    {
        if ((first == 0) && (strlen(strrchr(path, '/') + 1) != 0))
        {
            fprintf(datafile, "%c%s%c", '"', strrchr(path, '/') + 1, '"');
            first = 1;
        }
        else if (strlen(strrchr(path, '/') + 1) != 0)
            fprintf(datafile, "%c%c%s%c", ',', '"', strrchr(path, '/') + 1, '"');
    }

    return (0);
}

static int auswert_mat(const char *path, const struct stat *statptr, int filetype)
{
    (void)statptr;

    if (filetype == FTW_F)
    {
        if ((first == 0) && (strlen(strrchr(path, '/') + 1) != 0))
        {
            fprintf(datafile, "%c%s%c", '"', strrchr(path, '/') + 1, '"');
            first = 1;
        }
        else if (strlen(strrchr(path, '/') + 1) != 0)
            fprintf(datafile, "%c%c%s%c", ',', '"', strrchr(path, '/') + 1, '"');
    }

    return (0);
}

static int auswert_pos(const char *path, const struct stat *statptr, int filetype)
{
    (void)statptr;

    if (filetype == FTW_D)
    {
        if ((first == 1) && (strlen(strrchr(path, '/') + 1) != 0))
        {
            fprintf(datafile, "%c%u", ',', pos);
        }

        if ((first == 0) && (strlen(strrchr(path, '/') + 1) != 0))
        {
            fprintf(datafile, "%u", pos);
            first = 1;
        }
    }

    if (filetype == FTW_F)
        pos++;

    return (0);
}

static int auswert_anz(const char *path, const struct stat *statptr, int filetype)
{
    (void)statptr;

    if (filetype == FTW_D)
    {
        if ((first == 1) && (strlen(strrchr(path, '/') + 1) != 0) && (pos != 0))
        {
            fprintf(datafile, "%c%u", ',', pos);
            pos = 0;
        }

        if ((first == 0) && (strlen(strrchr(path, '/') + 1) != 0) && (pos != 0))
        {
            fprintf(datafile, "%u", pos);
            first = 1;
            pos = 0;
        }
    }

    if (filetype == FTW_F)
        pos++;

    return (0);
}

static int get_values(const char *path, const struct stat *statptr, int filetype)
{
    (void)statptr;

    char dummy[100];
    char begrenz[100];
    float ambientColor[3];
    float diffuseColor[3];
    float specularColor[3];
    float emissiveColor[3];
    float shininess;
    float transparency;

    if (filetype == FTW_F)
    {
        if (strlen(strrchr(path, '/') + 1) != 0)
        {
            fprintf(datafile, "%c%c%s%c%c", '{', '"', strrchr(path, '/') + 1, '"', ',');

            // now open this file and get data...
            matfile = fopen(path, "r");
            if (fscanf(matfile, "%[^{]%s%s%f%f%f%s%f%f%f%s%f%f%f%s%f%f%f%s%f%s%f", dummy, begrenz,
                       dummy, &ambientColor[0], &ambientColor[1], &ambientColor[2],
                       dummy, &diffuseColor[0], &diffuseColor[1], &diffuseColor[2],
                       dummy, &specularColor[0], &specularColor[1], &specularColor[2],
                       dummy, &emissiveColor[0], &emissiveColor[1], &emissiveColor[2],
                       dummy, &shininess, dummy, &transparency) != 22)
            {
                fprintf(stderr, "Material: get_values: fscanf failed\n");
            }
            fclose(matfile);

            // ...and write it to file
            fprintf(datafile, "%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%c%c%f%c%s",
                    '"', ambientColor[0], '"', ',', '"', ambientColor[1], '"', ',', '"', ambientColor[2], '"', ',',
                    '"', diffuseColor[0], '"', ',', '"', diffuseColor[1], '"', ',', '"', diffuseColor[2], '"', ',',
                    '"', specularColor[0], '"', ',', '"', specularColor[1], '"', ',', '"', specularColor[2], '"', ',',
                    '"', emissiveColor[0], '"', ',', '"', emissiveColor[1], '"', ',', '"', emissiveColor[2], '"', ',',
                    '"', shininess, '"', ',', '"', transparency, '"', "},");
            fprintf(datafile, "\n");
        }
    }

    return (0);
}

int main(int argc, char *argv[])
{
    // create database file
    if ((datafile = fopen("Material.inc", "w")) != NULL)
    {
        char *pathname = NULL;
        if (argc > 1)
        {
            pathname = new char[strlen(argv[1]) + 10];
            strcpy(pathname, argv[1]);
        }
        else
        {
            pathname = new char[10];
            strcpy(pathname, slash);
        }
        // decide if pathname ends with a slash
        if (pathname[strlen(pathname) - 1] != '/')
        {
            strcat(pathname, slash);
        }

        // write data to file
        fd = ftw(pathname, auswert, 10);

        fprintf(datafile, "%s", "// This is a generated file. Do not edit!");
        fprintf(datafile, "\n\n");

        fprintf(datafile, "%s%li%s", "const int num_mat_groups = ", num_groups - 1, ";");
        fprintf(datafile, "\n");
        fprintf(datafile, "%s%li%s", "const int num_materials = ", num_materials, ";");
        fprintf(datafile, "\n");

        fprintf(datafile, "%s", "const int position[num_mat_groups] = {");
        first = 0;
        fd = ftw(pathname, auswert_pos, 10);
        fprintf(datafile, "%s", "};");
        fprintf(datafile, "\n");

        fprintf(datafile, "%s", "const int num_materials_gr[num_mat_groups] = {");
        first = 0;
        pos = 0;
        fd = ftw(pathname, auswert_anz, 10);
        fprintf(datafile, "%c%u", ',', pos);
        fprintf(datafile, "%s", "};");
        fprintf(datafile, "\n");

        fprintf(datafile, "%s", "const char *mat_groups[num_mat_groups] = {");
        first = 0;
        pos = 0;
        fd = ftw(pathname, auswert_grp, 10);
        fprintf(datafile, "%s", "};");
        fprintf(datafile, "\n");

        fprintf(datafile, "%s", "const char *mat_names[num_materials] = {");
        first = 0;
        fd = ftw(pathname, auswert_mat, 10);
        fprintf(datafile, "%s", "};");
        fprintf(datafile, "\n");

        fprintf(datafile, "\n");
        fprintf(datafile, "%s", "// material values");
        fprintf(datafile, "\n");
        fprintf(datafile, "%s", "// material name; ambientColor (3 values); diffuseColor (3 values);");
        fprintf(datafile, "\n");
        fprintf(datafile, "%s", "// specularColor (3 values); emissiveColor (3 values); shininess; transparency");
        fprintf(datafile, "\n");
        fprintf(datafile, "%s", "const char *mat_values[num_materials][15] = {");
        fprintf(datafile, "\n");
        fd = ftw(pathname, get_values, 10);
        fprintf(datafile, "%s", "};");
        fprintf(datafile, "\n");

        delete[] pathname;
    }

    fclose(datafile);
}
