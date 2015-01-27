/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:  COVISE TascFlowTDI  application module                   **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Sasha Cioringa                                                **
 **                                                                        **
 **                                                                        **
 ** Date:  21.11.00                                                        **
\**************************************************************************/

#include "t2c.h"
#include "coStepFile.h"
#include "FieldFile.h"
#include "BcfFile.h"
#include "covWriteFiles.h"
#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/// Mapping Fortran identifiers to "C" standard

#ifdef CO_hp1020
#define rread_ rread
#define nexts_ nexts
#define scalar_ scalar
#define blocko_ blocko
#define nxtr_ nxtr
#define dims_ dims
#define file_ file
#define rinit_ rinit
#define rclose_ rclose
#define ngbl_ ngbl
#define subreg_ subreg
#define region_ region
#endif

#ifdef CO_hp
#define rread_ rread
#define nexts_ nexts
#define scalar_ scalar
#define blocko_ blocko
#define nxtr_ nxtr
#define dims_ dims
#define gnum_ gnum
#define file_ file
#define rinit_ rinit
#define rclose_ rclose
#define ngbl_ ngbl
#define subreg_ subreg
#define region_ region
#endif

extern "C" {

void dims_(int *, int *, int *, int *, char *, int *, int);
void file_(char *, char *, char *, int *, int, int, int);
void rinit_(int *nr, int *ni, int *nc, int *);
void rread_(float *, int *, char *, int *, int);
void nexts_(float *, int *, int *, char *, int *, int *, int);
void rclose_(int *);
void ngbl_(int *, int *);
void scalar_(const char *, float *, int *, int *, float *, int *, int);
void blocko_(int *, int *, int *, float *, int *);
void nxtr_(float *, int *, int *, char *, int *, int *, int);
void subreg_(char *, int *, int *, int);
void region_(char *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int);
}

TascFlow::TascFlow(char *grd, char *rso, char *bcf, char *step)
    : gridpath(grd)
    , rsopath(rso)
    , bcfpath(bcf)
    , step_rsopath(step)
{

    // private data
    nr = 20000000;
    ni = 20000000;
    nc = 500000;

    const char *confVar;

    // REAL memory
    confVar = getenv("T2C_RWK");
    if (confVar)
        nr = atoi(confVar);

    // INTEGER memory
    confVar = getenv("T2C_IWK");
    if (confVar)
        ni = atoi(confVar);

    // CHAR memory
    confVar = getenv("T2C_CWK");
    if (confVar)
        nc = atoi(confVar);

    //CoviseConfig::getEntry("ReadTascflowTDI.RWK",&nr);
    //CoviseConfig::getEntry("ReadTascflowTDI.IWK",&ni);
    //CoviseConfig::getEntry("ReadTascflowTDI.CWK",&nc);

    IWK = new int[ni]; // TDI array for integers
    RWK = new float[nr]; // TDI array for floats
    CWK = new char[nc]; // TDI array for characters

    All_Regions = new char *[MAX_REGIONS];
    All_Regions[0] = new char[5];
    strcpy(All_Regions[0], "none");
    for (int i = 1; i < MAX_REGIONS; i++)
        All_Regions[i] = NULL;
    ;
    nb_regions = 0;

    has_grd_file = 0;
    has_rso_file = 0;
    has_bcf_file = 0;
    has_step_rso_file = 0;
    //gridpath = NULL;
    //rsopath = NULL;
    //bcfpath = NULL;
    main_field = NULL;
    param_counter = 0;
}

TascFlow::~TascFlow()
{

    delete[] RWK;
    delete[] IWK;
    delete[] CWK;

    for (int i = 0; i <= nb_regions; i++)
        if (All_Regions[i])
            delete[] All_Regions[i];

    delete[] All_Regions;

    if (main_field)
        delete main_field;

    //if(bcf)
    //   delete bcf;
}

int TascFlow::compute()
{
    //const char* ChoiceColorVal[] = {"white","red","blue","green","grey","black"};
    int dimx, dimy, dimz, dimg;

    char buf[128];
    char **GridNames;

    float *x_coord, *y_coord, *z_coord;
    int num_corners = 0, num_polygons = 0, num_points = 0;
    int *corner_list, *polygon_list;

    char **All_Fields = new char *[MAX_VECT_FIELDS + MAX_SCAL_FIELDS];
    int nb_tot_fields;

    open_database(gridpath, rsopath, bcfpath, 1);

    read_fields(&All_Fields, &nb_tot_fields);
    main_field = new Fields(All_Fields, nb_tot_fields);
    main_field->getAddresses(&VectChoiceVal, &ScalChoiceVal, &VectFieldList, &nb_vect, &nb_scal, &nb_field_list);

    int i;
    for (i = 0; i < nb_tot_fields; i++)
        if (All_Fields[i])
            delete[] All_Fields[i];

    delete[] All_Fields;

    for (i = 1; i <= nb_regions; i++)
        if (All_Regions[i])
            delete[] All_Regions[i];

    read_regions(&All_Regions, &nb_regions);
    rclose_(&tascflowError);
    if (tascflowError > 0)
        printf("ERROR: Could not close the database");

    for (i = 1; i <= nb_scal; i++)
        cerr << "   " << i << ". " << ScalChoiceVal[i - 1] << endl;

    do
    {
        cout << "Chose the scalar data: " << flush;
        cin >> data1;
    } while (data1 <= 0 || data1 > nb_scal);

    for (i = 1; i <= nb_vect; i++)
        cerr << "   " << i << ". " << VectChoiceVal[i - 1] << endl;

    do
    {
        cout << "Chose the vector data: " << flush;
        cin >> vector1;
    } while (vector1 <= 0 || vector1 > nb_vect);

    //get the value of parameters
    //reg = p_region->getValue();
    //int col= p_color->getValue();

    /*const char *step_rsopath = p_steprsopath->getValue();

   if (strcmp(step_rsopath,init_path))
      has_step_rso_file = 1;

   int timesteps = p_timesteps->getValue();
   int skip_value = p_skip->getValue();*/

    //regions

    //open and read bcf file
    if (has_bcf_file)
    {
        bcf = new BcfFile(bcfpath);
        bcf->get_nb_polygons(&num_polygons);
    }
    else
    {
        bcf = NULL;
        num_polygons = 0;
    }

    num_corners = num_polygons * 4;
    num_points = num_corners;

    //sprintf(buf, "%s", p_outPort9->getObjName());
    //   coDoPolygons *pol_out=new coDoPolygons(p_outPort9->getObjName(),num_points,num_corners,num_polygons);

    //pol_out->getAddresses(&x_coord,&y_coord,&z_coord,&corner_list,&polygon_list);
    x_coord = new float[num_points];
    y_coord = new float[num_points];
    z_coord = new float[num_points];
    corner_list = new int[num_corners];
    polygon_list = new int[num_polygons];

    for (i = 0; i < num_polygons; i++)
        polygon_list[i] = i * 4;
    for (i = 0; i < num_corners; i++)
        corner_list[i] = i;

    int coord_index = 0;

    char buffer[MAX_GRIDNAME_LENGHT + 1];
    buffer[MAX_GRIDNAME_LENGHT] = '\0';

    ngbl_(&ngrids, &tascflowError);
    if (tascflowError > 0)
    {
        printf("ERROR: An error was occured while executing the routine: TRNGBL");
        return 0;
    }

    GridNames = new char *[ngrids + 1];
    GridNames[ngrids] = NULL;

    int fd = covOpenOutFile("grid.covise");
    if (!fd)
    {
        cerr << "Error in opening the file grid.covise!" << endl;
        return 0;
    }
    int fd2 = covOpenOutFile("bcf.covise");
    if (!fd2)
    {
        cerr << "Error in opening the file bcf.covise!" << endl;
        return 0;
    }
    covWriteSetBegin(fd, ngrids);

    for (i = 1; i <= ngrids; i++)
    {
        dims_(&dimx, &dimy, &dimz, &i, buffer, &tascflowError, MAX_GRIDNAME_LENGHT);
        if (tascflowError > 0)
        {
            printf("ERROR: An error was occured while executing the routine: TRGDIM");
            return 0;
        }

        int r, s, t;
        int **patch_list;
        int nb_patches;
        char *pch;
        pch = strtok(buffer, " ");
        if (pch != NULL)
        {
            GridNames[i - 1] = new char[strlen(pch) + 1];
            strcpy(GridNames[i - 1], pch);
            if (has_bcf_file)
                bcf->get_patches(GridNames[i - 1], dimx, dimy, dimz, &nb_patches, &patch_list);
            else
            {
                nb_patches = 0;
                patch_list = NULL;
            }
        }
        else
            GridNames[i - 1] = NULL;

        dimg = dimx * dimy * dimz;

        // sprintf(buf, "%s_%d", p_outPort1->getObjName(), i );
        //grid_out = new coDoStructuredGrid( buf , dimx, dimy, dimz);
        //grid_out->getAddresses(&x,&y,&z);
        float *x = new float[dimg];
        float *y = new float[dimg];
        float *z = new float[dimg];

        float *x_temp = new float[dimg];
        float *y_temp = new float[dimg];
        float *z_temp = new float[dimg];

        scalar_("X", x_temp, &dimg, &i, RWK, &tascflowError, 1);
        if (tascflowError > 0)
        {
            printf("ERROR: An error was occured while executing the routine: TRSCAL");
            return 0;
        }
        scalar_("Y", y_temp, &dimg, &i, RWK, &tascflowError, 1);
        if (tascflowError > 0)
        {
            printf("ERROR: An error was occured while executing the routine: TRSCAL");
            return 0;
        }
        scalar_("Z", z_temp, &dimg, &i, RWK, &tascflowError, 1);
        if (tascflowError > 0)
        {
            printf("ERROR: An error was occured while executing the routine: TRSCAL");
            return 0;
        }

        for (r = 0; r < dimz; r++)
            for (s = 0; s < dimy; s++)
                for (t = 0; t < dimx; t++)
                {
                    x[t * dimy * dimz + s * dimz + r] = x_temp[r * dimy * dimx + s * dimx + t];
                    y[t * dimy * dimz + s * dimz + r] = y_temp[r * dimy * dimx + s * dimx + t];
                    z[t * dimy * dimz + s * dimz + r] = z_temp[r * dimy * dimx + s * dimx + t];
                }

        int idx;
        for (idx = 0; idx < nb_patches; idx++)
            for (int i4 = 0; i4 < 4; i4++)
            {
                x_coord[coord_index] = x[patch_list[idx][i4]];
                y_coord[coord_index] = y[patch_list[idx][i4]];
                z_coord[coord_index++] = z[patch_list[idx][i4]];
            }

        //outgrd[i-1] = grid_out;
        //outgrd[i] = NULL;

        delete[] x_temp;
        delete[] y_temp;
        delete[] z_temp;

        for (idx = 0; idx < nb_patches; idx++)
            delete[] patch_list[idx];

        if (patch_list != NULL)
            delete[] patch_list;

        covWriteSTRGRD(fd, dimx, dimy, dimz, x, y, z, NULL, NULL, 0);
    }

    covWriteSetEnd(fd, NULL, NULL, 0);
    covCloseOutFile(fd);

    covWritePOLYGN(fd2, num_polygons, polygon_list, num_corners, corner_list, num_points, x_coord, y_coord, z_coord, NULL, NULL, 0);
    covCloseOutFile(fd2);

    delete[] x_coord;
    delete[] y_coord;
    delete[] z_coord;

    int timesteps = 0;

    char *next_path = NULL;

    if (step_rsopath)
    {
        coStepFile *step_file = new coStepFile(step_rsopath);
        step_file->set_skip_value(0);
        // reading the rso fields

        do
        {
            step_file->get_nextpath(&next_path);
            cerr << "next_path= " << next_path << endl;
            timesteps++;
        } while (next_path);
        timesteps--;
        delete step_file;
    }

    for (int j = 0; j <= timesteps; j++)
    {
        if (data1 > 1)
        {
            if (j == 0)
            {
                sprintf(buf, "%s.covise", ScalChoiceVal[data1 - 1]);
                fd = covOpenOutFile(buf);
                if (!fd)
                {
                    cerr << "Error in opening the file " << buf << "!" << endl;
                    return 0;
                }
                if (timesteps > 0)
                    covWriteSetBegin(fd, timesteps + 1);
            }

            covWriteSetBegin(fd, ngrids);

            coStepFile *step_file;
            int data_ok = 1;

            if (j == 1)
                step_file = new coStepFile(step_rsopath);

            if (j >= 1)
            {
                step_file->get_nextpath(&next_path);
                data_ok = open_database(gridpath, next_path, bcfpath, 1);
                if (!data_ok)
                    return 0;
            }

            for (i = 1; i <= ngrids; i++)
            {
                dims_(&dimx, &dimy, &dimz, &i, buffer, &tascflowError, MAX_GRIDNAME_LENGHT);
                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TRGDIM");
                    return 0;
                }
                dimg = dimx * dimy * dimz;
                float *field = new float[dimg];
                float *field_temp = new float[dimg];

                scalar_(ScalChoiceVal[data1 - 1], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data1 - 1]));

                if (tascflowError > 0)
                {
                    printf("ERROR : An error was occured while executing the routine: TRSCAL");
                    return 0;
                }

                for (int r = 0; r < dimz; r++)
                    for (int s = 0; s < dimy; s++)
                        for (int t = 0; t < dimx; t++)
                            field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];

                covWriteSTRSDT(fd, dimg, field, dimx, dimy, dimz, NULL, NULL, 0);
                delete[] field_temp;
                delete[] field;
            }
            covWriteSetEnd(fd, NULL, NULL, 0);

            if (j == timesteps)
            {
                if (timesteps > 0)
                {
                    char *an[] = { "TIMESTEP" };
                    char nb[3];
                    sprintf(nb, "%d", timesteps);
                    char *at[] = { nb };
                    covWriteSetEnd(fd, an, at, 1);
                }
                covCloseOutFile(fd);
            }
        }

        if (vector1 > 1)
        {
            if (j == 0)
            {
                sprintf(buf, "%s.covise", VectChoiceVal[vector1 - 1]);
                fd2 = covOpenOutFile(buf);
                if (!fd2)
                {
                    cerr << "Error in opening the file " << buf << "!" << endl;
                    return 0;
                }

                if (timesteps > 0)
                {
                    covWriteSetBegin(fd2, timesteps + 1);
                    //open_database(gridpath,rsopath,bcfpath,1);
                }
            }

            covWriteSetBegin(fd2, ngrids);

            coStepFile *step_file;

            if (j == 1)
                step_file = new coStepFile(step_rsopath);

            if (j >= 1)
                step_file->get_nextpath(&next_path);

            for (i = 1; i <= ngrids; i++)
            {
                dims_(&dimx, &dimy, &dimz, &i, buffer, &tascflowError, MAX_GRIDNAME_LENGHT);
                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TRGDIM");
                    return 0;
                }
                dimg = dimx * dimy * dimz;
                int index;
                //sprintf(buf, "%s_0_%d", p_outPort4->getObjName(),i);
                //vector_out  = new DO_Structured_V3D_Data(buf, dimx, dimy, dimz);
                //vector_out->getAddresses(&x,&y,&z);

                float *x = new float[dimg];
                float *y = new float[dimg];
                float *z = new float[dimg];

                index = (vector1 - 2) * 3;

                float *x_temp = new float[dimg];
                float *y_temp = new float[dimg];
                float *z_temp = new float[dimg];

                scalar_(VectFieldList[index], x_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index]));
                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TRSCAL");
                    return 0;
                }
                scalar_(VectFieldList[index + 1], y_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 1]));
                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TRSCAL");
                    return 0;
                }
                scalar_(VectFieldList[index + 2], z_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 2]));
                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TRSCAL");
                    return 0;
                }

                for (int r = 0; r < dimz; r++)
                    for (int s = 0; s < dimy; s++)
                        for (int t = 0; t < dimx; t++)
                        {
                            x[t * dimy * dimz + s * dimz + r] = x_temp[r * dimy * dimx + s * dimx + t];
                            y[t * dimy * dimz + s * dimz + r] = y_temp[r * dimy * dimx + s * dimx + t];
                            z[t * dimy * dimz + s * dimz + r] = z_temp[r * dimy * dimx + s * dimx + t];
                        }

                covWriteSTRVDT(fd2, dimg, x, y, z, dimx, dimy, dimz, NULL, NULL, 0);

                delete[] x_temp;
                delete[] y_temp;
                delete[] z_temp;

                delete[] x;
                delete[] y;
                delete[] z;
            }

            covWriteSetEnd(fd2, NULL, NULL, 0);

            if (j == timesteps)
            {
                if (timesteps > 0)
                {
                    char *an[] = { "TIMESTEP" };
                    char nb[3];
                    sprintf(nb, "%d", timesteps);
                    char *at[] = { nb };
                    covWriteSetEnd(fd2, an, at, 1);
                }
                covCloseOutFile(fd2);
            }
        }
    }

    //scalar data 1

    rclose_(&tascflowError);
    if (tascflowError > 0)
    {
        printf("ERROR: An error was occured while executing the routine: TGCLOS");
        return 0;
    }
    return 1;
}

//===================private functions==================================

int TascFlow::open_database(char *grid_path, char *rso_path, char *bcf_path, int in_execution)
{
    if (grid_path != NULL)
    {
        FILE *fd;
        int lengrd = strlen(grid_path);

        fd = fopen(grid_path, "r");
        if (fd)
        {
            rewind(fd);
            char c = getc(fd);
            fclose(fd);
            if (c != '\377')
            {
                char fform[2];
                fform[0] = c;
                fform[1] = '\0';

                rinit_(&nr, &ni, &nc, &tascflowError);

                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TGINIT\n");
                    return 0;
                }

                file_("grd", grid_path, fform, &tascflowError, 3, lengrd, 1);

                if (tascflowError > 0)
                {
                    printf("ERROR: An error was occured while executing the routine: TGINIT\n");
                    return 0;
                }
                has_grd_file = 1;

                if (rso_path != NULL)
                {
                    int lenrso = strlen(rso_path);

                    fd = fopen(rso_path, "r");
                    if (fd)
                    {
                        rewind(fd);
                        char c = getc(fd);
                        fclose(fd);
                        char fform[2];
                        if (c != '\377')
                        {
                            if (c == ' ')
                                fform[0] = 'F';
                            else
                                fform[0] = 'U';
                            fform[1] = '\0';
                            file_("rso", rso_path, fform, &tascflowError, 3, lenrso, 1);
                            has_rso_file = 1;
                        }
                        else
                        {
                            //sendError("%s is directory!",rso_path);
                            has_rso_file = 0;
                        }
                    }
                    else
                    {
                        printf("ERROR: Could not open Rsofile: %s", rso_path);
                        has_rso_file = 0;
                    }

                } //if has_rso_file

                if (bcf_path != NULL)
                {
                    //bcfpath = bcf_path;
                    int lenbcf = strlen(bcf_path);

                    fd = fopen(bcf_path, "r");
                    if (fd)
                    {
                        rewind(fd);
                        char c = getc(fd);
                        fclose(fd);
                        char fform[2];
                        if (c != '\377')
                        {
                            if (c == ' ')
                                fform[0] = 'F';
                            else
                                fform[0] = 'U';
                            fform[1] = '\0';
                            file_("bcf", bcf_path, fform, &tascflowError, 3, lenbcf, 1);
                            has_bcf_file = 1;
                        }
                        else
                        {
                            //printf("%s is directory!",bcf_path);
                            has_bcf_file = 0;
                        }
                    }
                    else
                    {
                        printf("ERROR: Could not open Bcf file: %s", bcf_path);
                        has_bcf_file = 0;
                    }
                } //if has_bcf_file
            }
            else
            {
                printf("%s is directory!", grid_path);
                has_grd_file = 0;
            }
        }
        else
        {
            printf("ERROR: Could not open Grdfile: %s", grid_path);
            cerr << "fd= " << fd << endl;
            return 0;
        }

        printf("Reading the input files! Please wait...");
        rread_(RWK, IWK, CWK, &tascflowError, nc);

        if (tascflowError > 0)
        {
            if (in_execution)
                printf("ERROR: An error was occured while executing the routine: TRREAD");
            else
                printf("WARNING: Cannot execute the routine TRREAD. Please check if the files match.");
            return 0;
        }
        printf("The files have been read successfully!\n");
    }
    else
    {
        if (in_execution)
            printf("Please indicate the grid file!");
        return 0;
    }

    return 1;
}

void TascFlow::read_fields(char ***AllFields, int *nb_tot_fields)
{
    int init = 1;
    int done = 0;
    *nb_tot_fields = 0;
    char buffer[MAX_FIELD_LENGHT + 1];

    nexts_(RWK, IWK, &init, buffer, &done, &tascflowError, MAX_FIELD_LENGHT);
    if (tascflowError > 0)
    {
        printf("ERROR: An error was occured while executing the routine: TRNXTS");
        return;
    }
    if (!done)
    {
        int i = 0;
        char ch = '_';
        while (i < MAX_FIELD_LENGHT && (isalnum(ch) || ch == '_'))
        {
            ch = buffer[i];
            i++;
        }
        buffer[i - 1] = '\0';
        (*AllFields)[*nb_tot_fields] = new char[MAX_FIELD_LENGHT + 1];
        strcpy((*AllFields)[(*nb_tot_fields)++], buffer);
    }
    init = 0;
    while (!done)
    {
        nexts_(RWK, IWK, &init, buffer, &done, &tascflowError, MAX_FIELD_LENGHT);
        if (tascflowError > 0)
        {
            printf("ERROR: An error was occured while executing the routine: TRNXTS");
            return;
        }
        int i = 0;
        char ch = '_';
        while (i < MAX_FIELD_LENGHT && (isalnum(ch) || ch == '_'))
        {
            ch = buffer[i];
            i++;
        }
        buffer[i - 1] = '\0';
        (*AllFields)[*nb_tot_fields] = new char[MAX_FIELD_LENGHT + 1];
        strcpy((*AllFields)[(*nb_tot_fields)++], buffer);
    }
}

void TascFlow::read_regions(char ***AllRegions, int *nb_tot_regions)
{
    int init = 1;
    int done = 0;
    *nb_tot_regions = 0;
    char buffer[MAX_REGION_LENGHT + 1];
    buffer[MAX_REGION_LENGHT] = '\0';

    nxtr_(RWK, IWK, &init, buffer, &done, &tascflowError, MAX_REGION_LENGHT);
    if (tascflowError > 0)
    {
        printf("ERROR: An error was occured while executing the routine: TRNXTR");
        return;
    }
    if (!done)
    {
        (*AllRegions)[(*nb_tot_regions) + 1] = new char[MAX_REGION_LENGHT + 1];
        int i = 0;
        char ch = '_';
        while (i < MAX_REGION_LENGHT && (isalnum(ch) || ch == '_'))
        {
            ch = buffer[i];
            i++;
        }
        buffer[i - 1] = '\0';
        strcpy((*AllRegions)[++(*nb_tot_regions)], buffer);
    }
    init = 0;
    while (!done)
    {
        (*AllRegions)[(*nb_tot_regions) + 1] = new char[MAX_REGION_LENGHT + 1];
        nxtr_(RWK, IWK, &init, buffer, &done, &tascflowError, MAX_REGION_LENGHT);
        if (tascflowError > 0)
        {
            printf("ERROR: An error was occured while executing the routine: TRNXTS");
            return;
        }
        int i = 0;
        char ch = '_';
        while (i < MAX_REGION_LENGHT && (isalnum(ch) || ch == '_'))
        {
            ch = buffer[i];
            i++;
        }
        buffer[i - 1] = '\0';
        strcpy((*AllRegions)[++(*nb_tot_regions)], buffer);
    }
}

int main(int argc, char *argv[])

{
    if (argc < 4 || argc > 5)
    {
        cerr << "\nUsage: \n\n  " << argv[0]
             << "  grd_path rso_path bcf_path [first_rso_step_path]\n"
             << endl;
        exit(-1);
    }

    char *grdFile = argv[1];
    char *rsoFile = argv[2];
    char *bcfFile = argv[3];
    char *stepFile = (argc > 4) ? argv[3] : NULL;

    TascFlow *application = new TascFlow(grdFile, rsoFile, bcfFile, stepFile);
    application->compute();
    delete application;

    return 0;
}
