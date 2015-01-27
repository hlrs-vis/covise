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

#include "ReadTascflowTDI.h"
#include <api/coStepFile.h>
#include "FieldFile.h"
#include "BcfFile.h"

#include <util/coviseCompat.h>

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

TascFlow::TascFlow(int argc, char *argv[])
    : coModule(argc, argv, "Read TascFlow data")
{
    const char *ChoiceVal[] = { "show complete region", "white lines", "red lines", "blue lines", "green lines", "grey lines", "black lines" };
    char *ChoiceInitVal[] = { "none" };
    strcpy(init_path, "data/nofile");

    //parameters
    p_steprsopath = addFileBrowserParam("steprsopath", "path of the first timestep rso file");
    p_steprsopath->setValue(init_path, "*");

    p_gridpath = addFileBrowserParam("gridpath", "path of the grid file");
    p_gridpath->setValue(init_path, "*");
    p_rsopath = addFileBrowserParam("rsopath", "path of the rso file");
    p_rsopath->setValue(init_path, "*");
    p_bcfpath = addFileBrowserParam("bcfpath", "path of the bcf file");
    p_bcfpath->setValue(init_path, "*");
    p_timesteps = addInt32Param("timesteps", "timesteps");
    p_timesteps->setValue(1);
    p_skip = addInt32Param("skipped_files", "number of skip files for each timestep");
    p_skip->setValue(0);
    p_region = addChoiceParam("region_names", "names of the regions");
    p_region->setValue(1, ChoiceInitVal, 0);
    p_color = addChoiceParam("show_region", "the way of showing the regions");
    p_color->setValue(7, ChoiceVal, 1);

    p_vector1 = addChoiceParam("vector_data_1", "first vector data");
    p_vector1->setValue(1, ChoiceInitVal, 0);
    p_vector2 = addChoiceParam("vector_data_2", "second vector data");
    p_vector2->setValue(1, ChoiceInitVal, 0);
    p_data1 = addChoiceParam("scalar_data_1", "first scalar data");
    p_data1->setValue(1, ChoiceInitVal, 0);
    p_data2 = addChoiceParam("scalar_data_2", "second scalar data");
    p_data2->setValue(1, ChoiceInitVal, 0);
    p_data3 = addChoiceParam("scalar_data_3", "third scalar data (used for complete region too)");
    p_data3->setValue(1, ChoiceInitVal, 0);

    //ports
    p_outPort1 = addOutputPort("grid", "StructuredGrid", "grid out");
    p_outPort2 = addOutputPort("block", "IntArr", "block off array");
    p_outPort3 = addOutputPort("region", "Lines|StructuredGrid", "highlighted region");
    p_outPort4 = addOutputPort("vector1", "Vec3", "vector1 out");
    p_outPort5 = addOutputPort("vector2", "Vec3", "vector2 out");
    p_outPort6 = addOutputPort("data1", "Float", "data1 out");
    p_outPort7 = addOutputPort("data2", "Float", "data2 out");
    p_outPort8 = addOutputPort("data3", "Float", "data3 out || complete region ");
    p_outPort9 = addOutputPort("bcf_polygons", "Polygons", "polygons out");

    // private data
    nr = 20000000;
    ni = 20000000;
    nc = 500000;
    nr = coCoviseConfig::getInt("Module.ReadTascflowTDI.RWK", 20000000);
    ni = coCoviseConfig::getInt("Module.ReadTascflowTDI.IWK", 20000000);
    nc = coCoviseConfig::getInt("Module.ReadTascflowTDI.CWK", 500000);

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
    gridpath = NULL;
    rsopath = NULL;
    bcfpath = NULL;
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

    /*if(bcf)
      delete bcf;*/
}

void TascFlow::param(const char *paramName)
{
    const char *fileName = NULL;
    int path_param = 0;

    if (in_map_loading() && strcmp(paramName, "SetModuleTitle"))
        param_counter++;
    if (strcmp(p_gridpath->getName(), paramName) == 0)
    {
        path_param = 1;
        fileName = p_gridpath->getValue();
        if (fileName)
        {
            if (gridpath)
                delete[] gridpath;
            gridpath = new char[strlen(fileName) + 1];
            strcpy(gridpath, fileName);
            if (strcmp(gridpath, init_path))
                has_grd_file = 1;
        }
        else
            sendError("ERROR: The name of the grd file is NULL");
    }
    else if (strcmp(p_rsopath->getName(), paramName) == 0)
    {
        path_param = 1;
        fileName = p_rsopath->getValue();
        if (fileName)
        {
            if (rsopath)
                delete[] rsopath;
            rsopath = new char[strlen(fileName) + 1];
            strcpy(rsopath, fileName);
            if (strcmp(rsopath, init_path))
                has_rso_file = 1;
        }
        else
            sendError("ERROR: The name of the rso file is NULL");
    }
    else if (strcmp(p_bcfpath->getName(), paramName) == 0)
    {
        path_param = 1;
        fileName = p_bcfpath->getValue();
        if (fileName)
        {
            if (bcfpath)
                delete[] bcfpath;
            bcfpath = new char[strlen(fileName) + 1];
            strcpy(bcfpath, fileName);
            if (strcmp(bcfpath, init_path))
                has_bcf_file = 1;
        }
        else
            sendError("ERROR: The name of the bcf file is NULL");
    }
    else if (strcmp(p_region->getName(), paramName) == 0)
    {
        reg = p_region->getValue();
    }

    if ((in_map_loading() && param_counter == 9) || !in_map_loading() && path_param)
    {
        int data_ok = open_database(gridpath, rsopath, bcfpath, 0);

        if (data_ok == SUCCESS)
        {

            char **OldVectVal = NULL, **OldScalVal = NULL;
            int old_nb_vect = 0, old_nb_scal = 0;

            if (main_field)
            {
                main_field->get_fields(&OldVectVal, &OldScalVal, &old_nb_vect, &old_nb_scal);
                delete main_field;
            }

            char **All_Fields = new char *[MAX_VECT_FIELDS + MAX_SCAL_FIELDS];
            int nb_tot_fields;

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
                sendError("ERROR: Could not close the database");

            vector1 = p_vector1->getValue();
            vector2 = p_vector2->getValue();
            data1 = p_data1->getValue();
            data2 = p_data2->getValue();
            data3 = p_data3->getValue();

            if (OldVectVal)
            {
                int found1 = 0, found2 = 0;

                for (i = 0; i < nb_vect; i++)
                {
                    if (!found1 && !strcmp(OldVectVal[vector1], VectChoiceVal[i]))
                    {
                        found1 = 1;
                        vector1 = i + 1;
                    }
                    if (!found2 && !strcmp(OldVectVal[vector2], VectChoiceVal[i]))
                    {
                        found2 = 1;
                        vector2 = i + 1;
                    }
                }
                if (!found1)
                    vector1 = 1;
                if (!found2)
                    vector2 = 1;
            }

            if (OldScalVal)
            {
                int found1 = 0, found2 = 0, found3 = 0;
                for (i = 0; i < nb_scal; i++)
                {
                    if (!found1 && !strcmp(OldScalVal[data1], ScalChoiceVal[i]))
                    {
                        found1 = 1;
                        data1 = i + 1;
                    }
                    if (!found2 && !strcmp(OldScalVal[data2], ScalChoiceVal[i]))
                    {
                        found2 = 1;
                        data2 = i + 1;
                    }
                    if (!found3 && !strcmp(OldScalVal[data3], ScalChoiceVal[i]))
                    {
                        found3 = 1;
                        data3 = i + 1;
                    }
                }
                if (!found1)
                    data1 = 1;
                if (!found2)
                    data2 = 1;
                if (!found3)
                    data3 = 1;
            }

            for (i = 0; i < old_nb_vect; i++)
                if (OldVectVal[i])
                    delete[] OldVectVal[i];

            if (OldVectVal)
                delete[] OldVectVal;

            for (i = 0; i < old_nb_scal; i++)
                if (OldScalVal[i])
                    delete[] OldScalVal[i];

            if (OldScalVal)
                delete[] OldScalVal;

            //	 if (!in_map_loading())
            // BUGFIX: Choices count from 1 !!!
            p_region->setValue(nb_regions + 1, All_Regions, reg);

            p_vector1->setValue(nb_vect, VectChoiceVal, vector1);
            p_vector2->setValue(nb_vect, VectChoiceVal, vector2);
            p_data1->setValue(nb_scal, ScalChoiceVal, data1);
            p_data2->setValue(nb_scal, ScalChoiceVal, data2);
            p_data3->setValue(nb_scal, ScalChoiceVal, data3);
        }
    }
    /*   if (tascflowError>0)
      sendError("ERROR: Could not close the database"); */
}

int TascFlow::compute(const char *)
{
    const char *ChoiceColorVal[] = { "white", "red", "blue", "green", "grey", "black" };
    int dimx, dimy, dimz, dimg;
    int *block_array, *block_array_temp;
    float *x, *y, *z;
    float *field;
    char buf[128];
    char **GridNames;

    float *x_coord, *y_coord, *z_coord;
    int num_corners = 0, num_polygons = 0, num_points = 0;
    int *corner_list, *polygon_list;

    //get the value of parameters
    //reg = p_region->getValue();
    int col = p_color->getValue();
    vector1 = p_vector1->getValue();
    vector2 = p_vector2->getValue();
    data1 = p_data1->getValue();
    data2 = p_data2->getValue();
    data3 = p_data3->getValue();

    const char *step_rsopath = p_steprsopath->getValue();

    if (strcmp(step_rsopath, init_path))
        has_step_rso_file = 1;

    int timesteps = p_timesteps->getValue();
    int skip_value = p_skip->getValue();

    coDistributedObject **outgrd;
    coDistributedObject **outblock;
    coDistributedObject **outregion;
    coDistributedObject **outvector1;
    coDistributedObject **outvector2;
    coDistributedObject **outdata1;
    coDistributedObject **outdata2;
    coDistributedObject **outdata3;

    coDistributedObject **time_outputgrid;
    coDistributedObject **time_outputblock;
    coDistributedObject **time_outputvector1;
    coDistributedObject **time_outputvector2;
    coDistributedObject **time_outputdata1;
    coDistributedObject **time_outputdata2;
    coDistributedObject **time_outputdata3;

    if (timesteps > 1 && has_step_rso_file)
    {
        time_outputgrid = new coDistributedObject *[timesteps + 1];
        time_outputgrid[timesteps] = NULL;

        time_outputblock = new coDistributedObject *[timesteps + 1];
        time_outputblock[timesteps] = NULL;

        time_outputdata1 = new coDistributedObject *[timesteps + 1];
        time_outputdata1[timesteps] = NULL;

        time_outputdata2 = new coDistributedObject *[timesteps + 1];
        time_outputdata2[timesteps] = NULL;

        time_outputdata3 = new coDistributedObject *[timesteps + 1];
        time_outputdata3[timesteps] = NULL;

        time_outputvector1 = new coDistributedObject *[timesteps + 1];
        time_outputvector1[timesteps] = NULL;

        time_outputvector2 = new coDistributedObject *[timesteps + 1];
        time_outputvector2[timesteps] = NULL;
    }
    //regions

    /*coDistributedObject **set_outputregion = new coDistributedObject*[ngrids+1];
   set_outputregion[ngrids]=NULL;*/

    coDoSet *outputgrid;
    coDoSet *outputblock;
    //coDoSet *outputregion;
    coDoSet *outputvector1;
    coDoSet *outputvector2;
    coDoSet *outputdata1;
    coDoSet *outputdata2;
    coDoSet *outputdata3;

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
    coDoPolygons *pol_out = new coDoPolygons(p_outPort9->getObjName(), num_points, num_corners, num_polygons);

    pol_out->getAddresses(&x_coord, &y_coord, &z_coord, &corner_list, &polygon_list);
    int i;
    for (i = 0; i < num_polygons; i++)
        polygon_list[i] = i * 4;
    for (i = 0; i < num_corners; i++)
        corner_list[i] = i;

    int coord_index = 0;

    // open and read the database
    int data_ok = open_database(gridpath, rsopath, bcfpath, 1);
    if (data_ok == SUCCESS)
    {
        char buffer[MAX_GRIDNAME_LENGHT + 1];
        buffer[MAX_GRIDNAME_LENGHT] = '\0';

        ngbl_(&ngrids, &tascflowError);
        if (tascflowError > 0)
        {
            sendError("ERROR: An error was occured while executing the routine: TRNGBL");
            return FAIL;
        }
        GridNames = new char *[ngrids + 1];
        GridNames[ngrids] = NULL;

        //regions

        int len_reg;
        int num_subreg, n_sreg = 0;
        //float *x_region, *y_region, *z_region;
        if (reg > 1)
        {
            len_reg = strlen(All_Regions[reg - 1]);
            subreg_(All_Regions[reg - 1], &num_subreg, &tascflowError, len_reg);
            if (tascflowError > 0)
            {
                sendError("ERROR: An error was occured while executing the routine: TRREGN");
                return FAIL;
            }
        }

        outgrd = new coDistributedObject *[ngrids + 1];
        outgrd[ngrids] = NULL;

        outblock = new coDistributedObject *[ngrids + 1];
        outblock[ngrids] = NULL;

        if (vector1 > 0)
        {
            outvector1 = new coDistributedObject *[ngrids + 1];
            outvector1[ngrids] = NULL;
        }

        if (vector2 > 0)
        {
            outvector2 = new coDistributedObject *[ngrids + 1];
            outvector2[ngrids] = NULL;
        }

        if (data1 > 0)
        {
            outdata1 = new coDistributedObject *[ngrids + 1];
            outdata1[ngrids] = NULL;
        }

        if (data2 > 0)
        {
            outdata2 = new coDistributedObject *[ngrids + 1];
            outdata2[ngrids] = NULL;
        }

        if (data3 > 0)
        {
            if ((col > 1) || (col == 1 && reg == 1))
            {
                outdata3 = new coDistributedObject *[ngrids + 1];
                outdata3[ngrids] = NULL;
            }
            else
            {
                outdata3 = new coDistributedObject *[num_subreg + 1];
                outdata3[num_subreg] = NULL;
            }
        }

        coDoStructuredGrid *grid_out = NULL;
        coDoIntArr *block_out = NULL;
        coDoLines *region_out = NULL;
        coDoStructuredGrid *region_grid_out = NULL;
        coDoFloat *data_out = NULL;
        coDoFloat *reg_data_out = NULL;
        coDoVec3 *vector_out = NULL;

        outregion = new coDistributedObject *[300];

        //create the set of output grids
        int i;
        for (i = 1; i <= ngrids; i++)
        {
            dims_(&dimx, &dimy, &dimz, &i, buffer, &tascflowError, MAX_GRIDNAME_LENGHT);
            if (tascflowError > 0)
            {
                sendError("ERROR: An error was occured while executing the routine: TRGDIM");
                return FAIL;
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

            sprintf(buf, "%s_%d", p_outPort1->getObjName(), i);
            grid_out = new coDoStructuredGrid(buf, dimx, dimy, dimz);
            grid_out->getAddresses(&x, &y, &z);

            if (!grid_out->objectOk())
            {
                sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort1->getName());
                return FAIL;
            }

            float *x_temp = new float[dimg];
            float *y_temp = new float[dimg];
            float *z_temp = new float[dimg];

            scalar_("X", x_temp, &dimg, &i, RWK, &tascflowError, 1);
            if (tascflowError > 0)
            {
                sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                return FAIL;
            }
            scalar_("Y", y_temp, &dimg, &i, RWK, &tascflowError, 1);
            if (tascflowError > 0)
            {
                sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                return FAIL;
            }
            scalar_("Z", z_temp, &dimg, &i, RWK, &tascflowError, 1);
            if (tascflowError > 0)
            {
                sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                return FAIL;
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
            outgrd[i - 1] = grid_out;
            outgrd[i] = NULL;

            delete[] x_temp;
            delete[] y_temp;
            delete[] z_temp;

            for (idx = 0; idx < nb_patches; idx++)
                delete[] patch_list[idx];

            if (patch_list != NULL)
                delete[] patch_list;

            //block off

            // create the first block_off object

            sprintf(buf, "%s_0_%d", p_outPort2->getObjName(), i);
            int dim[1];
            dim[0] = (dimx - 1) * (dimy - 1) * (dimz - 1);
            block_out = new coDoIntArr(buf, 1, dim);

            if (!block_out->objectOk())
            {
                sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort2->getName());
                return FAIL;
            }

            block_out->getAddress(&block_array);

            block_array_temp = new int[dimg];
            blocko_(block_array_temp, &dimg, &i, RWK, &tascflowError);
            if (tascflowError > 0)
            {
                sendError("ERROR: An error was occured while executing the routine: TRBLOF");
                return FAIL;
            }
            for (t = 0; t < dimx - 1; t++)
                for (s = 0; s < dimy - 1; s++)
                    for (r = 0; r < dimz - 1; r++)
                        block_array[t * (dimy - 1) * (dimz - 1) + s * (dimz - 1) + r] = block_array_temp[r * (dimy) * (dimx) + s * (dimx) + t];

            delete[] block_array_temp;
            outblock[i - 1] = block_out;
            outblock[i] = NULL;

            // regions
            if (reg > 1)
            {
                // outregion = new coDistributedObject*[num_subreg+1];
                // outregion[num_subreg] = NULL;
                //n_sreg = 0;
                int j;
                int n_x = 1, n_y = 1, n_z = 1, i_l = 0, j_l = 0, k_l = 0;

                for (j = 0; j < num_subreg; j++)
                {
                    //sendInfo("num_subreg");
                    int i_u, j_u, k_u;
                    //  sprintf(buf, "%s_%d_%d", p_outPort3->getObjName(),i, j );
                    int ng = i;
                    region_(All_Regions[reg - 1], &i_l, &j_l, &k_l, &i_u, &j_u, &k_u, &ng, &j, &tascflowError, len_reg);

                    if (tascflowError > 0)
                    {
                        sendError("ERROR: An error was occured while executing the routine: TRREGS");
                        return FAIL;
                    }
                    if (ng == i)
                    {
                        n_x = i_u - i_l + 1;
                        n_y = j_u - j_l + 1;
                        n_z = k_u - k_l + 1;
                        float *x_start, *y_start, *z_start;

                        if (col > 1)
                        {
                            //Show lines
                            int num_line_points, num_line_corners, num_lines;
                            int *line_corner_list, *line_list;
                            int point_index = 0;

                            if (n_x == 1)
                            {
                                num_lines = 4;
                                num_line_corners = 2 * (n_y + n_z);
                            }
                            else if (n_y == 1)
                            {
                                num_lines = 4;
                                num_line_corners = 2 * (n_x + n_z);
                            }
                            else if (n_z == 1)
                            {
                                num_lines = 4;
                                num_line_corners = 2 * (n_x + n_y);
                            }
                            else
                            {
                                num_lines = 12;
                                num_line_corners = 4 * (n_x + n_y + n_z);
                            }

                            num_line_points = num_line_corners;

                            sprintf(buf, "%s_%d", p_outPort3->getObjName(), n_sreg + 1);

                            region_out = new coDoLines(buf, num_line_points, num_line_corners, num_lines);
                            region_out->getAddresses(&x_start, &y_start, &z_start, &line_corner_list, &line_list);

                            if (n_x == 1)
                            {
                                line_list[0] = 0;
                                int k;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[1] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[2] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[3] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                            }
                            else if (n_y == 1)
                            {
                                line_list[0] = 0;
                                int k;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_l - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_l - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[1] = point_index;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[2] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[3] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                            }
                            else if (n_z == 1)
                            {
                                line_list[0] = 0;
                                int k;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[1] = point_index;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[2] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[3] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                            }
                            else
                            {
                                line_list[0] = 0;
                                int k;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[1] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[2] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (j_l - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[3] = point_index;
                                for (k = 0; k < n_z; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (j_u - 1) * dimz + k + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[4] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    y_start[point_index] = y[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    z_start[point_index] = z[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[5] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(i_l - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[6] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[7] = point_index;
                                for (k = 0; k < n_y; k++)
                                {
                                    x_start[point_index] = x[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(i_u - 1) * dimy * dimz + (k + j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[8] = point_index;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_l - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_l - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[9] = point_index;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_l - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[10] = point_index;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_l - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_l - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_l - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                                line_list[11] = point_index;
                                for (k = 0; k < n_x; k++)
                                {
                                    x_start[point_index] = x[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    y_start[point_index] = y[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    z_start[point_index] = z[(k + i_l - 1) * dimy * dimz + (j_u - 1) * dimz + k_u - 1];
                                    line_corner_list[point_index] = point_index;
                                    point_index++;
                                }
                            }

                            region_out->addAttribute("COLOR", ChoiceColorVal[col - 2]);
                            outregion[n_sreg] = region_out;
                            outregion[n_sreg + 1] = NULL;
                            n_sreg++;
                        } //if col > 1
                        else
                        {
                            //Show complete region
                            //n_sreg = 1;
                            //sprintf(buf, "%s", p_outPort3->getObjName());
                            sprintf(buf, "%s_%d", p_outPort3->getObjName(), n_sreg + 1);
                            region_grid_out = new coDoStructuredGrid(buf, n_x, n_y, n_z);
                            region_grid_out->getAddresses(&x_start, &y_start, &z_start);

                            if (!region_grid_out->objectOk())
                            {
                                sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort3->getName());
                                return FAIL;
                            }
                            for (r = 0; r < n_z; r++)
                                for (s = 0; s < n_y; s++)
                                    for (t = 0; t < n_x; t++)
                                    {
                                        x_start[t * n_y * n_z + s * n_z + r] = x[(t + i_l - 1) * dimy * dimz + (s + j_l - 1) * dimz + r + k_l - 1];
                                        y_start[t * n_y * n_z + s * n_z + r] = y[(t + i_l - 1) * dimy * dimz + (s + j_l - 1) * dimz + r + k_l - 1];
                                        z_start[t * n_y * n_z + s * n_z + r] = z[(t + i_l - 1) * dimy * dimz + (s + j_l - 1) * dimz + r + k_l - 1];
                                    }
                            outregion[n_sreg] = region_grid_out;
                            outregion[n_sreg + 1] = NULL;

                            if (data3 > 0)
                            {
                                float *field_temp = new float[dimg];
                                sprintf(buf, "%s_0_%d", p_outPort8->getObjName(), n_sreg + 1);
                                reg_data_out = new coDoFloat(buf, n_x, n_y, n_z);
                                if (!reg_data_out->objectOk())
                                {
                                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort8->getName());
                                    return FAIL;
                                }
                                float *field_reg;
                                reg_data_out->getAddress(&field_reg);
                                scalar_(ScalChoiceVal[data3], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data3]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }
                                for (int r = 0; r < n_z; r++)
                                    for (int s = 0; s < n_y; s++)
                                        for (int t = 0; t < n_x; t++)
                                            field_reg[t * n_y * n_z + s * n_z + r] = field_temp[(r + k_l - 1) * dimy * dimx + (s + j_l - 1) * dimx + t + i_l - 1];
                                delete[] field_temp;
                                outdata3[n_sreg] = reg_data_out;
                                outdata3[n_sreg + 1] = NULL;
                            }
                            n_sreg++;
                        }
                    } //if ng = i
                } //for 1 to num_subreg
            }

            //scalar data 1
            if (data1 > 1)
            {
                sprintf(buf, "%s_0_%d", p_outPort6->getObjName(), i);
                data_out = new coDoFloat(buf, dimx, dimy, dimz);
                if (!data_out->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort6->getName());
                    return FAIL;
                }
                data_out->getAddress(&field);
                float *field_temp = new float[dimg];
                scalar_(ScalChoiceVal[data1 - 1], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data1 - 1]));

                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }
                for (int r = 0; r < dimz; r++)
                    for (int s = 0; s < dimy; s++)
                        for (int t = 0; t < dimx; t++)
                            field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];
                delete[] field_temp;

                outdata1[i - 1] = data_out;
                outdata1[i] = NULL;
            }

            //scalar data 2
            if (data2 > 0)
            {
                sprintf(buf, "%s_0_%d", p_outPort7->getObjName(), i);
                data_out = new coDoFloat(buf, dimx, dimy, dimz);
                if (!data_out->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort7->getName());
                    return FAIL;
                }
                data_out->getAddress(&field);
                float *field_temp = new float[dimg];
                scalar_(ScalChoiceVal[data2 - 1], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data2]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }

                for (int r = 0; r < dimz; r++)
                    for (int s = 0; s < dimy; s++)
                        for (int t = 0; t < dimx; t++)
                            field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];
                delete[] field_temp;

                outdata2[i - 1] = data_out;
                outdata2[i] = NULL;
            }

            //scalar data 3
            if (data3 > 0)
            {
                float *field_temp = new float[dimg];
                if (col > 1 || (col == 1 && reg == 1))
                {
                    sprintf(buf, "%s_0_%d", p_outPort8->getObjName(), i);
                    data_out = new coDoFloat(buf, dimx, dimy, dimz);
                    if (!data_out->objectOk())
                    {
                        sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort8->getName());
                        return FAIL;
                    }
                    data_out->getAddress(&field);

                    scalar_(ScalChoiceVal[data3 - 1], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data3]));
                    if (tascflowError > 0)
                    {
                        sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                        return FAIL;
                    }
                    for (int r = 0; r < dimz; r++)
                        for (int s = 0; s < dimy; s++)
                            for (int t = 0; t < dimx; t++)
                                field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];
                    outdata3[i - 1] = data_out;
                    outdata3[i] = NULL;
                }
                delete[] field_temp;
            }

            // vector data 1
            if (vector1 > 0)
            {
                int index;
                sprintf(buf, "%s_0_%d", p_outPort4->getObjName(), i);
                vector_out = new coDoVec3(buf, dimx, dimy, dimz);
                if (!vector_out->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort4->getName());
                    return FAIL;
                }
                vector_out->getAddresses(&x, &y, &z);
                index = (vector1 - 3) * 3;

                float *x_temp = new float[dimg];
                float *y_temp = new float[dimg];
                float *z_temp = new float[dimg];

                scalar_(VectFieldList[index], x_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }
                scalar_(VectFieldList[index + 1], y_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 1]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }
                scalar_(VectFieldList[index + 2], z_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 2]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }

                for (int r = 0; r < dimz; r++)
                    for (int s = 0; s < dimy; s++)
                        for (int t = 0; t < dimx; t++)
                        {
                            x[t * dimy * dimz + s * dimz + r] = x_temp[r * dimy * dimx + s * dimx + t];
                            y[t * dimy * dimz + s * dimz + r] = y_temp[r * dimy * dimx + s * dimx + t];
                            z[t * dimy * dimz + s * dimz + r] = z_temp[r * dimy * dimx + s * dimx + t];
                        }

                delete[] x_temp;
                delete[] y_temp;
                delete[] z_temp;

                outvector1[i - 1] = vector_out;
                outvector1[i] = NULL;
            }

            // vector data 2
            if (vector2 > 0)
            {
                int index;
                sprintf(buf, "%s_0_%d", p_outPort5->getObjName(), i);
                vector_out = new coDoVec3(buf, dimx, dimy, dimz);
                if (!vector_out->objectOk())
                {
                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort5->getName());
                    return FAIL;
                }
                vector_out->getAddresses(&x, &y, &z);

                index = (vector2 - 3) * 3;
                float *x_temp = new float[dimg];
                float *y_temp = new float[dimg];
                float *z_temp = new float[dimg];
                scalar_(VectFieldList[index], x_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }
                scalar_(VectFieldList[index + 1], y_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 1]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }
                scalar_(VectFieldList[index + 2], z_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 2]));
                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                    return FAIL;
                }

                for (int r = 0; r < dimz; r++)
                    for (int s = 0; s < dimy; s++)
                        for (int t = 0; t < dimx; t++)
                        {
                            x[t * dimy * dimz + s * dimz + r] = x_temp[r * dimy * dimx + s * dimx + t];
                            y[t * dimy * dimz + s * dimz + r] = y_temp[r * dimy * dimx + s * dimx + t];
                            z[t * dimy * dimz + s * dimz + r] = z_temp[r * dimy * dimx + s * dimx + t];
                        }

                delete[] x_temp;
                delete[] y_temp;
                delete[] z_temp;

                outvector2[i - 1] = vector_out;
                outvector2[i] = NULL;
            }
        } //end for i = 1 to ngrids

        rclose_(&tascflowError);
        if (tascflowError > 0)
        {
            sendError("ERROR: An error was occured while executing the routine: TGCLOS");
            return FAIL;
        }

        //create first object for every timesteps array
        if (timesteps > 1 && has_step_rso_file)
        {
            sprintf(buf, "%s_0", p_outPort1->getObjName());
            outputgrid = new coDoSet(buf, outgrd);
            time_outputgrid[0] = outputgrid;
        }
        else
        {
            sprintf(buf, "%s", p_outPort1->getObjName());
            outputgrid = new coDoSet(buf, outgrd);
            p_outPort1->setCurrentObject(outputgrid);
        }

        for (i = 0; i < ngrids; i++)
            delete outgrd[i];
        delete[] outgrd;

        if (timesteps > 1 && has_step_rso_file)
        {
            sprintf(buf, "%s_0", p_outPort2->getObjName());
            outputblock = new coDoSet(buf, outblock);
            time_outputblock[0] = outputblock;
        }
        else
        {
            sprintf(buf, "%s", p_outPort2->getObjName());
            outputblock = new coDoSet(buf, outblock);
            p_outPort2->setCurrentObject(outputblock);
        }

        for (i = 0; i < ngrids; i++)
            if (outblock[i] != NULL)
                delete outblock[i];
        //delete [] outblock;

        if (vector1 > 0)
        {
            if (timesteps > 1 && has_step_rso_file)
            {
                sprintf(buf, "%s_0", p_outPort4->getObjName());
                outputvector1 = new coDoSet(buf, outvector1);
                time_outputvector1[0] = outputvector1;
            }
            else
            {
                sprintf(buf, "%s", p_outPort4->getObjName());
                outputvector1 = new coDoSet(buf, outvector1);
                p_outPort4->setCurrentObject(outputvector1);
            }
            for (i = 0; i < ngrids; i++)
                delete outvector1[i];
        }

        if (vector2 > 0)
        {
            if (timesteps > 1 && has_step_rso_file)
            {
                sprintf(buf, "%s_0", p_outPort5->getObjName());
                outputvector2 = new coDoSet(buf, outvector2);
                time_outputvector2[0] = outputvector2;
            }
            else
            {
                sprintf(buf, "%s", p_outPort5->getObjName());
                outputvector2 = new coDoSet(buf, outvector2);
                p_outPort5->setCurrentObject(outputvector2);
            }
            for (i = 0; i < ngrids; i++)
                delete outvector2[i];
        }

        if (data1 > 0)
        {
            if (timesteps > 1 && has_step_rso_file)
            {
                sprintf(buf, "%s_0", p_outPort6->getObjName());
                outputdata1 = new coDoSet(buf, outdata1);
                time_outputdata1[0] = outputdata1;
            }
            else
            {
                sprintf(buf, "%s", p_outPort6->getObjName());
                outputdata1 = new coDoSet(buf, outdata1);
                p_outPort6->setCurrentObject(outputdata1);
            }
            for (i = 0; i < ngrids; i++)
                delete outdata1[i];
        }

        if (data2 > 0)
        {
            if (timesteps > 1 && has_step_rso_file)
            {
                sprintf(buf, "%s_0", p_outPort7->getObjName());
                outputdata2 = new coDoSet(buf, outdata2);
                time_outputdata2[0] = outputdata2;
            }
            else
            {
                sprintf(buf, "%s", p_outPort7->getObjName());
                outputdata2 = new coDoSet(buf, outdata2);
                p_outPort7->setCurrentObject(outputdata2);
            }
            for (i = 0; i < ngrids; i++)
                delete outdata2[i];
        }

        if (data3 > 0)
        {
            if (timesteps > 1 && has_step_rso_file)
            {
                sprintf(buf, "%s_0", p_outPort8->getObjName());
                outputdata3 = new coDoSet(buf, outdata3);
                time_outputdata3[0] = outputdata3;
            }
            else
            {
                sprintf(buf, "%s", p_outPort8->getObjName());
                outputdata3 = new coDoSet(buf, outdata3);
                p_outPort8->setCurrentObject(outputdata3);
                int n_elem;
                if ((col > 1) || (col == 1 && reg == 1))
                    n_elem = ngrids;
                else
                    n_elem = num_subreg;
                for (i = 0; i < n_elem; i++)
                    delete outdata3[i];
            }
        }

        p_outPort9->setCurrentObject(pol_out);

        //========================= TIMESTEPS ========================================

        int curr, numSteps = 1;
        if (has_step_rso_file && timesteps > 1)
        {
            int has_data1_field, has_data2_field, has_data3_field, has_vector1_field, has_vector2_field;
            int nb_scal_rso = 0;
            int nb_vect_rso = 0;
            int nb_fields_rso = 0;
            const char *const *VectStepRsoVal, *const *ScalStepRsoVal, *const *VectStepRsoList;
            char *next_path = NULL;
            coStepFile *step_file = new coStepFile(step_rsopath);
            step_file->set_skip_value(skip_value);
            // reading the rso fields

            for (curr = 1; curr < timesteps; curr++)
            {
                step_file->get_nextpath(&next_path);
                if (next_path)
                {
                    data_ok = open_database(gridpath, next_path, bcfpath, 1);
                    if (data_ok == SUCCESS)
                    {

                        char **All_Rso_Fields = new char *[MAX_VECT_FIELDS + MAX_SCAL_FIELDS];
                        int nb_rso_fields;
                        read_fields(&All_Rso_Fields, &nb_rso_fields);
                        Fields *rso_field = new Fields(All_Rso_Fields, nb_rso_fields);
                        rso_field->getAddresses(&VectStepRsoVal, &ScalStepRsoVal, &VectStepRsoList, &nb_vect_rso, &nb_scal_rso, &nb_fields_rso);

                        int i;
                        for (i = 0; i < nb_rso_fields; i++)
                            if (All_Rso_Fields[i])
                                delete[] All_Rso_Fields[i];

                        delete[] All_Rso_Fields;

                        has_data1_field = 0;
                        has_data2_field = 0;
                        has_data3_field = 0;
                        has_vector1_field = 0;
                        has_vector2_field = 0;
                        int j;
                        for (j = 0; j < nb_scal_rso; j++)
                        {
                            if (!strcmp(ScalChoiceVal[data1], ScalStepRsoVal[j]))
                                has_data1_field = 1;
                            if (!strcmp(ScalChoiceVal[data2], ScalStepRsoVal[j]))
                                has_data2_field = 1;
                            if (!strcmp(ScalChoiceVal[data3], ScalStepRsoVal[j]))
                                has_data3_field = 1;
                        }

                        for (j = 0; j < nb_vect_rso; j++)
                        {
                            if (!strcmp(VectChoiceVal[vector1], VectStepRsoVal[j]))
                                has_vector1_field = 1;
                            if (!strcmp(VectChoiceVal[vector2], VectStepRsoVal[j]))
                                has_vector2_field = 1;
                        }

                        //grid

                        time_outputgrid[numSteps] = outputgrid;
                        outputgrid->incRefCount();
                        time_outputgrid[numSteps + 1] = NULL;

                        for (i = 1; i <= ngrids; i++)
                        {
                            dims_(&dimx, &dimy, &dimz, &i, buffer, &tascflowError, MAX_GRIDNAME_LENGHT);
                            dimg = dimx * dimy * dimz;
                            if (tascflowError > 0)
                            {
                                sendError("ERROR: An error was occured while executing the routine: TRGDIM");
                                return FAIL;
                            }

                            //next block-off objects

                            sprintf(buf, "%s_%d_%d", p_outPort2->getObjName(), numSteps, i);
                            int dim[1];
                            dim[0] = (dimx - 1) * (dimy - 1) * (dimz - 1);
                            block_out = new coDoIntArr(buf, 1, dim);
                            block_out->getAddress(&block_array);

                            if (!block_out->objectOk())
                            {
                                sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort2->getName());
                                return FAIL;
                            }

                            block_out->getAddress(&block_array);
                            block_array_temp = new int[dimg];

                            blocko_(block_array_temp, &dimg, &i, RWK, &tascflowError);
                            if (tascflowError > 0)
                            {
                                sendError("ERROR: An error was occured while executing the routine: TRBLOF");
                                return FAIL;
                            }
                            int r, s, t;
                            for (t = 0; t < dimx - 1; t++)
                                for (s = 0; s < dimy - 1; s++)
                                    for (r = 0; r < dimz - 1; r++)
                                        block_array[t * (dimy - 1) * (dimz - 1) + s * (dimz - 1) + r] = block_array_temp[r * (dimy) * (dimx) + s * (dimx) + t];

                            delete[] block_array_temp;

                            outblock[i - 1] = block_out;
                            outblock[i] = NULL;

                            //scalar data1
                            if ((data1 > 1) && has_data1_field)
                            {

                                sprintf(buf, "%s_%d_%d", p_outPort6->getObjName(), numSteps, i);
                                data_out = new coDoFloat(buf, dimx, dimy, dimz);
                                if (!data_out->objectOk())
                                {
                                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort6->getName());
                                    return FAIL;
                                }
                                data_out->getAddress(&field);
                                float *field_temp = new float[dimg];
                                scalar_(ScalChoiceVal[data1], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data1]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }

                                for (int r = 0; r < dimz; r++)
                                    for (int s = 0; s < dimy; s++)
                                        for (int t = 0; t < dimx; t++)
                                            field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];
                                delete[] field_temp;

                                outdata1[i - 1] = data_out;
                                outdata1[i] = NULL;

                            } //if data1

                            //scalar data2
                            if ((data2 > 1) && has_data2_field)
                            {

                                sprintf(buf, "%s_%d_%d", p_outPort7->getObjName(), numSteps, i);
                                data_out = new coDoFloat(buf, dimx, dimy, dimz);
                                if (!data_out->objectOk())
                                {
                                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort7->getName());
                                    return FAIL;
                                }
                                data_out->getAddress(&field);
                                float *field_temp = new float[dimg];
                                scalar_(ScalChoiceVal[data2], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data2]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }
                                for (int r = 0; r < dimz; r++)
                                    for (int s = 0; s < dimy; s++)
                                        for (int t = 0; t < dimx; t++)
                                            field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];
                                delete[] field_temp;
                                outdata2[i - 1] = data_out;
                                outdata2[i] = NULL;

                            } //if data2

                            //scalar data3
                            if ((data3 > 1) && has_data3_field)
                            {

                                sprintf(buf, "%s_%d_%d", p_outPort8->getObjName(), numSteps, i);
                                data_out = new coDoFloat(buf, dimx, dimy, dimz);
                                if (!data_out->objectOk())
                                {
                                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort8->getName());
                                    return FAIL;
                                }
                                data_out->getAddress(&field);
                                float *field_temp = new float[dimg];
                                scalar_(ScalChoiceVal[data3], field_temp, &dimg, &i, RWK, &tascflowError, strlen(ScalChoiceVal[data3]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }
                                for (int r = 0; r < dimz; r++)
                                    for (int s = 0; s < dimy; s++)
                                        for (int t = 0; t < dimx; t++)
                                            field[t * dimy * dimz + s * dimz + r] = field_temp[r * dimy * dimx + s * dimx + t];
                                delete[] field_temp;

                                outdata3[i - 1] = data_out;
                                outdata3[i] = NULL;

                            } //if data3
                            //vector data 1
                            if ((vector1 > 1) && has_vector1_field)
                            {
                                sprintf(buf, "%s_%d_%d", p_outPort4->getObjName(), numSteps, i);
                                vector_out = new coDoVec3(buf, dimx, dimy, dimz);
                                if (!vector_out->objectOk())
                                {
                                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort4->getName());
                                    return FAIL;
                                }

                                vector_out->getAddresses(&x, &y, &z);

                                int index = (vector1 - 2) * 3;

                                float *x_temp = new float[dimg];
                                float *y_temp = new float[dimg];
                                float *z_temp = new float[dimg];
                                scalar_(VectFieldList[index], x_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }

                                scalar_(VectFieldList[index + 1], y_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 1]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }

                                scalar_(VectFieldList[index + 2], z_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 2]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }
                                for (int r = 0; r < dimz; r++)
                                    for (int s = 0; s < dimy; s++)
                                        for (int t = 0; t < dimx; t++)
                                        {
                                            x[t * dimy * dimz + s * dimz + r] = x_temp[r * dimy * dimx + s * dimx + t];
                                            y[t * dimy * dimz + s * dimz + r] = y_temp[r * dimy * dimx + s * dimx + t];
                                            z[t * dimy * dimz + s * dimz + r] = z_temp[r * dimy * dimx + s * dimx + t];
                                        }

                                delete[] x_temp;
                                delete[] y_temp;
                                delete[] z_temp;

                                outvector1[i - 1] = vector_out;
                                outvector1[i] = NULL;
                            } //if vector1

                            //vector data 2
                            if ((vector2 > 1) && has_vector2_field)
                            {

                                sprintf(buf, "%s_%d_%d", p_outPort5->getObjName(), numSteps, i);
                                vector_out = new coDoVec3(buf, dimx, dimy, dimz);
                                if (!vector_out->objectOk())
                                {
                                    sendError("Failed to create the object '%s' for the port '%s'", buf, p_outPort5->getName());
                                    return FAIL;
                                }

                                vector_out->getAddresses(&x, &y, &z);

                                int index = (vector2 - 2) * 3;

                                float *x_temp = new float[dimg];
                                float *y_temp = new float[dimg];
                                float *z_temp = new float[dimg];
                                scalar_(VectFieldList[index], x_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }

                                scalar_(VectFieldList[index + 1], y_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 1]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }

                                scalar_(VectFieldList[index + 2], z_temp, &dimg, &i, RWK, &tascflowError, strlen(VectFieldList[index + 2]));
                                if (tascflowError > 0)
                                {
                                    sendError("ERROR: An error was occured while executing the routine: TRSCAL");
                                    return FAIL;
                                }

                                for (int r = 0; r < dimz; r++)
                                    for (int s = 0; s < dimy; s++)
                                        for (int t = 0; t < dimx; t++)
                                        {
                                            x[t * dimy * dimz + s * dimz + r] = x_temp[r * dimy * dimx + s * dimx + t];
                                            y[t * dimy * dimz + s * dimz + r] = y_temp[r * dimy * dimx + s * dimx + t];
                                            z[t * dimy * dimz + s * dimz + r] = z_temp[r * dimy * dimx + s * dimx + t];
                                        }

                                delete[] x_temp;
                                delete[] y_temp;
                                delete[] z_temp;

                                outvector2[i - 1] = vector_out;
                                outvector2[i] = NULL;
                            } //if vector2
                        } //for i=1 to ngrids

                        sprintf(buf, "%s_%d", p_outPort2->getObjName(), numSteps);
                        outputblock = new coDoSet(buf, outblock);
                        for (i = 0; i < ngrids; i++)
                            if (outblock[i] != NULL)
                                delete outblock[i];
                        //delete [] outblock;
                        time_outputblock[numSteps] = outputblock;
                        time_outputblock[numSteps + 1] = NULL;

                        if (data1 > 1)
                        {
                            if (has_data1_field)
                            {
                                sprintf(buf, "%s_%d", p_outPort6->getObjName(), numSteps);
                                outputdata1 = new coDoSet(buf, outdata1);
                                for (i = 0; i < ngrids; i++)
                                    delete outdata1[i];
                                time_outputdata1[numSteps] = outputdata1;
                            }
                            else
                            {
                                time_outputdata1[numSteps] = outputdata1;
                                outputdata1->incRefCount();
                            }
                            time_outputdata1[numSteps + 1] = NULL;
                        }
                        if (data2 > 1)
                        {
                            if (has_data2_field)
                            {
                                sprintf(buf, "%s_%d", p_outPort7->getObjName(), numSteps);
                                outputdata2 = new coDoSet(buf, outdata2);
                                for (i = 0; i < ngrids; i++)
                                    delete outdata2[i];
                                time_outputdata2[numSteps] = outputdata2;
                            }
                            else
                            {
                                time_outputdata2[numSteps] = outputdata2;
                                outputdata2->incRefCount();
                            }
                            time_outputdata2[numSteps + 1] = NULL;
                        }
                        if (data3 > 1)
                        {
                            if (has_data3_field)
                            {
                                sprintf(buf, "%s_%d", p_outPort8->getObjName(), numSteps);
                                outputdata3 = new coDoSet(buf, outdata3);
                                for (i = 0; i < ngrids; i++)
                                    delete outdata3[i];
                                time_outputdata3[numSteps] = outputdata3;
                            }
                            else
                            {
                                time_outputdata3[numSteps] = outputdata3;
                                outputdata3->incRefCount();
                            }
                            time_outputdata3[numSteps + 1] = NULL;
                        }
                        if (vector1 > 1)
                        {
                            if (has_vector1_field)
                            {
                                sprintf(buf, "%s_%d", p_outPort4->getObjName(), numSteps);
                                outputvector1 = new coDoSet(buf, outvector1);
                                for (i = 0; i < ngrids; i++)
                                    delete outvector1[i];
                                time_outputvector1[numSteps] = outputvector1;
                            }
                            else
                            {
                                time_outputvector1[numSteps] = outputvector1;
                                outputvector1->incRefCount();
                            }
                            time_outputvector1[numSteps + 1] = NULL;
                        }
                        if (vector2 > 1)
                        {
                            if (has_vector2_field)
                            {
                                sprintf(buf, "%s_%d", p_outPort5->getObjName(), numSteps);
                                outputvector2 = new coDoSet(buf, outvector2);
                                for (i = 0; i < ngrids; i++)
                                    delete outvector2[i];
                                time_outputvector2[numSteps] = outputvector2;
                            }
                            else
                            {
                                time_outputvector2[numSteps] = outputvector2;
                                outputvector2->incRefCount();
                            }
                            time_outputvector2[numSteps + 1] = NULL;
                        }
                        rclose_(&tascflowError);
                        if (tascflowError > 0)
                        {
                            sendError("ERROR: An error was occured while executing the routine: TGCLOS");
                            return FAIL;
                        }
                        numSteps++;

                        delete rso_field;

                    } //if data_ok
                    delete[] next_path;
                }
            } //for curr = 1 to timesteps
            delete step_file;
        }
        else if (timesteps > 1)
        {
            sendError("Please indicate the steprsopath (first file of timestep data)!");
            return FAIL;
        }

        //build objects for the output ports
        if (numSteps > 1 && has_step_rso_file)
        {
            coDoSet *time_grd = new coDoSet(p_outPort1->getObjName(), time_outputgrid);
            sprintf(buf, "1 %d", numSteps);
            time_grd->addAttribute("TIMESTEP", buf);

            for (i = 0; i < numSteps; i++)
                delete time_outputgrid[i];
            delete[] time_outputgrid;

            p_outPort1->setCurrentObject(time_grd);

            coDoSet *time_block = new coDoSet(p_outPort2->getObjName(), time_outputblock);
            sprintf(buf, "1 %d", numSteps);
            time_block->addAttribute("TIMESTEP", buf);
            for (i = 0; i < numSteps; i++)
                delete time_outputblock[i];
            delete[] time_outputblock;

            p_outPort2->setCurrentObject(time_block);

            //data
            if (vector1 > 1)
            {
                coDoSet *time_vector1 = new coDoSet(p_outPort4->getObjName(), time_outputvector1);

                time_vector1->addAttribute("TIMESTEP", buf);
                for (i = 1; i < numSteps; i++)
                    delete time_outputvector1[i];
                delete[] time_outputvector1;
                // delete [] outvector1;

                p_outPort4->setCurrentObject(time_vector1);
            }

            if (vector2 > 1)
            {
                coDoSet *time_vector2 = new coDoSet(p_outPort5->getObjName(), time_outputvector2);
                time_vector2->addAttribute("TIMESTEP", buf);
                for (i = 1; i < numSteps; i++)
                    delete time_outputvector2[i];
                delete[] time_outputvector2;
                //delete [] outvector2;

                p_outPort5->setCurrentObject(time_vector2);
            }

            if (data1 > 1)
            {
                coDoSet *time_data1 = new coDoSet(p_outPort6->getObjName(), time_outputdata1);
                time_data1->addAttribute("TIMESTEP", buf);
                for (i = 1; i < numSteps; i++)
                    delete time_outputdata1[i];
                delete[] time_outputdata1;
                //delete [] outdata1;
                p_outPort6->setCurrentObject(time_data1);
            }

            if (data2 > 1)
            {
                coDoSet *time_data2 = new coDoSet(p_outPort7->getObjName(), time_outputdata2);
                time_data2->addAttribute("TIMESTEP", buf);
                delete[] time_outputdata2;
                p_outPort7->setCurrentObject(time_data2);
                //delete [] outdata2;
            }

            if (data3 > 1)
            {
                coDoSet *time_data3 = new coDoSet(p_outPort8->getObjName(), time_outputdata3);
                time_data3->addAttribute("TIMESTEP", buf);
                delete[] time_outputdata3;
                //delete [] outdata3;
                p_outPort8->setCurrentObject(time_data3);
            }
        }

        if (vector1 > 1)
            delete[] outvector1;
        if (vector2 > 1)
            delete[] outvector2;
        if (data1 > 1)
            delete[] outdata1;
        if (data2 > 1)
            delete[] outdata2;
        if (data3 > 1)
            delete[] outdata3;
        delete[] outblock;
        //regions
        if (reg > 1 && n_sreg > 0)
        {
            coDoSet *out_region = new coDoSet(p_outPort3->getObjName(), outregion);
            for (int j = 0; j < n_sreg; j++)
                if (outregion[j] != NULL)
                    delete outregion[j];
            p_outPort3->setCurrentObject(out_region);
        }
        else
        {
            if (col > 1)
            {
                coDoLines *empty_lines = new coDoLines(p_outPort3->getObjName(), 0, 0, 0);
                p_outPort3->setCurrentObject(empty_lines);
            }
            else
            {
                coDoStructuredGrid *empty_region_grid = new coDoStructuredGrid(p_outPort3->getObjName(), 0, 0, 0);
                p_outPort3->setCurrentObject(empty_region_grid);
            }
        }

        delete[] outregion;
        for (i = 0; i < ngrids; i++)
            if (GridNames != NULL)
                delete[] GridNames[i];

        delete[] GridNames;

    } //if data_ok == SUCCESS

    if (bcf != NULL)
        delete bcf;
    return SUCCESS;
}

//===================private functions==================================

int TascFlow::open_database(char *grid_path, char *rso_path, char *bcf_path, int in_execution)
{

    if (has_grd_file)
    {
        FILE *fd;
        int lengrd = strlen(grid_path);

        fd = fopen(gridpath, "r");
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
                    sendError("ERROR: An error was occured while executing the routine: TGINIT");
                    return FAIL;
                }

                file_("grd", grid_path, fform, &tascflowError, 3, lengrd, 1);

                if (tascflowError > 0)
                {
                    sendError("ERROR: An error was occured while executing the routine: TGINIT");
                    return FAIL;
                }
                if (has_rso_file)
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
                        }
                        else
                        {
                            //sendError("%s is directory!",rso_path);
                            has_rso_file = 0;
                        }
                    }
                    else
                    {
                        sendError("ERROR: Could not open Rsofile: %s", rso_path);
                        has_rso_file = 0;
                    }

                } //if has_rso_file

                if (has_bcf_file)
                {
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
                        }
                        else
                        {
                            //sendError("%s is directory!",bcf_path);
                            has_bcf_file = 0;
                        }
                    }
                    else
                    {
                        sendError("ERROR: Could not open Bcf file: %s", bcf_path);
                        has_bcf_file = 0;
                    }
                } //if has_bcf_file
            }
            else
            {
                sendError("%s is directory!", grid_path);
                has_grd_file = 0;
            }
        }
        else
        {
            sendError("ERROR: Could not open Grdfile: %s", grid_path);
            return FAIL;
        }

        sendInfo("Reading the input files! Please wait...");
        rread_(RWK, IWK, CWK, &tascflowError, nc);

        if (tascflowError > 0)
        {
            if (in_execution)
                sendError("ERROR: An error was occured while executing the routine: TRREAD");
            else
                sendWarning("WARNING: Cannot execute the routine TRREAD. Please check if the files match.");
            return FAIL;
        }
        sendInfo("The files have been read successfully!");
    }
    else
    {
        if (in_execution)
            sendError("Please indicate the grid file!");
        return FAIL;
    }

    return SUCCESS;
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
        sendError("ERROR: An error was occured while executing the routine: TRNXTS");
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
            sendError("ERROR: An error was occured while executing the routine: TRNXTS");
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
        sendError("ERROR: An error was occured while executing the routine: TRNXTR");
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
            sendError("ERROR: An error was occured while executing the routine: TRNXTS");
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

MODULE_MAIN(IO, TascFlow)
