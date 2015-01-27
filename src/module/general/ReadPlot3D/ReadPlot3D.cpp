/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:   COVISE ReadPlot3D application module                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Lars Frenzel                                                  **
 **                                                                        **
 **                                                                        **
 ** Date: 26.09.97                                                         **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include "ReadPlot3D.h"

#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

using namespace boost;

//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    Application *application = new Application(argc, argv);

    // and back to covise
    application->run();

    // done
    return 1;
}

Application::Application(int argc, char *argv[])
{
    Covise::set_module_description("Read Plot3D files");

    Covise::add_port(OUTPUT_PORT, "grid", "StructuredGrid|UnstructuredGrid", "grid out");
    Covise::add_port(OUTPUT_PORT, "data1", "Float|Vec3|Float|Vec3", "data out / density");
    Covise::add_port(OUTPUT_PORT, "data2", "Float|Vec3|Vec3", "x,-,y-, z-momentum");
    Covise::add_port(OUTPUT_PORT, "data3", "Float|Float", "energy per volume unit");
    Covise::add_port(OUTPUT_PORT, "data4", "Float|Float", "energy per volume unit");
    Covise::add_port(OUTPUT_PORT, "data5", "Float|Float", "energy per volume unit");
    Covise::add_port(OUTPUT_PORT, "data6", "Float|Float", "energy per volume unit");
    Covise::add_port(OUTPUT_PORT, "data7", "Float|Float", "energy per volume unit");
    Covise::add_port(OUTPUT_PORT, "iblank", "IntArr", "iblank values, random if not iblank");

    Covise::add_port(PARIN, "grid_path", "Browser", "file grid_path");
    Covise::set_port_default("grid_path", "data/nofile");

    Covise::add_port(PARIN, "grid_path___filter", "BrowserFilter", "file grid_path");
    Covise::set_port_default("grid_path___filter", "grid_path *");

    Covise::add_port(PARIN, "data_path", "Browser", "file data_path");
    Covise::set_port_default("data_path", "data/nofile");

    Covise::add_port(PARIN, "data_path___filter", "BrowserFilter", "file data_path");
    Covise::set_port_default("data_path___filter", "data_path *");

    Covise::add_port(PARIN, "gridtype", "Choice", "???");
    Covise::set_port_default("gridtype", "1 Structured Unstructured IBLANKed");

    Covise::add_port(PARIN, "subtype", "Choice", "???");
    Covise::set_port_default("subtype", "1 Single Multi");

    Covise::add_port(PARIN, "datatype", "Choice", "???");
    Covise::set_port_default("datatype", "1 Solution Scalar/Vector Function");

    Covise::add_port(PARIN, "remove_timesteps", "IntScalar", "After each timestep remove n timesteps");
    Covise::set_port_default("remove_timesteps", "0");

    Covise::add_port(PARIN, "max_timesteps", "IntScalar", "Trims the number of timesteps");
    Covise::set_port_default("max_timesteps", "999999");

    Covise::init(argc, argv);
    Covise::set_start_callback(Application::computeCallback, this);

    new_buffer = 1;
    blockSize64 = 0;
    blockSize = 0;
}

//////
////// computeCallback (do nothing but call our real compute-function)
//////

void Application::computeCallback(void *userData, void *)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(NULL);
}

//////
////// this is our compute-routine
//////

void Application::compute(const char *)
{
    FILE *fData, *fGrid;

    // get parameters
    Covise::get_browser_param("grid_path", &grid_path);
    Covise::get_browser_param("data_path", &data_path);

    Covise::get_choice_param("gridtype", &gridtype);
    Covise::get_choice_param("datatype", &datatype);
    Covise::get_choice_param("subtype", &subtype);

    Covise::get_scalar_param("remove_timesteps", &remove_timesteps);
    Covise::get_scalar_param("max_timesteps", &max_timesteps);

    // compute parameters
    if ((fGrid = Covise::fopen(grid_path, "rb")) <= 0)
    {
        Covise::sendError("ERROR: can't open file %s", grid_path);
        return;
    }

    filetype = _FILE_TYPE_ASCII;
    byteswap_flag = 0;
    int k = 0;
    while (!feof(fGrid) && k < 1000)
    {
        char c = getc(fGrid);
        if (!isprint(c) && c != '\n' && c != '\r')
            filetype = _FILE_TYPE_BINARY;
        k++;
    }

    fclose(fGrid);

    if (filetype == _FILE_TYPE_BINARY)
    {
        if ((fGrid = Covise::fopen(grid_path, "rb")) <= 0)
        {
            Covise::sendError("ERROR: can't open file %s", grid_path);
            return;
        }
        rewind(fGrid);
        //here test FORTRAN!!
        int bsize1 = 0, bsize2 = 0, numBlocks = 0;
        //char* array;
        int seekRes = 0;

        while (bsize1 == bsize2 && numBlocks < 3)
        {
            //file_read( fGrid, &bsize1, 1 );
            if (fread(&bsize1, sizeof(int), 1, fGrid) != 1)
            {
                cerr << "ReadPlot3D::compute: fread1 failed" << endl;
            }
            if (numBlocks == 0 && (bsize1 > 1000 || bsize1 < 0))
            {
                byteswap_flag = 1;
                byteswap(&bsize1, sizeof(int));
            }
#ifdef __sgi
            seekRes = fseek64(fGrid, bsize1, SEEK_CUR);
#else
            seekRes = fseek(fGrid, bsize1, SEEK_CUR);
#endif
            if (seekRes)
            {
                numBlocks = -1;
                break;
            }
            else
            {
                if (fread(&bsize2, sizeof(int), 1, fGrid) != 1)
                {
                    cerr << "ReadPlot3D::compute: fread2 failed" << endl;
                }
                //file_read( fGrid, &bsize2, 1 );
                if (byteswap_flag)
                    byteswap(&bsize2, sizeof(int));
                numBlocks++;
            }
        }
        if (numBlocks == 3)
            filetype = _FILE_TYPE_FORTRAN;
        else
        {
            rewind(fGrid);
            //here test FORTRAN!!
            coInt64 bsize1 = 0, bsize2 = 0;
            int numBlocks = 0;
            //char* array;
            int seekRes = 0;

            while (bsize1 == bsize2 && numBlocks < 3)
            {
                //file_read( fGrid, &bsize1, 1 );
                if (fread(&bsize1, sizeof(coInt64), 1, fGrid) != 1)
                {
                    cerr << "ReadPlot3D::compute: fread1 failed" << endl;
                }
                if (numBlocks == 0 && (bsize1 > 1000 || bsize1 < 0))
                {
                    byteswap_flag = 1;
                    byteswap(&bsize1, sizeof(coInt64));
                }
#ifdef __sgi
                seekRes = fseek64(fGrid, bsize1, SEEK_CUR);
#else
                seekRes = fseek(fGrid, bsize1, SEEK_CUR);
#endif
                if (seekRes)
                {
                    numBlocks = -1;
                    break;
                }
                else
                {
                    if (fread(&bsize2, sizeof(coInt64), 1, fGrid) != 1)
                    {
                        cerr << "ReadPlot3D::compute: fread2 failed" << endl;
                    }
                    //file_read( fGrid, &bsize2, 1 );
                    if (byteswap_flag)
                        byteswap(&bsize2, sizeof(coInt64));
                    numBlocks++;
                }
            }
            if (numBlocks == 3)
                filetype = _FILE_TYPE_FORTRAN64;
            //else
            //   byteswap_flag = 0;
        }
        fclose(fGrid);
    }

    if (filetype == _FILE_TYPE_ASCII)
    {
        if ((fGrid = Covise::fopen(grid_path, "r")) <= 0)
        {
            Covise::sendError("ERROR: can't open file %s", grid_path);
            return;
        }
        if ((fData = Covise::fopen(data_path, "r")) <= 0)
        {
            Covise::sendInfo("WARNING: no data-file selected");
            data_path = NULL;
        }
    }
    else
    {
        if ((fGrid = Covise::fopen(grid_path, "rb")) <= 0)
        {
            Covise::sendError("ERROR: can't open file %s", grid_path);
            return;
        }
        if ((fData = Covise::fopen(data_path, "rb")) <= 0)
        {
            Covise::sendInfo("WARNING: no data-file selected");
            data_path = NULL;
        }
    }
    rewind(fGrid);

    switch (gridtype)
    {
    case 1:
        gridtype = _FILE_STRUCTURED_GRID;
        break;
    case 2:
        gridtype = _FILE_UNSTRUCTURED_GRID;
        break;
    case 3:
        gridtype = _FILE_IBLANKED;
        break;
    }
    switch (subtype)
    {
    case 1:
        subtype = _FILE_SINGLE_ZONE;
        break;
    case 2:
        subtype = _FILE_MULTI_ZONE;
        break;
    }
    switch (datatype)
    {
    case 1:
        datatype = _FILE_SOLUTION;
        break;
    case 2:
        datatype = _FILE_DATA;
        break;
    case 3:
        datatype = _FILE_FUNCTION;
        break;
    }

    // get list of data files

    std::vector<std::string> data_paths;
    try
    {
        filesystem::path fullpath(data_path);
        filesystem::path path_only = fullpath.parent_path();
        filesystem::path filename_only = fullpath.filename();
        regex pattern(filename_only.string());
        for (filesystem::directory_iterator iter(path_only), end; iter != end; ++iter)
        {
            string name = iter->path().filename().string();
            if (regex_match(name, pattern))
            {
                data_paths.push_back(iter->path().string());
            }
        }
    }
    catch (...)
    {
        data_paths.clear();
    };
    std::sort(data_paths.begin(), data_paths.end());
    if (data_paths.size() == 0)
    {
        Covise::sendInfo("WARNING: no data-file matches the regular expression");
    }

    // reduce timesteps

    std::vector<std::string>::iterator it = data_paths.begin();
    int cnt(0);
    while (it != data_paths.end())
    {
        if (cnt < remove_timesteps)
        {
            it = data_paths.erase(it);
            ++cnt;
        }
        else
        {
            ++it;
            cnt = 0;
        }
    }
    while (data_paths.size() > max_timesteps)
    {
        data_paths.pop_back();
    }

    // prepare

    char timestepAttribute[20];
    sprintf(timestepAttribute, "0 %d", (int)data_paths.size());

    // READ GRID

    Covise::sendInfo("Reading the grid. Please wait ...");
    new_buffer = 1;
    const char *grid_name = Covise::get_object_name("grid");
    char bfr[500];
    if (data_paths.size() > 1)
    {
        sprintf(bfr, "%s_%s", grid_name, "timestep");
    }
    else
    {
        sprintf(bfr, "%s", grid_name);
    }
    coDistributedObject **obj = ReadPlot3D(fGrid, _READ_GRID, bfr, "", "");
    if ((obj == NULL) || (obj[0] == NULL))
    {
        fclose(fGrid);
        Covise::sendInfo("The input grid was not succesfully read.");
        return;
    }
    fclose(fGrid);
    if (data_paths.size() > 1)
    {
        coDistributedObject **array = new coDistributedObject *[data_paths.size() + 1];
        for (int i = 0; i < data_paths.size(); ++i)
        {
            array[i] = obj[0];
            if (i != 0)
                obj[0]->incRefCount();
        }
        array[data_paths.size()] = NULL;
        coDoSet *s = new coDoSet(grid_name, array);
        s->addAttribute("TIMESTEP", timestepAttribute);
        delete[] array;
    }
    Covise::sendInfo("The input grid was read.");

    // READ DATA

    if (data_paths.size() == 1)
    {
        if ((fData = Covise::fopen(data_paths[0].c_str(), "r")) > 0)
        {
            new_buffer = 1;
            Covise::sendInfo("Reading the data. Please wait ...");
            ReadPlot3D(fData, _READ_DATA, Covise::get_object_name("data1"), Covise::get_object_name("data2"), Covise::get_object_name("data3"));
            fclose(fData);
            Covise::sendInfo("The input data was read.");
        }
        else
        {
            Covise::sendInfo("WARNING: data-file could not be opened");
        }
    }
    else if (data_paths.size() > 1)
    {
        Covise::sendInfo("Reading the data. Please wait ...");
        coDistributedObject **data1 = new coDistributedObject *[data_paths.size() + 1];
        coDistributedObject **data2 = new coDistributedObject *[data_paths.size() + 1];
        coDistributedObject **data3 = new coDistributedObject *[data_paths.size() + 1];
        const char *name1 = Covise::get_object_name("data1");
        const char *name2 = Covise::get_object_name("data2");
        const char *name3 = Covise::get_object_name("data3");
        char bfr1[500];
        char bfr2[500];
        char bfr3[500];
        for (int i = 0; i < data_paths.size(); ++i)
        {
            sprintf(bfr1, "%s_%d", name1, i);
            sprintf(bfr2, "%s_%d", name2, i);
            sprintf(bfr3, "%s_%d", name3, i);
            fData = Covise::fopen(data_paths[i].c_str(), "r");
            new_buffer = 1;
            coDistributedObject **result = ReadPlot3D(fData, _READ_DATA, bfr1, bfr2, bfr3);
            fclose(fData);
            data1[i] = result[0];
            data2[i] = result[1];
            data3[i] = result[2];
        }
        data1[data_paths.size()] = NULL;
        data2[data_paths.size()] = NULL;
        data3[data_paths.size()] = NULL;
        if (data1[0] != NULL)
        {
            coDoSet *s = new coDoSet(name1, data1);
            s->addAttribute("TIMESTEP", timestepAttribute);
        }
        if (data2[0] != NULL)
        {
            coDoSet *s = new coDoSet(name2, data2);
            s->addAttribute("TIMESTEP", timestepAttribute);
        }
        if (data3[0] != NULL)
        {
            coDoSet *s = new coDoSet(name3, data3);
            s->addAttribute("TIMESTEP", timestepAttribute);
        }
        delete[] data1;
        delete[] data2;
        delete[] data3;
        Covise::sendInfo("The input data was read.");
    }

    // done
    return;
}

//////
////// the read-function itself
//////

coDistributedObject **Application::ReadPlot3D(FILE *fFile, int read_flag, const char *name1, const char *name2, const char *name3)
{

    // output stuff
    int npoints, nzones;

    float *x_coord, *y_coord, *z_coord;
    int *el_out, *tl_out, *vl_out;
    float *data_out[3];
    int x_dim, y_dim, z_dim;
    int *zones_x_dim, *zones_y_dim, *zones_z_dim;
    int *zones_comp;
    int numComp = 0;
    long sfPos;

    int num_elem, num_vert, num_coord;
    int num_triang, num_tetra;

    float mach, re, alpha, time_var;

    // output objects
    coDoStructuredGrid *str_grid = NULL;
    coDoUnstructuredGrid *unstr_grid = NULL;

    coDoFloat *unstr_s3d_out = NULL;
    coDoVec3 *unstr_v3d_out = NULL;

    coDoSet *set_out = NULL;
    coDistributedObject **set_out_objs = NULL;
    coDistributedObject **sol_set_out_objs[3];

    // this leaks 16 bytes of memory, but better than returning static var
    coDistributedObject **return_object = new coDistributedObject *[4];
    return_object[0] = return_object[1] = return_object[2] = return_object[3] = NULL;
    //coDistributedObject *return_object[] = {NULL, NULL, NULL, NULL};

    // counters
    int i, j, k;

    // computed stuff

    // temp stuff
    char bfr[500];

    // here we go

    //////
    ////// STRUCTURED_GRID / IBLANKed
    //////

    if ((gridtype == _FILE_STRUCTURED_GRID || gridtype == _FILE_IBLANKED) && read_flag == _READ_GRID)
    {
        if (subtype == _FILE_SINGLE_ZONE)
        {
            // get grid size
            read_size_header(fFile, &x_dim, &y_dim, &z_dim, &npoints);
            if (x_dim <= 0 || y_dim <= 0 || z_dim <= 0 || npoints <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }
            // show information to user
            Covise::sendInfo("IDIM: %d  JDIM: %d  KDIM: %d  NPOINTS: %d", x_dim, y_dim, z_dim, npoints);

            // create object and get pointers
            str_grid = new coDoStructuredGrid(name1, x_dim, y_dim, z_dim);
            str_grid->getAddresses(&x_coord, &y_coord, &z_coord);

            // create the iblank field if necessary
            int *iblank = NULL;
            if (gridtype == _FILE_IBLANKED)
            {
                const char *obj_name = Covise::get_object_name("iblank");
                int size[1];
                size[0] = npoints;
                coDoIntArr *celltab = new coDoIntArr(obj_name, 1, size);
                celltab->getAddress(&iblank);
            }

            // read in the coordinates

            read_structured_grid_record(fFile, x_coord, x_dim, y_coord, y_dim,
                                        z_coord, z_dim, npoints, iblank, gridtype);

            // skip IBLANKed-values
            //if( gridtype==_FILE_IBLANKED )
            //   read_iblanked( fFile, npoints );

            // done
            return_object[0] = str_grid;
        }
        else if (subtype == _FILE_MULTI_ZONE)
        {
            // get nzones
            read_nzones(fFile, &nzones);

            if (nzones == 0)
            {
                Covise::sendError("Found no zones in file");
                return NULL;
            }

            // init sets : OBJ + iblank
            set_out_objs = new coDistributedObject *[nzones + 1];
            set_out_objs[nzones] = NULL;

            coDistributedObject **ibl_out_objs = new coDistributedObject *[nzones + 1];
            ibl_out_objs[nzones] = NULL;

            // show information
            Covise::sendInfo("NZONES: %d", nzones);

            // get grid sizes
            zones_x_dim = new int[nzones];
            zones_y_dim = new int[nzones];
            zones_z_dim = new int[nzones];
            read_multi_header(fFile, &zones_x_dim, &zones_y_dim, &zones_z_dim, nzones);
            for (i = 0; i < nzones; i++)
            {
                if (zones_x_dim[i] <= 0 || zones_y_dim[i] <= 0 || zones_z_dim[i] <= 0)
                {
                    Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                    return NULL;
                }
            }
            // get object name
            char *ibl_name = Covise::get_object_name("iblank");

            // work through the set
            for (i = 0; i < nzones; i++)
            {
                // compute grid size
                x_dim = zones_x_dim[i];
                y_dim = zones_y_dim[i];
                z_dim = zones_z_dim[i];
                npoints = x_dim * y_dim * z_dim;

                // show information
                Covise::sendInfo("Zone: %d  IDIM: %d  JDIM: %d  KDIM: %d  NPOINTS: %d", i + 1, x_dim, y_dim, z_dim, npoints);

                // compute object name
                sprintf(bfr, "%s_%d", name1, i);

                // create object and get pointers
                str_grid = new coDoStructuredGrid(bfr, x_dim, y_dim, z_dim);
                str_grid->getAddresses(&x_coord, &y_coord, &z_coord);

                // create the iblank field if necessary
                int *iblank = NULL;
                if (gridtype == _FILE_IBLANKED)
                {
                    sprintf(bfr, "%s_%d", ibl_name, i);
                    int size[1];
                    size[0] = npoints;
                    coDoIntArr *celltab = new coDoIntArr(bfr, 1, size);
                    celltab->getAddress(&iblank);
                    ibl_out_objs[i] = celltab;
                }

                // read in the coordinates
                read_structured_grid_record(fFile, x_coord, x_dim, y_coord, y_dim,
                                            z_coord, z_dim, npoints, iblank, gridtype);

                // skip IBLANKed-values
                //if( gridtype==_FILE_IBLANKED )
                //   read_iblanked( fFile, npoints );

                // add element to set
                set_out_objs[i] = str_grid;
            }

            // create sets
            set_out = new coDoSet(name1, set_out_objs);

            if (gridtype == _FILE_IBLANKED)
                new coDoSet(ibl_name, ibl_out_objs);

            // clean up
            //for( i=0; i<nzones; i++ )
            //   delete set_out_objs[i];
            delete[] set_out_objs;

            delete[] zones_x_dim;
            delete[] zones_y_dim;
            delete[] zones_z_dim;

            // done
            return_object[0] = set_out;
        }
        else
        {
            Covise::sendError("ERROR: subtype not available");
            return (NULL);
        }
    }

    //////
    ////// UNSTRUCTURED_GRID
    //////

    if (gridtype == _FILE_UNSTRUCTURED_GRID && read_flag == _READ_GRID)
    {
        if (subtype == _FILE_SINGLE_ZONE)
        {
            // get size
            read_unstructured_header(fFile, &num_coord, &num_triang, &num_tetra);

            if (num_coord <= 0 || num_triang + num_tetra <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }

            // show information
            Covise::sendInfo("num_coord: %d  num_triang: %d  num_tetra: %d", num_coord, num_triang, num_tetra);

            // get object name and pointers
            unstr_grid = new coDoUnstructuredGrid(name1, num_triang + num_tetra,
                                                  (num_triang * 3) + (num_tetra * 4), num_coord, 1);
            unstr_grid->getAddresses(&el_out, &vl_out, &x_coord, &y_coord, &z_coord);
            unstr_grid->getTypeList(&tl_out);
            return_object[0] = unstr_grid;

            // read in the coordinates

            read_unstructured_coord(fFile, x_coord, y_coord, z_coord, num_coord);

            // init usg
            num_elem = 0;
            num_vert = 0;

            // read in the triangles
            for (i = 0; i < num_triang; i++)
            {
                tl_out[num_elem] = TYPE_TRIANGLE;
                el_out[num_elem] = num_vert;
                num_elem++;

                read_single_triangle(fFile, &(vl_out[num_vert]), &(vl_out[num_vert + 1]),
                                     &(vl_out[num_vert + 2]));
                if (vl_out[num_vert] > num_coord || vl_out[num_vert + 1] > num_coord || vl_out[num_vert + 2] > num_coord)
                {
                    Covise::sendError("The grid is not an unstructured one.");
                    return NULL;
                }
                num_vert += 3;
            }

            // skip triangle-flags
            read_unstructured_triangle_flags(fFile, num_triang);

            // read in the tetrahedras
            for (i = 0; i < num_tetra; i++)
            {
                tl_out[num_elem] = TYPE_TETRAHEDER;
                el_out[num_elem] = num_vert;
                num_elem++;

                read_single_tetrahedra(fFile, &(vl_out[num_vert]), &(vl_out[num_vert + 1]),
                                       &(vl_out[num_vert + 2]), &(vl_out[num_vert + 3]));
                if (vl_out[num_vert] > num_coord || vl_out[num_vert + 1] > num_coord || vl_out[num_vert + 2] > num_coord || vl_out[num_vert + 3] > num_coord)
                {
                    Covise::sendError("The grid is not an unstructured one.");
                    return NULL;
                }
                num_vert += 4;
            }

            // done
        }
        else if (subtype == _FILE_MULTI_ZONE)
        {
            // get nzones
            read_nzones(fFile, &nzones);
            if (nzones <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }

            // init set
            set_out_objs = new coDistributedObject *[nzones + 1];
            set_out_objs[nzones] = NULL;

            // show information
            Covise::sendInfo("NZONES: %d", nzones);

            // get grid sizes
            zones_x_dim = new int[nzones];
            zones_y_dim = new int[nzones];
            zones_z_dim = new int[nzones];

            for (i = 0; i < nzones; i++)
                read_unstructured_header(fFile, &(zones_x_dim[i]), &(zones_y_dim[i]),
                                         &(zones_z_dim[i]));

            // work through the set
            for (i = 0; i < nzones; i++)
            {
                // get size
                num_coord = zones_x_dim[i];
                num_triang = zones_y_dim[i];
                num_tetra = zones_z_dim[i];

                // show information
                Covise::sendInfo("Zone: %d  num_coord: %d  num_triang: %d  num_tetra: %d", i, num_coord, num_triang, num_tetra);

                // compute object name and get pointers
                sprintf(bfr, "%s_%d", name1, i);
                unstr_grid = new coDoUnstructuredGrid(bfr, num_triang + num_tetra,
                                                      (num_triang * 3) + (num_tetra * 4), num_coord, 1);
                unstr_grid->getAddresses(&el_out, &vl_out, &x_coord, &y_coord, &z_coord);
                unstr_grid->getTypeList(&tl_out);
                set_out_objs[i] = unstr_grid;

                // read in the coordinates
                read_unstructured_coord(fFile, x_coord, y_coord, z_coord, num_coord);

                // init usg
                num_elem = 0;
                num_vert = 0;

                // read in the triangles
                for (i = 0; i < num_triang; i++)
                {
                    tl_out[num_elem] = TYPE_TRIANGLE;
                    el_out[num_elem] = num_vert;
                    num_elem++;

                    read_single_triangle(fFile, &(vl_out[num_vert]), &(vl_out[num_vert + 1]),
                                         &(vl_out[num_vert + 2]));
                    num_vert += 3;
                }

                // skip triangle-flags
                read_unstructured_triangle_flags(fFile, num_triang);

                // read in the tetrahedras
                for (i = 0; i < num_tetra; i++)
                {
                    tl_out[num_elem] = TYPE_TETRAHEDER;
                    el_out[num_elem] = num_vert;
                    num_elem++;

                    read_single_tetrahedra(fFile, &(vl_out[num_vert]), &(vl_out[num_vert + 1]),
                                           &(vl_out[num_vert + 2]), &(vl_out[num_vert + 3]));
                    num_vert += 4;
                }
            }

            // create set
            set_out = new coDoSet(name1, set_out_objs);

            // clean up
            for (i = 0; i < nzones; i++)
                delete set_out_objs[i];
            delete[] set_out_objs;

            delete[] zones_x_dim;
            delete[] zones_y_dim;
            delete[] zones_z_dim;

            // done
            return_object[0] = set_out;
        }
        else
        {
            Covise::sendError("ERROR: subtype not available");
            return (NULL);
        }
    }

    //////
    ////// SOLUTION_DATA
    //////

    if (datatype == _FILE_SOLUTION && read_flag == _READ_DATA)
    {

        if (subtype == _FILE_SINGLE_ZONE)
        {
            // get grid size
            read_size_header(fFile, &x_dim, &y_dim, &z_dim, &npoints);

            if (x_dim <= 0 || y_dim <= 0 || z_dim <= 0 || npoints <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }

            // show information
            Covise::sendInfo("IDIM: %d  JDIM: %d  KDIM: %d  NPOINTS: %d", x_dim, y_dim, z_dim, npoints);

            // get conditions
            read_solution_conditions(fFile, &mach, &alpha, &re, &time_var);

            // read values

            if (gridtype == _FILE_UNSTRUCTURED_GRID)
            {
                unstr_s3d_out = new coDoFloat(name1, npoints);
                unstr_s3d_out->getAddress(&(data_out[0]));
                return_object[0] = unstr_s3d_out;
            }
            else
            {
                unstr_s3d_out = new coDoFloat(name1, x_dim * y_dim * z_dim);
                unstr_s3d_out->getAddress(&(data_out[0]));
                return_object[0] = unstr_s3d_out;
            }
            read_solution_record(fFile, data_out[0], x_dim, y_dim, z_dim, npoints);

            if (gridtype == _FILE_UNSTRUCTURED_GRID)
            {
                unstr_v3d_out = new coDoVec3(name2, npoints);
                unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                return_object[1] = unstr_v3d_out;
            }
            else
            {
                unstr_v3d_out = new coDoVec3(name2, x_dim * y_dim * z_dim);
                unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                return_object[1] = unstr_v3d_out;
            }
            read_solution_record(fFile, data_out[0], x_dim, y_dim, z_dim, npoints);
            read_solution_record(fFile, data_out[1], x_dim, y_dim, z_dim, npoints);
            read_solution_record(fFile, data_out[2], x_dim, y_dim, z_dim, npoints);

            if (gridtype == _FILE_UNSTRUCTURED_GRID)
            {
                unstr_s3d_out = new coDoFloat(name3, npoints);
                unstr_s3d_out->getAddress(&(data_out[0]));
                return_object[2] = unstr_s3d_out;
            }
            else
            {
                unstr_s3d_out = new coDoFloat(name3, x_dim * y_dim * z_dim);
                unstr_s3d_out->getAddress(&(data_out[0]));
                return_object[2] = unstr_s3d_out;
            }
            read_solution_record(fFile, data_out[0], x_dim, y_dim, z_dim, npoints);

            // set attributes
            for (i = 0; i < 3; i++)
                set_solution_attributes(return_object[i], mach, alpha, re, time_var);

            // done
        }
        else if (subtype == _FILE_MULTI_ZONE)
        {
            // get nzones
            read_nzones(fFile, &nzones);

            // show information
            Covise::sendInfo("NZONES: %d", nzones);

            // init sets
            for (i = 0; i < 3; i++)
            {
                sol_set_out_objs[i] = new coDistributedObject *[nzones + 1];
                sol_set_out_objs[i][nzones] = NULL;
            }

            // get grid sizes
            zones_x_dim = new int[nzones];
            zones_y_dim = new int[nzones];
            zones_z_dim = new int[nzones];

            /*         for( i=0; i<nzones; i++ )
            {
                 read_size_header( fFile, &(zones_x_dim[i]), &(zones_y_dim[i]), \ 
                          &(zones_z_dim[i]) );
                 if( zones_x_dim[i]<=0 || zones_y_dim[i]<=0 || zones_z_dim[i]<=0 )
                  {
                  Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
             }
             }*/

            read_multi_header(fFile, &zones_x_dim, &zones_y_dim,
                              &zones_z_dim, nzones);

            // get conditions
            read_solution_conditions(fFile, &mach, &alpha, &re, &time_var);

            // work through the sets
            for (i = 0; i < nzones; i++)
            {
                // compute grid size
                x_dim = zones_x_dim[i];
                y_dim = zones_y_dim[i];
                z_dim = zones_z_dim[i];
                npoints = x_dim * y_dim * z_dim;

                // show information
                Covise::sendInfo("Zone: %d  IDIM: %d  JDIM: %d  KDIM: %d  NPOINTS: %d", i + 1, x_dim, y_dim, z_dim, npoints);

                // read values

                sprintf(bfr, "%s_%d", name1, i);
                if (gridtype == _FILE_UNSTRUCTURED_GRID)
                {
                    unstr_s3d_out = new coDoFloat(bfr, npoints);
                    unstr_s3d_out->getAddress(&(data_out[0]));
                    // add to set
                    sol_set_out_objs[0][i] = unstr_s3d_out;
                }
                else
                {
                    unstr_s3d_out = new coDoFloat(bfr, x_dim * y_dim * z_dim);
                    unstr_s3d_out->getAddress(&(data_out[0]));
                    // add to set
                    sol_set_out_objs[0][i] = unstr_s3d_out;
                }

                // lf_te: added 14.10.2003
                file_beginBlock(fFile);

                read_solution_record(fFile, data_out[0], x_dim, y_dim, z_dim, npoints);

                sprintf(bfr, "%s_%d", name2, i);
                if (gridtype == _FILE_UNSTRUCTURED_GRID)
                {
                    unstr_v3d_out = new coDoVec3(bfr, npoints);
                    unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                    // add to set
                    sol_set_out_objs[1][i] = unstr_v3d_out;
                }
                else
                {
                    unstr_v3d_out = new coDoVec3(bfr, x_dim * y_dim * z_dim);
                    unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                    // add to set
                    sol_set_out_objs[1][i] = unstr_v3d_out;
                }
                read_solution_record(fFile, data_out[0], x_dim, y_dim, z_dim, npoints);
                read_solution_record(fFile, data_out[1], x_dim, y_dim, z_dim, npoints);
                read_solution_record(fFile, data_out[2], x_dim, y_dim, z_dim, npoints);

                sprintf(bfr, "%s_%d", name3, i);
                if (gridtype == _FILE_UNSTRUCTURED_GRID)
                {
                    unstr_s3d_out = new coDoFloat(bfr, npoints);
                    unstr_s3d_out->getAddress(&(data_out[0]));
                    // add to set
                    sol_set_out_objs[2][i] = unstr_s3d_out;
                }
                else
                {
                    unstr_s3d_out = new coDoFloat(bfr, x_dim * y_dim * z_dim);
                    unstr_s3d_out->getAddress(&(data_out[0]));
                    // add to set
                    sol_set_out_objs[2][i] = unstr_s3d_out;
                }
                read_solution_record(fFile, data_out[0], x_dim, y_dim, z_dim, npoints);

                // lf_te: added 14.10.2003
                file_endBlock(fFile);

                // lf_te: added 14.10.2003, each zone may have its own conditions
                sfPos = ftell(fFile);
                file_beginBlock(fFile);
#ifdef __sgi
                fseek64(fFile, sfPos, SEEK_SET);
#else
                fseek(fFile, sfPos, SEEK_SET);
#endif
                if (blockSize == 16 || blockSize64 == 32 || blockSize64 == 16 || blockSize == 32)
                    read_solution_conditions(fFile, &mach, &alpha, &re, &time_var);

                // set attributes
                for (j = 0; j < 3; j++)
                    set_solution_attributes(sol_set_out_objs[j][i], mach, alpha, re, time_var);
            }

            // create sets
            set_out = new coDoSet(name1, sol_set_out_objs[0]);
            return_object[0] = set_out;
            set_out = new coDoSet(name2, sol_set_out_objs[1]);
            return_object[1] = set_out;
            set_out = new coDoSet(name3, sol_set_out_objs[2]);
            return_object[2] = set_out;

            // clean up
            for (i = 0; i < nzones; i++)
                for (j = 0; j < 3; j++)
                    delete sol_set_out_objs[j][i];
            for (i = 0; i < 3; i++)
            {
                delete[] sol_set_out_objs[i];
            }
            delete[] zones_x_dim;
            delete[] zones_y_dim;
            delete[] zones_z_dim;

            // done
        }
        else
        {
            Covise::sendError("ERROR: subtype not available");
            return (NULL);
        }
    }

    //////
    ////// (scalar or vector) DATA
    //////

    if (datatype == _FILE_DATA && read_flag == _READ_DATA)
    {
        if (subtype == _FILE_SINGLE_ZONE)
        {
            // read header
            read_data_header(fFile, &x_dim, &y_dim, &z_dim, &numComp, &npoints);

            if (x_dim <= 0 || y_dim <= 0 || z_dim <= 0 || npoints <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }

            // show information
            Covise::sendInfo("IDIM: %d  JDIM: %d  KDIM: %d  C: %d  NPOINTS: %d", x_dim, y_dim, z_dim, numComp, npoints);

            // create object and get pointers
            if (numComp == 3)
            {
                if (gridtype == _FILE_STRUCTURED_GRID || gridtype == _FILE_IBLANKED)
                {
                    return_object[0] = unstr_v3d_out = new coDoVec3(
                        name1, x_dim * y_dim * z_dim);
                    unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                }
                else
                {
                    return_object[0] = unstr_v3d_out = new coDoVec3(
                        name1, npoints);
                    unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                }
            }
            else if (numComp == 1)
            {
                if (gridtype == _FILE_STRUCTURED_GRID || gridtype == _FILE_IBLANKED)
                {
                    return_object[0] = unstr_s3d_out = new coDoFloat(
                        name1, x_dim * y_dim * z_dim);
                    unstr_s3d_out->getAddress(&(data_out[0]));
                }
                else
                {
                    return_object[0] = unstr_s3d_out = new coDoFloat(
                        name1, npoints);
                    unstr_s3d_out->getAddress(&(data_out[0]));
                }
            }
            else
            {
                Covise::sendError("There is something wrong with the data");
                return (NULL);
            }

            file_beginBlock(fFile);

            // read in the data
            for (i = 0; i < numComp; i++)
                read_data_record(fFile, data_out[i], x_dim, y_dim, z_dim, npoints);

            file_endBlock(fFile);
            // done
        }
        else if (subtype == _FILE_MULTI_ZONE)
        {
            // get nzones
            read_nzones(fFile, &nzones);

            if (nzones <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }

            // init set
            set_out_objs = new coDistributedObject *[nzones + 1];
            set_out_objs[nzones] = NULL;

            // get grid sizes
            zones_x_dim = new int[nzones];
            zones_y_dim = new int[nzones];
            zones_z_dim = new int[nzones];
            zones_comp = new int[nzones];

            for (i = 0; i < nzones; i++)
                read_data_header(fFile, &(zones_x_dim[i]), &(zones_y_dim[i]),
                                 &(zones_z_dim[i]), &(zones_comp[i]));

            // show information
            Covise::sendInfo("NZONES: %d  C: %d", nzones, numComp);

            // check for straight data-type
            numComp = zones_comp[0];
            for (i = 0; i < nzones && zones_comp[i] == numComp; i++)
                ;
            if (i != nzones)
            {
                Covise::sendError("ERROR: multi-zone mixed scalar and vector data not supported !!!");
                return (NULL);
            }

            // work through the set
            for (i = 0; i < nzones; i++)
            {
                // compute grid size
                x_dim = zones_x_dim[i];
                y_dim = zones_y_dim[i];
                z_dim = zones_z_dim[i];
                npoints = x_dim * y_dim * z_dim;

                // show information
                Covise::sendInfo("Zone: %d  IDIM: %d  JDIM: %d  KDIM: %d  NPOINTS: %d", i, x_dim, y_dim, z_dim, npoints);

                // compute object name and create it
                sprintf(bfr, "%s_%d", name1, i);
                if (numComp == 3)
                {
                    if (gridtype == _FILE_STRUCTURED_GRID || gridtype == _FILE_IBLANKED)
                    {
                        return_object[0] = unstr_v3d_out = new coDoVec3(
                            bfr, x_dim * y_dim * z_dim);
                        unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                    }
                    else
                    {
                        return_object[0] = unstr_v3d_out = new coDoVec3(
                            bfr, npoints);
                        unstr_v3d_out->getAddresses(&(data_out[0]), &(data_out[1]), &(data_out[2]));
                    }
                }
                else if (numComp == 1)
                {
                    if (gridtype == _FILE_STRUCTURED_GRID || gridtype == _FILE_IBLANKED)
                    {
                        return_object[0] = unstr_s3d_out = new coDoFloat(
                            bfr, x_dim * y_dim * z_dim);
                        unstr_s3d_out->getAddress(&(data_out[0]));
                    }
                    else
                    {
                        return_object[0] = unstr_s3d_out = new coDoFloat(
                            bfr, npoints);
                        unstr_s3d_out->getAddress(&(data_out[0]));
                    }
                }
                else
                {
                    Covise::sendError("ERROR: only scalar or vector data supported");
                    return (NULL);
                }

                file_beginBlock(fFile);
                // read in the data
                for (k = 0; k < numComp; k++)
                    read_data_record(fFile, data_out[k], x_dim, y_dim, z_dim, npoints);

                file_endBlock(fFile);
                // add element to set
                set_out_objs[i] = return_object[0];
            }

            // create set
            set_out = new coDoSet(name1, set_out_objs);

            // clean up
            for (i = 0; i < nzones; i++)
                delete set_out_objs[i];
            delete[] set_out_objs;

            delete[] zones_x_dim;
            delete[] zones_y_dim;
            delete[] zones_z_dim;
            delete[] zones_comp;

            // done
            return_object[0] = set_out;
        }
        else
        {
            Covise::sendError("ERROR: subtype not available");
            return (NULL);
        }
    }

    //////
    ////// function DATA
    //////

    if (datatype == _FILE_FUNCTION && read_flag == _READ_DATA)
    {
        if (subtype == _FILE_SINGLE_ZONE)
            Covise::sendError("Reading of function data in single-zone file not yet implemented");
        else
        {
            // get nzones
            read_nzones(fFile, &nzones);

            if (nzones <= 0)
            {
                Covise::sendError("File contains no valid information. Please check all parameters that describe the file content.");
                return NULL;
            }

            // init set
            set_out_objs = new coDistributedObject *[nzones + 1];
            set_out_objs[nzones] = NULL;

            // get grid sizes
            zones_x_dim = new int[nzones];
            zones_y_dim = new int[nzones];
            zones_z_dim = new int[nzones];
            zones_comp = new int[nzones];

            read_multi_data_header(fFile, &zones_x_dim, &zones_y_dim,
                                   &zones_z_dim, &zones_comp, nzones);

            // check for straight data-type
            numComp = zones_comp[0];
            for (i = 0; i < nzones && zones_comp[i] == numComp; i++)
                ;
            if (i != nzones)
            {
                Covise::sendError("ERROR: multi-zone mixed (different number of fields for function data) not supported !!!");
                return (NULL);
            }

            // show information
            Covise::sendInfo("NZONES: %d  Components: %d", nzones, numComp);

            if (numComp > 3)
            {
                Covise::sendError("ERROR: more than three fileds for data not supported !!!");
                return (NULL);
            }

            // get object names
            const char *out_obj_names[3];
            out_obj_names[0] = name1;
            out_obj_names[1] = name2;
            out_obj_names[2] = name3;

            for (i = 0; i < numComp; i++)
            {
                sol_set_out_objs[i] = new coDistributedObject *[nzones + 1];
                sol_set_out_objs[i][nzones] = NULL;
            }

            // work through the set
            for (i = 0; i < nzones; i++)
            {
                // compute grid size
                x_dim = zones_x_dim[i];
                y_dim = zones_y_dim[i];
                z_dim = zones_z_dim[i];
                npoints = x_dim * y_dim * z_dim;

                // show information
                Covise::sendInfo("Data zone: %d  IDIM: %d  JDIM: %d  KDIM: %d  NPOINTS: %d", i + 1, x_dim, y_dim, z_dim, npoints);

                //read
                // read in the data
                file_beginBlock(fFile);
                for (k = 0; k < numComp; k++)
                {
                    float *data_array;
                    sprintf(bfr, "%s_%d", out_obj_names[k], i);
                    coDoFloat *s_data = new coDoFloat(bfr, x_dim * y_dim * z_dim);
                    if (!s_data->objectOk())
                        Covise::sendError("Failed to create the structure grid");

                    s_data->getAddress(&data_array);

                    read_data_record(fFile, data_array, x_dim, y_dim, z_dim, npoints);
                    sol_set_out_objs[k][i] = s_data;
                }
                file_endBlock(fFile);
            }

            // create sets
            for (k = 0; k < numComp; k++)
            {
                set_out = new coDoSet(out_obj_names[k], sol_set_out_objs[k]);
                return_object[k] = set_out;
            }

            // clean up
            for (i = 0; i < nzones; i++)
                for (k = 0; k < numComp; k++)
                    delete sol_set_out_objs[k][i];
            for (i = 0; i < numComp; i++)
                delete[] sol_set_out_objs[i];
        }
    }
    // done
    return (return_object);
}

//////
////// low-level read function that takes care of the file_type setting
//////

/*void Application::file_read( FILE *f, int *p, int n )
{
   switch( filetype )
   {
      case _FILE_TYPE_BINARY:
      case _FILE_TYPE_FORTRAN:
         fread(p, sizeof(int),n,f);
         break;
      case _FILE_TYPE_ASCII:
         {
         char bfr[1024];
int i;
for( i=0; i<n; i++ )
{
file_read_ascii( f, bfr );
sscanf( bfr, "%d", &p[i] );
}
}
break;
default:
Covise::sendError("ERROR: selected filetype not yet implemented");
break;
}
}*/

void Application::file_beginBlock(FILE *f)
{
    if (filetype == _FILE_TYPE_FORTRAN)
    {
        file_read(f, &blockSize, 1);
    }
    else if (filetype == _FILE_TYPE_FORTRAN64)
    {
        file_read(f, &blockSize64, 1);
    }
}

int Application::file_endBlock(FILE *f)
{
    if (filetype == _FILE_TYPE_FORTRAN)
    {
        int bs = blockSize;
        file_read(f, &bs, 1);
        if (bs != blockSize && !feof(f)) // NOTE: final blockmarker may be replaced by eof
        {
            Covise::sendError("ERROR wrong FORTRAN block marker");
            return -1;
        }
    }
    else if (filetype == _FILE_TYPE_FORTRAN64)
    {
        coInt64 bs = blockSize64;
        file_read(f, &bs, 1);
        if (bs != blockSize64 && !feof(f)) // NOTE: final blockmarker may be replaced by eof
        {
            Covise::sendError("ERROR wrong FORTRAN block marker");
            return -1;
        }
    }
    return 0;
}

void Application::file_read(FILE *f, int *p, int n)
{
    int i;
    switch (filetype)
    {
    case _FILE_TYPE_BINARY:
    case _FILE_TYPE_FORTRAN:
    case _FILE_TYPE_FORTRAN64:
        if (fread(p, sizeof(int), n, f) != n)
        {
            cerr << "ReadPlot3D::file_read: fread failed" << endl;
        }
        if (byteswap_flag)
        {
            for (i = 0; i < n; i++)
                byteswap(p + i, sizeof(int));
        }
        break;
    case _FILE_TYPE_ASCII:
    {
        int i;
        char *bfr;
        for (i = 0; i < n; i++)
        {
            if (!feof(f))
            {
                if (new_buffer)
                {
                    if (fgets(buffer, 1024, f) == NULL)
                    {
                        cerr << "ReadPlot3D::file_read: fgets1 failed" << endl;
                    }
                    bfr = strtok(buffer, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets2 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%d", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf1 failed" << endl;
                        }
                    }
                    new_buffer = 0;
                }
                else
                {
                    bfr = strtok(NULL, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets3 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%d", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf2 failed" << endl;
                        }
                    }
                }
            }
        }
    }
    break;
    default:
        Covise::sendError("ERROR: selected filetype not yet implemented");
        break;
    }
}

void Application::file_read(FILE *f, coInt64 *p, int n)
{
    int i;
    switch (filetype)
    {
    case _FILE_TYPE_BINARY:
    case _FILE_TYPE_FORTRAN:
    case _FILE_TYPE_FORTRAN64:
        if (fread(p, sizeof(coInt64), n, f) != n)
        {
            cerr << "ReadPlot3D::file_read: fread failed" << endl;
        }
        if (byteswap_flag)
        {
            for (i = 0; i < n; i++)
                byteswap(p + i, sizeof(coInt64));
        }
        break;
    case _FILE_TYPE_ASCII:
    {
        int i;
        char *bfr;
        for (i = 0; i < n; i++)
        {
            if (!feof(f))
            {
                if (new_buffer)
                {
                    if (fgets(buffer, 1024, f) == NULL)
                    {
                        cerr << "ReadPlot3D::file_read: fgets1 failed" << endl;
                    }
                    bfr = strtok(buffer, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets2 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%lld", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf1 failed" << endl;
                        }
                    }
                    new_buffer = 0;
                }
                else
                {
                    bfr = strtok(NULL, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets3 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%lld", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf2 failed" << endl;
                        }
                    }
                }
            }
        }
    }
    break;
    default:
        Covise::sendError("ERROR: selected filetype not yet implemented");
        break;
    }
}

void Application::file_read(FILE *f, float *p, int n)
{
    int i;
    switch (filetype)
    {
    case _FILE_TYPE_BINARY:
    case _FILE_TYPE_FORTRAN:
    case _FILE_TYPE_FORTRAN64:
        if (fread(p, sizeof(float), n, f) != n)
        {
            cerr << "ReadPlot3D::file_read: fread2 failed" << endl;
        }
        if (byteswap_flag)
        {
            for (i = 0; i < n; i++)
                byteswap(p + i, sizeof(float));
        }
        break;
    case _FILE_TYPE_ASCII:
    {
        int i;
        char *bfr;
        for (i = 0; i < n; i++)
        {
            if (!feof(f))
            {
                if (new_buffer)
                {
                    if (fgets(buffer, 1024, f) == NULL)
                    {
                        cerr << "ReadPlot3D::file_read: fgets4 failed" << endl;
                    }
                    bfr = strtok(buffer, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets5 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%f", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf3 failed" << endl;
                        }
                    }
                    new_buffer = 0;
                }
                else
                {
                    bfr = strtok(NULL, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets6 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%f", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf4 failed" << endl;
                        }
                    }
                }
            }
        }
    }
    break;
    default:
        Covise::sendError("ERROR: selected filetype not yet implemented");
        break;
    }
}

void Application::file_read(FILE *f, double *p, int n)
{
    int i;
    switch (filetype)
    {
    case _FILE_TYPE_BINARY:
    case _FILE_TYPE_FORTRAN:
    case _FILE_TYPE_FORTRAN64:
        if (fread(p, sizeof(double), n, f) != n)
        {
            cerr << "ReadPlot3D::file_read: fread2 failed" << endl;
        }
        if (byteswap_flag)
        {
            for (i = 0; i < n; i++)
                byteswap(p + i, sizeof(double));
        }
        break;
    case _FILE_TYPE_ASCII:
    {
        int i;
        char *bfr;
        for (i = 0; i < n; i++)
        {
            if (!feof(f))
            {
                if (new_buffer)
                {
                    if (fgets(buffer, 1024, f) == NULL)
                    {
                        cerr << "ReadPlot3D::file_read: fgets4 failed" << endl;
                    }
                    bfr = strtok(buffer, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets5 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%lf", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf3 failed" << endl;
                        }
                    }
                    new_buffer = 0;
                }
                else
                {
                    bfr = strtok(NULL, " \n");
                    while (bfr == NULL && !feof(f))
                    {
                        if (fgets(buffer, 1024, f) == NULL)
                        {
                            cerr << "ReadPlot3D::file_read: fgets6 failed" << endl;
                        }
                        bfr = strtok(buffer, " \n");
                    }
                    if (bfr != NULL)
                    {
                        if (sscanf(bfr, "%lf", &p[i]) != 1)
                        {
                            cerr << "ReadPlot3D::file_read: sscanf4 failed" << endl;
                        }
                    }
                }
            }
        }
    }
    break;
    default:
        Covise::sendError("ERROR: selected filetype not yet implemented");
        break;
    }
}

void Application::file_read_ascii(FILE *f, char *p)
{
    int i;
    i = 0;
    p[0] = fgetc(f);
    while (p[0] == ' ' || p[0] == '\n')
        p[0] = fgetc(f);
    i++;
    p[i] = fgetc(f);
    while (p[i] != ' ' && p[i] != '\n')
    {
        i++;
        p[i] = fgetc(f);
    }
    p[i] = '\0';
    return;
}

void Application::byteswap(void *p, int n)
{
    if (!byteswap_flag)
        return;
    char *t = new char[n];
    char *pt = (char *)p;
    int i;
    memcpy(t, pt, n);
    for (i = 0; i < n; i++)
        pt[i] = t[n - 1 - i];
    delete[] t;
}

//////
////// sub-level read functions as interface to damn Plot3D-format
//////

void Application::read_size_header(FILE *f, int *x, int *y, int *z, int *n)
{
    file_beginBlock(f);
    file_read(f, x, 1);
    file_read(f, y, 1);
    file_read(f, z, 1);
    file_endBlock(f);

    if (n)
        (*n) = (*x) * (*y) * (*z);

    return;
}

void Application::read_multi_header(FILE *f, int **x, int **y, int **z, int nblocks)
{

    file_beginBlock(f);
    for (int i = 0; i < nblocks; i++)
    {
        file_read(f, &(*x)[i], 1);
        file_read(f, &(*y)[i], 1);
        file_read(f, &(*z)[i], 1);
    }
    file_endBlock(f);
}

void Application::read_structured_grid_record(FILE *f,
                                              float *x, int x_dim,
                                              float *y, int y_dim,
                                              float *z, int z_dim,
                                              int n, int *iblank,
                                              int gridtype)
{
    float *x_coord, *y_coord, *z_coord;
    int u, v, w;
    int j, k;

    // alloc
    x_coord = new float[n];
    y_coord = new float[n];
    z_coord = new float[n];

    // do the work
    file_beginBlock(f);

    file_read(f, x_coord, n);
    file_read(f, y_coord, n);
    file_read(f, z_coord, n);

    // rewrite the data in order to be covise-confirm
    for (u = 0; u < x_dim; u++)
        for (v = 0; v < y_dim; v++)
            for (w = 0; w < z_dim; w++)
            {
                j = u * y_dim * z_dim + v * z_dim + w;
                k = u + v * x_dim + w * x_dim * y_dim;

                x[j] = x_coord[k];
                y[j] = y_coord[k];
                z[j] = z_coord[k];
            }

    // skip ibanked data
    if (gridtype == _FILE_IBLANKED)
    {
        file_read(f, x_coord, n);
        // this is dirty: we read an integer field, but use a float buffer
        int *iPtr = (int *)x_coord;
        for (u = 0; u < x_dim; u++)
            for (v = 0; v < y_dim; v++)
                for (w = 0; w < z_dim; w++)
                {
                    j = u * y_dim * z_dim + v * z_dim + w;
                    k = u + v * x_dim + w * x_dim * y_dim;

                    iblank[j] = iPtr[k];
                }
    }

    file_endBlock(f);

    // clean up
    delete[] x_coord;
    delete[] y_coord;
    delete[] z_coord;

    // done
    return;
}

void Application::read_solution_conditions(FILE *f, float *mach, float *alpha, float *re, float *time)
{
    float t;

    if (mach)
    {
        file_beginBlock(f);
        if ((filetype == _FILE_TYPE_FORTRAN64 && blockSize64 == 32) || (filetype == _FILE_TYPE_FORTRAN && blockSize == 32))
        { // read double values
            double dm, da, dr, dt;
            file_read(f, &dm, 1);
            file_read(f, &da, 1);
            file_read(f, &dr, 1);
            file_read(f, &dt, 1);
            *mach = (float)dm;
            *alpha = (float)da;
            *re = (float)dr;
            *time = (float)dt;
        }
        else
        {
            file_read(f, mach, 1);
            file_read(f, alpha, 1);
            file_read(f, re, 1);
            file_read(f, time, 1);
        }
        file_endBlock(f);
    }
    else
    {
        file_beginBlock(f);
        file_read(f, &t, 1);
        file_read(f, &t, 1);
        file_read(f, &t, 1);
        file_read(f, &t, 1);
        file_endBlock(f);
    }

    return;
}

void Application::read_solution_record(FILE *f, float *val, int x_dim, int y_dim, int z_dim, int n)
{
    float *t;
    int u, v, w;
    int j, k;

    // alloc
    t = new float[n];

    // do the work
    file_read(f, t, n);

    // rewrite the data in order to be covise-confirm
    for (u = 0; u < x_dim; u++)
        for (v = 0; v < y_dim; v++)
            for (w = 0; w < z_dim; w++)
            {
                j = u * y_dim * z_dim + v * z_dim + w;
                k = u + v * x_dim + w * x_dim * y_dim;

                val[j] = t[k];
            }

    // clean up
    delete[] t;

    // done
    return;
}

void Application::read_data_header(FILE *f, int *x, int *y, int *z, int *c, int *n)
{
    file_beginBlock(f);
    file_read(f, x, 1);
    file_read(f, y, 1);
    file_read(f, z, 1);
    file_read(f, c, 1);
    file_endBlock(f);

    if (n)
        (*n) = (*x) * (*y) * (*z);

    return;
}

void Application::read_multi_data_header(FILE *f, int **x, int **y, int **z, int **c, int nblocks)
{
    file_beginBlock(f);
    for (int i = 0; i < nblocks; i++)
    {
        file_read(f, &(*x)[i], 1);
        file_read(f, &(*y)[i], 1);
        file_read(f, &(*z)[i], 1);
        file_read(f, &(*c)[i], 1);
    }
    file_endBlock(f);

    return;
}

void Application::read_data_record(FILE *f, float *val, int x_dim, int y_dim, int z_dim, int n)
{
    read_solution_record(f, val, x_dim, y_dim, z_dim, n);
    return;
}

void Application::read_iblanked(FILE *f, int n, int *p)
{
    int i;
    int t;

    file_beginBlock(f);
    if (p)
        file_read(f, p, n);
    else
        for (i = 0; i < n; i++)
            file_read(f, &t, 1);
    file_endBlock(f);

    return;
}

void Application::read_nzones(FILE *f, int *n)
{
    file_beginBlock(f);
    file_read(f, n, 1);
    if (*n > 100000 || *n < 0)
    {
        byteswap_flag = 1;
        byteswap(n, sizeof(int));
    }
    file_endBlock(f);
    return;
}

void Application::read_single_triangle(FILE *f, int *v1, int *v2, int *v3)
{
    file_beginBlock(f);
    file_read(f, v1, 1);
    file_read(f, v2, 1);
    file_read(f, v3, 1);
    file_endBlock(f);

    // damn fortran starts counting at 1 while we do at 1
    (*v1)--;
    (*v2)--;
    (*v3)--;

    return;
}

void Application::read_single_tetrahedra(FILE *f, int *v1, int *v2, int *v3, int *v4)
{
    file_beginBlock(f);
    file_read(f, v1, 1);
    file_read(f, v2, 1);
    file_read(f, v3, 1);
    file_read(f, v4, 1);
    file_endBlock(f);

    // damn fortran starts counting at 1 while we do at 1
    (*v1)--;
    (*v2)--;
    (*v3)--;
    (*v4)--;

    return;
}

void Application::read_unstructured_header(FILE *f, int *points, int *triang, int *tetra)
{
    file_beginBlock(f);
    file_read(f, points, 1);
    file_read(f, triang, 1);
    file_read(f, tetra, 1);
    file_endBlock(f);
    return;
}

void Application::read_unstructured_coord(FILE *f, float *x, float *y, float *z, int n)
{
    file_beginBlock(f);
    file_read(f, x, n);
    file_read(f, y, n);
    file_read(f, z, n);
    file_endBlock(f);
    return;
}

void Application::read_unstructured_triangle_flags(FILE *f, int n, int *flags)
{
    int i, t;

    file_beginBlock(f);
    if (flags)
        file_read(f, flags, n);
    else
        for (i = 0; i < n; i++)
            file_read(f, &t, 1);
    file_endBlock(f);

    return;
}

void Application::set_solution_attributes(coDistributedObject *r, float mach, float alpha, float re, float time)
{
    char bfr[100];

    // set attributes
    sprintf(bfr, "%f", mach);
    r->addAttribute("MACH", bfr);

    sprintf(bfr, "%f", alpha);
    r->addAttribute("ALPHA", bfr);

    sprintf(bfr, "%f", re);
    r->addAttribute("RE", bfr);

    sprintf(bfr, "%f", time);
    r->addAttribute("TIME", bfr);

    // done
    return;
}
