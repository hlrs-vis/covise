/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2011 HLRS **
 **                                                                        **
 ** Description: Read module for PLUTO data        	                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Authors:                                                                **
 **                                                                        **
 **                    Steffen Brinkmann, Uwe Woessner                     **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  22.11.11  V1.0                                                  **
\**************************************************************************/

#include "ReadPLUTO.h"
#include "string"

ReadPLUTO::ReadPLUTO(int argc, char **argv)
    : coSimpleModule(argc, argv,
                     "Read PLUTO")
{
    // ports
    p_mesh = addOutputPort("mesh", "StructuredGrid", "structured grid");
    p_rho = addOutputPort("rho", "Float", "density");
    p_rholog = addOutputPort("rholog", "Float", "logarithmic density");
    p_pressure = addOutputPort("pressure", "Float", "pressure");
    p_pressurelog = addOutputPort("pressurelog", "Float", "logarithmic pressure");
    p_velocity = addOutputPort("velocity", "Vec3", "velocity");
    p_magfield = addOutputPort("magfield", "Vec3", "magnetic field");
    p_velocity_cart = addOutputPort("velocity_cart", "Vec3", "velocity in cartesian coordinates");
    p_magfield_cart = addOutputPort("magfield_cart", "Vec3", "magnetic field in cartesian coordinates");

    // choice arrays
    const char *precisionChoice[] = { "single", "double" };
    const char *fileFormatChoice[] = { "single", "multiple" };

    // parameter
    p_path = addFileBrowserParam("path", "Data file path");
    p_path->setValue("$PLUTO_DIR/Torus_3D/", "grid.out");

    p_precision = addChoiceParam("format", "single or double precision");
    p_precision->setValue(2, precisionChoice, 0);

    p_file_format = addChoiceParam("single", "single file or multiple files");
    p_file_format->setValue(2, fileFormatChoice, 1);

    p_tbeg = addInt32Param("t_beg", "First timestep to read");
    p_tbeg->setValue(1);

    p_tend = addInt32Param("t_end", "Last timestep to read");
    p_tend->setValue(1);

    p_skip = addInt32Param("skip", "Number of timesteps to skip");
    p_skip->setValue(0);

    p_axisymm = addBooleanParam("axisymm", "Is the data spherical 2D axisammetric?");
    p_axisymm->setValue(0); // 0 == False

    p_n_axisymm = addInt32Param("n_axisymm", "Expand phi-coordinate with n cells");
    p_skip->setValue(20);

    mesh = NULL;
}

// read data from multiple files, i.e. one file per variable
int ReadPLUTO::readData(int fd, float *data) //returns <0 on Error
{
    int i, j, k;
    float number;
    if (axisymm == 0)
    {
        for (k = 0; k < n_x3; k++)
        {
            for (j = 0; j < n_x2; j++)
            {
                for (i = 0; i < n_x1; i++)
                {
                    //infile.read(buffer,sizeof(float));
                    //data[i] = atof(buffer));
                    //readFloat(fd,data[i*n_x2*n_x3 + j*n_x3 + k]);
                    if (k == n_x3 - 1)
                    {
                        data[i * n_x2 * n_x3 + j * n_x3 + k] = data[i * n_x2 * n_x3 + j * n_x3 + 0];
                    }
                    else
                    {
                        read(fd, &number, sizeof(float));
                        data[i * n_x2 * n_x3 + j * n_x3 + k] = number;
                    }
                }
            }
        }
    }
    else
    {
        float number = 0.;
        for (j = 0; j < n_x2; j++)
        {
            for (i = 0; i < n_x1; i++)
            {
                read(fd, &number, sizeof(float));
                for (k = 0; k < n_x3; k++)
                {
                    data[i * n_x2 * n_x3 + j * n_x3 + k] = number;
                }
            }
        }
    }
    return (0);
}

// read data from single file, i.e. all variables in one file
int ReadPLUTO::readData(int fd, float *rho, float *pr,
                        float *v1, float *v2, float *v3,
                        float *b1, float *b2, float *b3) //returns <0 on Error
{
    if (readData(fd, rho) < 0)
        return -1;
    if (readData(fd, v1) < 0)
        return -1;
    if (readData(fd, v2) < 0)
        return -1;
    if (readData(fd, v3) < 0)
        return -1;
    if (readData(fd, pr) < 0)
        return -1;
    if (readData(fd, b1) < 0)
        return -1;
    if (readData(fd, b2) < 0)
        return -1;
    if (readData(fd, b3) < 0)
        return -1;
    return (0);
}

int ReadPLUTO::openDataFile(int *fd, const char *dataPath)
{
    cout << "\nReadPLUTO: " << dataPath << endl;

#ifdef _WIN32
    if ((*fd = Covise::open(dataPath, _O_RDONLY | _O_BINARY)) < 0)
#else
    if ((*fd = Covise::open(dataPath, O_RDONLY)) < 0)
#endif
    {
        sendError("ERROR: Can't open file >> %s", dataPath);
    }
    return *fd;
}

int ReadPLUTO::compute(const char *)
{
    int fd, ret, n, i, j, k, t, ii;
    int tbeg, tend, skip;
    char buf[500], filename_buf[500];

    // read input parameters

    dataPath = p_path->getValue();
    dir_path = dataPath;
    dir_path.erase(dir_path.find_last_of('/'));
    tbeg = p_tbeg->getValue();
    tend = p_tend->getValue();
    skip = p_skip->getValue();
    readDouble = p_precision->getValue();
    if (readDouble == 1)
    {
        sendError("double precision not implemented yet");
    }
    fileFormat = p_file_format->getValue();

    axisymm = p_axisymm->getValue();
    n_axisymm = p_n_axisymm->getValue();

    // declare distributed object pointers

    coDistributedObject **grids;
    coDistributedObject **DOSrho;
    coDistributedObject **DOSrholog;
    coDistributedObject **DOSpress;
    coDistributedObject **DOSpresslog;
    coDistributedObject **DOSvel;
    coDistributedObject **DOSmagfield;
    coDistributedObject **DOSvel_cart;
    coDistributedObject **DOSmagfield_cart;

    grids = new coDistributedObject *[tend - tbeg + 2];
    DOSrho = new coDistributedObject *[tend - tbeg + 2];
    DOSrholog = new coDistributedObject *[tend - tbeg + 2];
    DOSpress = new coDistributedObject *[tend - tbeg + 2];
    DOSpresslog = new coDistributedObject *[tend - tbeg + 2];
    DOSvel = new coDistributedObject *[tend - tbeg + 2];
    DOSmagfield = new coDistributedObject *[tend - tbeg + 2];
    DOSvel_cart = new coDistributedObject *[tend - tbeg + 2];
    DOSmagfield_cart = new coDistributedObject *[tend - tbeg + 2];

    for (i = 0; i < tend - tbeg + 2; i++)
    {
        grids[i] = NULL;
        DOSrho[i] = NULL;
        DOSrholog[i] = NULL;
        DOSpress[i] = NULL;
        DOSpresslog[i] = NULL;
        DOSvel[i] = NULL;
        DOSmagfield[i] = NULL;
        DOSvel_cart[i] = NULL;
        DOSmagfield_cart[i] = NULL;
    }

    mesh_name = p_mesh->getObjName();
    rho_name = p_rho->getObjName();
    rholog_name = p_rholog->getObjName();
    pr_name = p_pressure->getObjName();
    prlog_name = p_pressurelog->getObjName();
    vel_name = p_velocity->getObjName();
    magfield_name = p_magfield->getObjName();
    vel_cart_name = p_velocity_cart->getObjName();
    magfield_cart_name = p_magfield_cart->getObjName();

    // read grid from grid.out
    {

        ifstream is_grid(dataPath, ifstream::in);
        float dummy, number;
        is_grid >> n_x1;
        cout << "n_x1 " << n_x1 << endl;
        vec_gridx1 = new float[n_x1];

        for (i = 0; i < n_x1; i++)
        {
            is_grid >> dummy; // counter
            is_grid >> dummy; // left border
            is_grid >> number; // centre of cell
            is_grid >> dummy; // right border
            is_grid >> dummy; // size of cell
            vec_gridx1[i] = number;
        }

        // x2 ccordinates
        is_grid >> n_x2;
        cout << "n_x2 " << n_x2 << endl;
        vec_gridx2 = new float[n_x2];
        for (j = 0; j < n_x2; j++)
        {
            is_grid >> dummy; // counter
            is_grid >> dummy; // left border
            is_grid >> number; // centre of cell
            is_grid >> dummy; // right border
            is_grid >> dummy; // size of cell
            vec_gridx2[j] = number;
        }

        // x3 ccordinates
        is_grid >> n_x3;
        cout << "n_x3 " << n_x3 << endl;
        n_x3 += 1; // for periodic grid

        if (axisymm == 1)
        {
            cout << "Expand phi direction with " << n_axisymm << " cells." << endl;
            n_x3 = n_axisymm;
        }

        vec_gridx3 = new float[n_x3];

        if (axisymm == 0)
        {
            for (k = 0; k < n_x3 - 1; k++)
            {
                is_grid >> dummy; // counter
                is_grid >> dummy; // left border
                is_grid >> number; // centre of cell
                is_grid >> dummy; // right border
                is_grid >> dummy; // size of cell
                vec_gridx3[k] = number;
            }
        }
        else
        {
            for (number = 0, k = 0; k < n_x3 - 1; number += 6.2831854 / n_axisymm, k++)
            {
                vec_gridx3[k] = number;
            }
        }
        // close the grid
        vec_gridx3[n_x3 - 1] = vec_gridx3[0];

        is_grid.close();

        // build 1D array for grid
        vec_gridx1_glob = new float[n_x1 * n_x2 * n_x3];
        vec_gridx2_glob = new float[n_x1 * n_x2 * n_x3];
        vec_gridx3_glob = new float[n_x1 * n_x2 * n_x3];

        for (k = 0; k < n_x3 - 1; k++)
        {
            for (j = 0; j < n_x2; j++)
            {
                for (i = 0; i < n_x1; i++)
                {
                    vec_gridx1_glob[i * n_x2 * n_x3 + j * n_x3 + k] = vec_gridx1[i] * sin(vec_gridx2[j]) * cos(vec_gridx3[k]);
                    vec_gridx2_glob[i * n_x2 * n_x3 + j * n_x3 + k] = vec_gridx1[i] * sin(vec_gridx2[j]) * sin(vec_gridx3[k]);
                    vec_gridx3_glob[i * n_x2 * n_x3 + j * n_x3 + k] = vec_gridx1[i] * cos(vec_gridx2[j]);
                }
            }
        }
        for (j = 0; j < n_x2; j++)
        {
            for (i = 0; i < n_x1; i++)
            {
                vec_gridx1_glob[i * n_x2 * n_x3 + j * n_x3 + n_x3 - 1] = vec_gridx1[i] * sin(vec_gridx2[j]) * cos(vec_gridx3[0]);
                vec_gridx2_glob[i * n_x2 * n_x3 + j * n_x3 + n_x3 - 1] = vec_gridx1[i] * sin(vec_gridx2[j]) * sin(vec_gridx3[0]);
                vec_gridx3_glob[i * n_x2 * n_x3 + j * n_x3 + n_x3 - 1] = vec_gridx1[i] * cos(vec_gridx2[j]);
            }
        }
    }

    for (n = 0, t = tbeg; t <= tend; n++, t += skip + 1)
    {
        // initialise data objects
        {
            sprintf(buf, "%s_%d", mesh_name, n);
            mesh = new coDoStructuredGrid(buf, n_x1, n_x2, n_x3,
                                          vec_gridx1_glob,
                                          vec_gridx2_glob,
                                          vec_gridx3_glob);
            if (!mesh->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", rho_name, n);
            DOrho = new coDoFloat(buf, n_x1 * n_x2 * n_x3);
            if (!DOrho->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", rholog_name, n);
            DOrholog = new coDoFloat(buf, n_x1 * n_x2 * n_x3);
            if (!DOrholog->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", pr_name, n);
            DOpress = new coDoFloat(buf, n_x1 * n_x2 * n_x3);
            if (!DOpress->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", prlog_name, n);
            DOpresslog = new coDoFloat(buf, n_x1 * n_x2 * n_x3);
            if (!DOpresslog->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", vel_name, n);
            DOvel = new coDoVec3(buf, n_x1 * n_x2 * n_x3);
            if (!DOvel->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", magfield_name, n);
            DOmagfield = new coDoVec3(buf, n_x1 * n_x2 * n_x3);
            if (!DOmagfield->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", vel_cart_name, n);
            DOvel_cart = new coDoVec3(buf, n_x1 * n_x2 * n_x3);
            if (!DOvel_cart->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", magfield_cart_name, n);
            DOmagfield_cart = new coDoVec3(buf, n_x1 * n_x2 * n_x3);
            if (!DOmagfield_cart->objectOk())
            {
                sendError("could not create output object:");
                break;
            }
        }

        // get data array addresses

        DOrho->getAddress(&rho);
        DOrholog->getAddress(&rholog);
        DOpress->getAddress(&pr);
        DOpresslog->getAddress(&prlog);
        DOvel->getAddresses(&v1, &v2, &v3);
        DOmagfield->getAddresses(&b1, &b2, &b3);
        DOvel_cart->getAddresses(&vx, &vy, &vz);
        DOmagfield_cart->getAddresses(&bx, &by, &bz);

        // read data from files

        if (fileFormat == 0)
        { // single file (data.%d.flt)

            sprintf(filename_buf, "%s/data.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, rho, pr, v1, v2, v3, b1, b2, b3) < 0)
                break;
            close(fd);
        }
        else
        { // multiple files (rho.%d.flt etc.)

            sprintf(filename_buf, "%s/rho.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, rho) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/pr.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, pr) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/v1.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, v1) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/v2.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, v2) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/v3.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, v3) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/b1.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, b1) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/b2.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, b2) < 0)
                break;
            close(fd);

            sprintf(filename_buf, "%s/b3.%04d.flt", dir_path.c_str(), t);
            fd = openDataFile(&fd, filename_buf);
            if (readData(fd, b3) < 0)
                break;
            close(fd);
        }

        // generate cartesian vector data arrays
        // and logarithmic arrays
        for (k = 0; k < n_x3; k++)
        {
            for (j = 0; j < n_x2; j++)
            {
                for (i = 0; i < n_x1; i++)
                {
                    ii = i * n_x2 * n_x3 + j * n_x3 + k;
                    bx[ii] = b1[ii] * cos(vec_gridx3[k]) * sin(vec_gridx2[j]) + b2[ii] * cos(vec_gridx3[k]) * cos(vec_gridx2[j]) - b3[ii] * sin(vec_gridx3[k]);
                    by[ii] = b1[ii] * sin(vec_gridx3[k]) * sin(vec_gridx2[j]) + b2[ii] * sin(vec_gridx3[k]) * cos(vec_gridx2[j]) + b3[ii] * cos(vec_gridx3[k]);
                    bz[ii] = -b1[ii] * cos(vec_gridx2[j]) + b2[ii] * sin(vec_gridx2[j]);

                    vx[ii] = v1[ii] * cos(vec_gridx3[k]) * sin(vec_gridx2[j]) + v2[ii] * cos(vec_gridx3[k]) * cos(vec_gridx2[j]) - v3[ii] * sin(vec_gridx3[k]);
                    vy[ii] = v1[ii] * sin(vec_gridx3[k]) * sin(vec_gridx2[j]) + v2[ii] * sin(vec_gridx3[k]) * cos(vec_gridx2[j]) + v3[ii] * cos(vec_gridx3[k]);
                    vz[ii] = -v1[ii] * cos(vec_gridx2[j]) + v2[ii] * sin(vec_gridx2[j]);

                    rholog[ii] = log10(rho[ii]);
                    prlog[ii] = log10(pr[ii]);
                }
            }
        }

        grids[n] = mesh;
        DOSrho[n] = DOrho;
        DOSrholog[n] = DOrholog;
        DOSpress[n] = DOpress;
        DOSpresslog[n] = DOpresslog;
        DOSvel[n] = DOvel;
        DOSmagfield[n] = DOmagfield;
        DOSvel_cart[n] = DOvel_cart;
        DOSmagfield_cart[n] = DOmagfield_cart;
    }

    sprintf(buf, "%d %d", tbeg, tend);
    coDoSet *set = new coDoSet(mesh_name, grids);
    set->addAttribute("TIMESTEP", buf);
    p_mesh->setCurrentObject(set);

    set = new coDoSet(rho_name, DOSrho);
    set->addAttribute("TIMESTEP", buf);
    p_rho->setCurrentObject(set);

    set = new coDoSet(rholog_name, DOSrholog);
    set->addAttribute("TIMESTEP", buf);
    p_rholog->setCurrentObject(set);

    set = new coDoSet(pr_name, DOSpress);
    set->addAttribute("TIMESTEP", buf);
    p_pressure->setCurrentObject(set);

    set = new coDoSet(prlog_name, DOSpresslog);
    set->addAttribute("TIMESTEP", buf);
    p_pressurelog->setCurrentObject(set);

    set = new coDoSet(vel_name, DOSvel);
    set->addAttribute("TIMESTEP", buf);
    p_velocity->setCurrentObject(set);

    set = new coDoSet(magfield_name, DOSmagfield);
    set->addAttribute("TIMESTEP", buf);
    p_magfield->setCurrentObject(set);

    set = new coDoSet(vel_cart_name, DOSvel_cart);
    set->addAttribute("TIMESTEP", buf);
    p_velocity_cart->setCurrentObject(set);

    set = new coDoSet(magfield_cart_name, DOSmagfield_cart);
    set->addAttribute("TIMESTEP", buf);
    p_magfield_cart->setCurrentObject(set);

    for (i = 0; i < n; i++)
    {
        delete grids[i];
        delete DOSrho[i];
        delete DOSrholog[i];
        delete DOSpress[i];
        delete DOSpresslog[i];
        delete DOSvel[i];
        delete DOSmagfield[i];
        delete DOSvel_cart[i];
        delete DOSmagfield_cart[i];
    }
    delete[] grids;
    delete[] DOSrho;
    delete[] DOSrholog;
    delete[] DOSpress;
    delete[] DOSpresslog;
    delete[] DOSvel_cart;
    delete[] DOSmagfield_cart;

    delete[] vec_gridx1;
    delete[] vec_gridx2;
    delete[] vec_gridx3;

    return SUCCESS;
}

MODULE_MAIN(Reader, ReadPLUTO)
