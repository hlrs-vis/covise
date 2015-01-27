/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//
// Read a uniform 3d grid of scalar data
// The data is binar
// 22.06.98				Yury

#include <appl/ApplInterface.h>
#include "READ_VPMODELL.h"

//
//
int main(int argc, char *argv[])
{
    //init
    Application *application = new Application(argc, argv);

    //and back to covise
    application->run();

    //done
    return 1;
}

// computeCallback (do nothing but call our real compute-function)

void Application::computeCallback(void *userData, void *)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute();
}

// this is our compute-routine
void Application::compute() ///here will be readed the xsnap_3d data
{

    Covise::get_browser_param("datapath", &datapath);

    Covise::get_vector_param("n1_n2_n3", 0, &n1);
    Covise::get_vector_param("n1_n2_n3", 1, &n2);
    Covise::get_vector_param("n1_n2_n3", 2, &n3);

    Covise::get_vector_param("o1_o2_o3", 0, &o1);
    Covise::get_vector_param("o1_o2_o3", 1, &o2);
    Covise::get_vector_param("o1_o2_o3", 2, &o3);

    Covise::get_vector_param("d1_d2_d3", 0, &d1);
    Covise::get_vector_param("d1_d2_d3", 1, &d2);
    Covise::get_vector_param("d1_d2_d3", 2, &d3);

    Covise::get_scalar_param("n_bytes", &n_bytes);

    //the read-function itself
    int nx, ny, nz; //anzahl von knoten und nicht von abstande
    float x0, y0, z0, dx, dy, dz;

    float *fvalues;
    unsigned char *cvalues;
    int i, j, k, n;

    char *data_name;
    char *grid_name;
    float x_min, y_min, z_min, x_max, y_max, z_max;
    int position;
    char message[100] = "";

    nx = n1;
    ny = n2;
    nz = n3;
    x0 = o1;
    y0 = o2;
    z0 = o3;
    dx = d1;
    dy = d2;
    dz = d3;

    data_name = Covise::get_object_name("data");
    grid_name = Covise::get_object_name("grid");

    coDoFloat *str_s3d_out; // = NULL;
    coDoUniformGrid *grid_out; // = NULL;

    // Öffnung der Datei
    //position=( n1 * n2 * n3 * n_bytes * time );
    position = (n1 * n2 * n3 * n_bytes * 0);
    //fstream datei;
    //datei.open(datapath,ios::in);
    //if (!datei) // datei was not opened
    struct stat sbuf;
    if (stat(datapath, &sbuf) < 0)
    {
        char buf[1000];
        sprintf(buf, "Could not find File %s", datapath);
        Covise::sendError(buf);
        return;
    }
    int file_size = sbuf.st_size;
    if (file_size != (n1 * n2 * n3 * n_bytes * 1))
    {
        char buf[1000];
        sprintf(buf, "file_size is not equal to (n1 * n2 * n3 * n_bytes )");
        Covise::sendError(buf);
        sprintf(buf, "file_size =%d bytes.", file_size);
        Covise::sendError(buf);

        return;
    }

    int file;
    file = Covise::open(datapath, O_RDONLY);
    if (file < 0)
    {
        char buf[1000];
        sprintf(buf, "Could not open File %s", datapath);
        Covise::sendError(buf);
        return;
    }

    sprintf(message, "File %s was opened and %d bytes will be read.",
            datapath, file_size);
    Covise::sendInfo(message);

    lseek(file, position, SEEK_SET);
    //datei.seekg( position,ios::beg );

    x_min = x0;
    y_min = y0;
    z_min = z0;
    x_max = x0 + ((nx - 1) * dx);
    y_max = y0 + ((ny - 1) * dy);
    z_max = z0 + ((nz - 1) * dz);

    // Daten einlessen
    cvalues = NULL;
    fvalues = NULL;

    //  	int format
    if (n_bytes == 1)
    {

        cvalues = new unsigned char[nx * ny * nz];
        if (read(file, cvalues, nx * ny * nz) < (nx * ny * nz))
        {
            Covise::sendError("Unexpected end of file");
            return;
        }
    }

    // 	float format
    if (n_bytes == 4)
    {

        fvalues = new float[nx * ny * nz];
        if (read(file, fvalues, nx * ny * nz * sizeof(float)) < nx * ny * nz * sizeof(float))
        {
            Covise::sendError("Unexpected end of file");
            return;
        }
    }

    close(file);
    Covise::sendInfo("The data was read");

    cout << "nx :\t" << nx << "\tny : \t" << ny << "\tnz : \t" << nz
         << "\n x0: \t" << x0
         << "\t y0: \t" << y0
         << "\t z0: \t" << z0
         << "\n dx: \t" << dx
         << "\t dy: \t" << dy
         << "\t dz: \t" << dz << endl;

    cout << "x_min=" << x_min << "\tx_max" << x_max << "\n";
    cout << "y_min=" << y_min << "\ty_max" << y_max << "\n";
    cout << "z_min=" << z_min << "\tz_max" << z_max << "\n";

    sprintf(message, "Changing data to covise format");
    Covise::sendInfo(message);

    str_s3d_out = new coDoFloat(data_name, nx, ny, nz);
    float *daten;
    str_s3d_out->getAddress(&daten);
    n = 0;
    if (n_bytes == 1)
    {
        for (k = 0; k < nz; k++)
        {
            for (j = 0; j < ny; j++)
            {
                for (i = 0; i < nx; i++)
                {

                    // output to covise in form
                    // data(i0;j0;k0) data(i0;j0;k1) data(i0;j0;k2) ...

                    daten[((i * ny * nz) + (j * nz) + k)] = (float)cvalues[n];
                    n++;
                }
            }
        }
    }
    else if (n_bytes == 4)
    {
        for (k = 0; k < nz; k++)
        {
            for (j = 0; j < ny; j++)
            {
                for (i = 0; i < nx; i++)
                {

                    // output to covise in form
                    // data(i0;j0;k0) data(i0;j0;k1) data(i0;j0;k2) ...

                    daten[((i * ny * nz) + (j * nz) + k)] = fvalues[n];
                    n++;
                }
            }
        }
    }
    //str_s3d_out = new coDoFloat( data_name, nx, ny, nz, daten);
    grid_out = new coDoUniformGrid(grid_name, nx, ny, nz, x_min, x_max, y_min, y_max, z_min, z_max);

    delete grid_out;
    delete str_s3d_out;
    delete[] cvalues;
    delete[] fvalues;

} // end of compute
