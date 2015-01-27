/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface reduction application module              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  April 1997  V1.0                                                **
 \**************************************************************************/

#include <appl/ApplInterface.h>
#include "RW_AVS_TriMesh.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>

char *MeshIn, *MeshOut;
char *DataIn, *DataOut;

#define INDEX 1
#define UCD 2
//  Shared memory data
coDoPolygons *mesh_in = NULL;
coDoPolygons *mesh_out = NULL;
coDoFloat *data_in = NULL;
coDoFloat *data_out = NULL;

//
// static stub callback functions calling the real class
// member functions
//

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
    return 0;
}

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//======================================================================
// Computation routine (called when START message arrives)
//======================================================================
void Application::compute(void *)
{
    const coDistributedObject *tmp_obj_1;
    const coDistributedObject *tmp_obj_2;
    char *filename;

    //	get parameter
    Covise::get_browser_param("filename", &filename);

    MeshIn = Covise::get_object_name("meshIn");
    DataIn = Covise::get_object_name("dataIn");

    MeshOut = Covise::get_object_name("meshOut");
    DataOut = Covise::get_object_name("dataOut");

    mesh_in = NULL;
    mesh_out = NULL;

    //	retrieve data object from shared memory
    tmp_obj_1 = coDistributedObject::createFromShm(MeshIn);
    tmp_obj_2 = coDistributedObject::createFromShm(DataIn);

    HandleObjects(tmp_obj_1, tmp_obj_2, MeshOut, DataOut, filename);

    //if(MeshIn) delete tmp_obj_1;
    //if(DataIn) delete tmp_obj_2;
}

void Application::HandleObjects(const coDistributedObject *mesh_object, const coDistributedObject *data_object, char *Mesh_out_name, char *Data_out_name, char *filename)
{
    const char *dtype, *gtype;

    int n_t, n_v, n_c;
    int *tl, *vl;
    float *cx, *cy, *cz, *dt;
    int numdata;
    RW_TriangleMesh *surface;

    if (mesh_object == NULL)
    {
        surface = new RW_TriangleMesh();
        surface->Read_TriangleMesh(filename);
        surface->CreatecoDistributedObjects(Mesh_out_name, Data_out_name);
        delete surface;
    }
    else
    {
        if (mesh_object->objectOk())
        {
            gtype = mesh_object->getType();

            if (strcmp(gtype, "POLYGN") == 0)
            {
                mesh_in = (coDoPolygons *)mesh_object;
                n_c = mesh_in->getNumPoints();
                n_v = mesh_in->getNumVertices();
                n_t = mesh_in->getNumPolygons();
                mesh_in->getAddresses(&cx, &cy, &cz, &vl, &tl);
            }
            else
            {
                Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
            return;
        }

        if (n_c == 0 || n_v == 0 || n_t == 0)
        {
            Covise::sendError("ERROR: Data object 'meshIn' is empty");
            return;
        }

        if (data_object != NULL)
        {
            if (data_object->objectOk())
            {
                dtype = data_object->getType();

                if (strcmp(dtype, "USTSDT") == 0)
                {
                    data_in = (coDoFloat *)data_object;
                    numdata = data_in->getNumPoints();
                    data_in->getAddress(&dt);
                }
                else
                {
                    Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
                    return;
                }
            }
            else
            {
                Covise::sendError("ERROR: Data object 'dataIn' can't be accessed in shared memory");
                return;
            }

            if (n_c != numdata)
            {
                Covise::sendError("ERROR: Data object 'dataIn' has not the same dimensions as 'meshIn'");
                return;
            }
        }
        else
            dt = NULL;

        surface = new RW_TriangleMesh(n_t, n_v, n_c, tl, vl, cx, cy, cz, dt);
        surface->Write_TriangleMesh(filename);
        delete surface;
    }

    if (mesh_in)
        delete mesh_in;
    if (data_in)
        delete data_in;

    return;
}

int RW_TriangleMesh::Write_TriangleMesh(char *filename)
{
    FILE *file_out;

    int mode = INDEX;
    if (strcmp(filename + strlen(filename) - 4, ".idx") != 0)
    {
        mode = UCD;
    }
    if ((file_out = Covise::fopen(filename, "w")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s", filename);
        return (0);
    }
    if (mode == INDEX)
    {
        fprintf(file_out, "Triangle mesh in INDEX Exchange format\n");
        //fprintf(file_out, "\n");

        /////////////////////////////////////////////////////////////////////
        // Write surface parameters to file:                               //
        // number of triangles, number of vertices, number of coordinates  //
        // Data                                                            //
        /////////////////////////////////////////////////////////////////////

        fprintf(file_out, "%d  %d  %d %d\n", num_tri, num_vert, num_points, Data);
        //fprintf(file_out, "\n");

        /////////////////////////////////////////////////////////////////////
        // Write Triangle List to file                                     //
        /////////////////////////////////////////////////////////////////////
        int i;
        for (i = 0; i < num_tri; i++)
            fprintf(file_out, "%d\n", tri_list[i]);

        //fprintf(file_out, "\n");

        /////////////////////////////////////////////////////////////////////
        // Write Vertex List to file                                       //
        /////////////////////////////////////////////////////////////////////

        for (i = 0; i < num_vert; i = i + 3)
            fprintf(file_out, "%d %d %d\n", vertex_list[i], vertex_list[i + 1], vertex_list[i + 2]);

        //fprintf(file_out, "\n");

        /////////////////////////////////////////////////////////////////////
        // Write Coordinates and Scalar Data (if existent) to file         //
        /////////////////////////////////////////////////////////////////////

        for (i = 0; i < num_points; i++)
        {
            fprintf(file_out, "%f  %f  %f  ", coords_x[i], coords_y[i], coords_z[i]);
            if (Data)
                fprintf(file_out, "%f\n", data[i]);
            else
                fprintf(file_out, "\n");
        }
    }
    else
    { // avs UCD
        fprintf(file_out, "#UCD triangle file created by COVISE\n");
        fprintf(file_out, "%d %d 0 0 0\n", num_points, num_tri);
        for (int i = 0; i < num_points; i++)
        {
            fprintf(file_out, "%d %f %f %f\n", i + 1, coords_x[i], coords_y[i], coords_z[i]);
        }
        for (int i = 0; i < num_tri; i++)
        {
            fprintf(file_out, "%d 1 tri %d %d %d\n", i + 1, vertex_list[tri_list[i]] + 1, vertex_list[tri_list[i] + 1] + 1, vertex_list[tri_list[i] + 2] + 1);
        }
    }

    fclose(file_out);
    return (1);
}

int RW_TriangleMesh::Read_TriangleMesh(char *filename)
{
    FILE *file_in;
    char buf[300];
    int i;
    int mode = INDEX;
    if (strcmp(filename + strlen(filename) - 4, ".idx") != 0)
    {
        mode = UCD;
    }
    if ((file_in = Covise::fopen(filename, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s", filename);
        return (0);
    }

    if (fgets(buf, 300, file_in) != NULL)
    {
        fprintf(stderr, "fgets_1 failed in ReadNasASC.cpp");
    }

    /////////////////////////////////////////////////////////////////////
    // Read surface parameters from file:                              //
    // number of triangles, number of vertices, number of coordinates  //
    // Data                                                            //
    /////////////////////////////////////////////////////////////////////

    do
    {
        if (fgets(buf, 300, file_in) != NULL)
        {
            fprintf(stderr, "fgets_4 failed in ReadNasASC.cpp");
        }
    } while (buf[0] == '#');
    if (mode == INDEX)
    {
        sscanf(buf, "%d%d%d%d", &num_tri, &num_vert, &num_points, &Data);
    }
    else
    {
        sscanf(buf, "%d%d%d", &num_points, &num_tri, &Data);
        num_vert = num_tri * 3;
    }

    /////////////////////////////////////////////////////////////////////
    // Create dynamic lists                                            //
    /////////////////////////////////////////////////////////////////////

    tri_list = new int[num_tri];
    vertex_list = new int[num_vert];
    coords_x = new float[num_points];
    coords_y = new float[num_points];
    coords_z = new float[num_points];
    if (Data)
        data = new float[num_points];
    else
        data = NULL;

    if (mode == INDEX)
    {
        /////////////////////////////////////////////////////////////////////
        // Read Triangle List from file                                    //
        /////////////////////////////////////////////////////////////////////

        for (i = 0; i < num_tri; i++)
            if (fscanf(file_in, "%d", &tri_list[i]) != 1)
            {
                fprintf(stderr, "fscanf_2 failed in ReadLat.cpp");
            }

        //if(fgets(buf, 300, file_in) == NULL)
        //{
        //  Covise::sendError("Unexpected end of file in triangle list");
        //  return(0);
        //}

        /////////////////////////////////////////////////////////////////////
        // Read Vertex List from file                                      //
        /////////////////////////////////////////////////////////////////////

        for (i = 0; i < num_vert; i = i + 3)
            if (fscanf(file_in, "%d %d %d", &vertex_list[i], &vertex_list[i + 1], &vertex_list[i + 2]) != 3)
            {
                fprintf(stderr, "fscanf_3 failed in ReadLat.cpp");
            }

        //if(fgets(buf, 300, file_in) == NULL)
        //{
        //  Covise::sendError("Unexpected end of file in vertex list");
        //  return(0);
        //}

        /////////////////////////////////////////////////////////////////////
        // Read Coordinates and Scalar Data (if existent) from file        //
        /////////////////////////////////////////////////////////////////////

        if (!Data)
        {
            for (i = 0; i < num_points; i++)
                if (fscanf(file_in, "%f%f%f", &coords_x[i], &coords_y[i], &coords_z[i]) != 3)
                {
                    fprintf(stderr, "fscanf_4 failed in ReadLat.cpp");
                }
                else
                {
                    for (i = 0; i < num_points; i++)
                        if (fscanf(file_in, "%f%f%f%f", &coords_x[i], &coords_y[i], &coords_z[i], &data[i]) != 4)
                        {
                            fprintf(stderr, "fscanf_5 failed in ReadLat.cpp");
                        }
                }
        }

        //if(fgets(buf, 300, file_in) == NULL && i < num_points)
        //{
        //  Covise::sendError("Unexpected end of file in coordinates");
        //  return(0);
        //}
    }
    else
    {
        int num;
        int n = 0;
        char type[100];
        for (i = 0; i < num_points; i++)
        {
            if (fscanf(file_in, "%d %f %f %f", &num, &coords_x[i], &coords_y[i], &coords_z[i]) != 4)
            {
                fprintf(stderr, "fscanf_4 failed in read UCD");
            }
        }
        for (i = 0; i < num_vert; i = i + 3)
        {
            if (fscanf(file_in, "%d %d %s %d %d %d", &num, &num, type, &vertex_list[i], &vertex_list[i + 1], &vertex_list[i + 2]) != 6)
            {
                fprintf(stderr, "fscanf_3 failed in UCD");
            }
            vertex_list[i]--;
            vertex_list[i + 1]--;
            vertex_list[i + 2]--;
            tri_list[n] = i;
            n++;
        }
    }

    fclose(file_in);
    return (0);
}

void RW_TriangleMesh::CreatecoDistributedObjects(char *Triangle_name, char *Data_name)
{
    ////////////////////////////////////////////////////////////////////////
    // This procedure is COVISE-related: It produces the data objects in  //
    // shared memory to be given to the next module (renderer, simplifier,//
    // or any module working with geometry objects).                      //
    // It has to be replaced e.g. by the according AVS procedure making a //
    // field structure out of the connected lists.                        //
    ////////////////////////////////////////////////////////////////////////

    coDoPolygons *polygons_out;
    coDoFloat *scalar_out;

    polygons_out = new coDoPolygons(Triangle_name, num_points, coords_x, coords_y, coords_z, num_vert, vertex_list, num_tri, tri_list);

    if (!polygons_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'meshOut' failed");
        return;
    }
    polygons_out->addAttribute("vertexOrder", "2");
    polygons_out->addAttribute("COLOR_BINDING", "PER_VERTEX");
    polygons_out->addAttribute("NORMAL_BINDING", "PER_VERTEX");
    polygons_out->addAttribute("COLOR", "blue");

    if (Data)
    {
        scalar_out = new coDoFloat(Data_name, num_points, data);

        if (!scalar_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }

        delete scalar_out;
    }
    else
        scalar_out = NULL;

    delete polygons_out;

    return;
}
