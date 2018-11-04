/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description: Writing of Elements in Patran Format                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Monika Wierse                               **
 **                          SGI/Cray Research                             **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.12.97  V0.1                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "WritePatran.h"
#include <util/coviseCompat.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoPolygons.h>

//#define DEBUG

//
// static stub callback functions calling the real class
// member functions
//

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

//
//
//..........................................................................
//
//void Application::quit(void *callbackData)
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

#ifdef DEBUG
static void PrintOutMesh(int num_triangles, int *vertex_list)

{
    int i, j, count;

    cout << "num_triangles: " << num_triangles << endl;
    cout << "Triangle List: \n" << endl;

    for (i = 0; i < num_triangles; i++)
    {
        cout << i << ": ";
        count = i * 3;
        for (j = 0; j < 3; j++)
            cout << vertex_list[count + j] << " ";
        cout << "\n" << endl;
    }
}

static void PrintInMesh(int num_poly, int num_vertices, int *pl_in, int *vl_in)

{
    int i;

    cout << "num_poly: " << num_poly << " num_vertices: " << num_vertices << endl;
    cout << "Polygon List: \n" << endl;

    for (i = 0; i < num_poly; i++)
        cout << i << " " << pl_in[i] << "\n" << endl;

    for (i = 0; i < num_vertices; i++)
        cout << i << " " << vl_in[i] << "\n" << endl;
}
#endif

static int ToTriangulationForPolygons(int **vl, int num_poly, int num_vertices, int *cl, int *el)
{
    int i, j, num_triangles, *vertex_list;
    int vert;
    int count;
    int tri_count;

    num_triangles = num_poly;
    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = cl[i + 1] - cl[i]) > 3)
            num_triangles += vert - 3;
    }

    j = 0;
    while (cl[num_poly - 1] + j < num_vertices)
        j++;

    num_triangles += j - 3;

    count = 0;
    vertex_list = new int[num_triangles * 3];

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = cl[i + 1] - cl[i]) == 3)
        {
            tri_count = 3 * count;
            vertex_list[tri_count] = el[cl[i]];
            vertex_list[tri_count + 1] = el[cl[i] + 1];
            vertex_list[tri_count + 2] = el[cl[i] + 2];

            count++;
        }
        else
        {
            for (j = 1; j < vert - 1; j++)
            {
                tri_count = count * 3;
                vertex_list[tri_count] = el[cl[i]];
                vertex_list[tri_count + 1] = el[cl[i] + j];
                vertex_list[tri_count + 2] = el[cl[i] + j + 1];

                count++;
            }
        }
    }
    j = 1;
    while (cl[num_poly - 1] + j + 1 < num_vertices)
    {
        //new triangle 0, j, j+1
        tri_count = count * 3;
        vertex_list[tri_count] = el[cl[num_poly - 1]];
        vertex_list[tri_count + 1] = el[cl[num_poly - 1] + j];
        vertex_list[tri_count + 2] = el[cl[num_poly - 1] + j + 1];

        count++;
        j++;
    }

    num_vertices = 3 * (num_triangles);

    if (num_triangles != count)
    {
        Covise::sendInfo("Triangle list non-consistent!!!");
        printf("num_triangles %d count %d \n", num_triangles, count);
    }
    *vl = vertex_list;
    //  PrintOutMesh(num_triangles,vertex_list);
    return (num_triangles);
}

static int ToTriangulationForTriStrips(int **vl, int num_poly, int num_vertices, int *cl, int *el)
{
    int i, j, num_triangles, *vertex_list;
    int vert;
    int count;
    int tri_count;

    num_triangles = num_poly;
    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = cl[i + 1] - cl[i]) > 3)
            num_triangles += vert - 3;
    }

    j = 0;
    while (cl[num_poly - 1] + j < num_vertices)
        j++;

    num_triangles += j - 3;

    count = 0;
    vertex_list = new int[num_triangles * 3];

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = cl[i + 1] - cl[i]) == 3)
        {
            tri_count = 3 * count;
            vertex_list[tri_count] = el[cl[i]];
            vertex_list[tri_count + 1] = el[cl[i] + 1];
            vertex_list[tri_count + 2] = el[cl[i] + 2];

            count++;
        }
        else
        {
            for (j = 0; j < vert - 2; j++)
            {
                tri_count = count * 3;
                vertex_list[tri_count] = el[cl[i] + j];
                vertex_list[tri_count + 1] = el[cl[i] + j + 1];
                vertex_list[tri_count + 2] = el[cl[i] + j + 2];

                count++;
            }
        }
    }
    j = 0;
    while (cl[num_poly - 1] + j + 1 < num_vertices - 1)
    {
        //new triangle 0, j, j+1
        tri_count = count * 3;
        vertex_list[tri_count] = el[cl[num_poly - 1] + j];
        vertex_list[tri_count + 1] = el[cl[num_poly - 1] + j + 1];
        vertex_list[tri_count + 2] = el[cl[num_poly - 1] + j + 2];

        count++;
        j++;
    }

    num_vertices = 3 * (num_triangles);

    if (num_triangles != count)
    {
        Covise::sendInfo("Triangle list non-consistent!!!");
        printf("num_triangles %d count %d \n", num_triangles, count);
    }
    *vl = vertex_list;
    //    PrintOutMesh(num_triangles,vertex_list);
    return (num_triangles);
}

void Application::writeObject(const coDistributedObject *new_data,
                              FILE *file,
                              int verbose,
                              int indent)
{
    (void)verbose;
    (void)indent;
    int i, num_points, num_triangles;
    int *el, *cl;
    float *v[3];
    char buffer[300];

    const char *type = new_data->getType();

    printf(" TYPE: %s \n", type);
    fputs("25       0       0       1\n", file);
    fputs("OUTPUT from COVISE\n", file);

    ////////////////////////////// POLYGN || TRIANG /////////////////////////////
    if (strcmp(type, "POLYGN") == 0 || strcmp(type, "TRIANG") == 0)
    {
        int *vl;

        if (strcmp(type, "POLYGN") == 0)
        {
            const coDoPolygons *obj = (const coDoPolygons *)new_data;
            int num_poly = obj->getNumPolygons();
            int num_vertices = obj->getNumVertices();
            num_points = obj->getNumPoints();

            obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);
            //   PrintInMesh(num_poly,num_vertices,el,cl) ;
            num_triangles = ToTriangulationForPolygons(&vl, num_poly, num_vertices, el, cl);
        }
        else
        {
            const coDoTriangleStrips *obj = (const coDoTriangleStrips *)new_data;

            int num_poly = obj->getNumStrips();
            int num_vertices = obj->getNumVertices();
            num_points = obj->getNumPoints();

            obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);
            //  PrintInMesh(num_poly,num_vertices,el,cl) ;
            num_triangles = ToTriangulationForTriStrips(&vl, num_poly, num_vertices, el, cl);
        }
        sprintf(buffer, "26       0       0       1%8d%8d       0       0       0\n",
                num_points, num_triangles);
        fputs(buffer, file);
        fputs("15-APR-98   03:52:14     2.4\n", file);

        for (i = 0; i < num_points; i++) /* writing of coordinates */
        {
            sprintf(buffer, " 1%8d       0       2       0       0       0       0       0\n", i + 1);
            fputs(buffer, file);
            sprintf(buffer, "% 16.9E% 16.9E% 16.9E\n", v[0][i], v[1][i], v[2][i]);
#ifdef WIN32
            // Repair E+000 to E+00 on Win32
            size_t length = strlen(buffer);
            if (length > 50)
            {
                for (int ctr = 14; ctr < 30; ++ctr)
                {
                    buffer[ctr] = buffer[ctr + 1];
                }
                for (int ctr = 30; ctr < 46; ++ctr)
                {
                    buffer[ctr] = buffer[ctr + 2];
                }
                for (int ctr = 46; ctr < length - 2; ++ctr)
                {
                    buffer[ctr] = buffer[ctr + 3];
                }
            }
#endif

            fputs(buffer, file);
            fputs("1G       6       0       0  000000\n", file);
        }

        for (i = 0; i < num_triangles; i++)
        {
            sprintf(buffer, " 2 %7d       3       2       0       0       0       0       0\n", i + 1);
            fputs(buffer, file);
            fputs("       3       0       1       0 0.000000000E+00 0.000000000E+00 0.000000000E+00\n", file);
            sprintf(buffer, " %7d %7d %7d \n", vl[i * 3] + 1, vl[i * 3 + 1] + 1, vl[i * 3 + 2] + 1);
            fputs(buffer, file);
        }
        fputs("99      0       0       1\n", file);
        delete vl;
    }

    //////////////////// Framework for all types /////////////////////
    else
    {
        Covise::sendError("Sorry, WritePatran doesn't support type %s", type);
    }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

//void Application::compute(void *callbackData)
void Application::compute(void *)
{

    char *path;
    Covise::get_browser_param("path", &path);

    if (path == NULL)
    {
        Covise::sendError("Could not get filename");
        return;
    }
    int newFile;
    Covise::get_boolean_param("new", &newFile);
    int verbose;
    Covise::get_boolean_param("verbose", &verbose);

    FILE *file = Covise::fopen(path, (newFile ? "w" : "a"));
    if ((!file) || (fseek(file, 0, SEEK_CUR)))
    {
        Covise::sendError("Could not create file");
        return;
    }

    // ========================== Get input data ======================

    char *InputName = Covise::get_object_name("dataIn");
    if (InputName == NULL)
    {
        Covise::sendError("Error creating object name for 'dataIn'");
        fclose(file);
        return;
    }

    const coDistributedObject *new_data = coDistributedObject::createFromShm(InputName);
    if (new_data == NULL)
    {
        Covise::sendError("createFromShm() failed for data");
        fclose(file);
        return;
    }

    writeObject(new_data, file, verbose, 0);
    delete new_data;
    fclose(file);
}

/*******************************\ 
 **                             **
 **        Ex ApplMain.C        **
 **                             **
\*******************************/

int main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();
    return 0;
}
