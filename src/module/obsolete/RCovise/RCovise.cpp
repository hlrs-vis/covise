/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for COVISE USG data        	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.11.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "RCovise.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
    return 0;
}

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

void Application::parseStartStep(char *str, int *start, int *step)
{
    int i, j;
    char buf[300];

    //fprintf(stderr, "parseStartStep: %s\n", str);

    // we love parsing
    for (i = 0; str[i] && !(str[i] == '.' && str[i + 1] == '.'); i++)
        ;
    if (str[i])
    {
        i += 2;
        // we have multiple stationary files
        for (j = 0; str[i] && !(str[i] == '.' && str[i + 1] == '.'); i++)
        {
            buf[j] = str[i];
            j++;
        }
        buf[j] = '\0';
        sscanf(buf, "%d", start);
        if (str[i])
        {
            i += 2;
            // here comes the hot-stepper, trammladamm
            for (j = 0; str[i]; i++)
            {
                buf[j] = str[i];
                j++;
            }
            buf[j] = '\0';
            sscanf(buf, "%d", step);
        }
        else
            *step = 1;
    }
    else
    {
        *start = -1;
        *step = -1;
    }

    return;
}

//
//
//..........................................................................
//
void Application::quit(void * /*callbackData*/)
{
    //
    // ...... delete your data here .....
    //
}

int fp;

void Application::compute(void * /*callbackData*/)
{
    //
    // ...... do work here ........
    //

    // read input parameters and data object name
    char buf[300];
    coDistributedObject *tmp_obj;
    long num_timesteps, timestep, waste;

    Covise::get_browser_param("grid_path", &grid_Path);
    Covise::get_slider_param("timestep", &waste, &num_timesteps, &timestep);

    Mesh = Covise::get_object_name("mesh");
    Mesh_in = Covise::get_object_name("mesh_in");
    if (Mesh_in != NULL)
    {
        // WRITE DATA

        // if ((fp = open(grid_Path,O_WRONLY|O_CREAT,0660)) <0)
        if ((fp = Covise::open(grid_Path, O_WRONLY | O_CREAT)) < 0)
        {
            strcpy(buf, "ERROR: Can't open file >> ");
            strcat(buf, grid_Path);
            Covise::sendError(buf);
            return;
        }
        tmp_obj = new coDistributedObject(Mesh_in);
        writeobj(tmp_obj);
        delete tmp_obj;
        close(fp);
    }
    else
    {
        // READ DATA

        if (num_timesteps)
        {
            int start, step;
            parseStartStep(grid_Path, &start, &step);
            if ((start != -1) && step)
            {
                int i;
                char buf[300];
                for (i = 0; !(grid_Path[i] == '.' && grid_Path[i + 1] == '.'); i++)
                    buf[i] = grid_Path[i];
                buf[i] = '\0';
                grid_Path = new char[300];
                //sprintf(grid_Path, buf, start+step*(timestep-1));
                sprintf(grid_Path, buf, start + step * (timestep));

                //fprintf(stderr, "using file %s \n", grid_Path);
            }
        }

        if ((fp = Covise::open(grid_Path, O_RDONLY)) < 0)
        {
            strcpy(buf, "ERROR: Can't open file >> ");
            strcat(buf, grid_Path);
            Covise::sendError(buf);
            return;
        }

        tmp_obj = readData(Mesh);
        if (num_timesteps)
        {
            char buf[300];
            sprintf(buf, "%ld %ld", timestep, num_timesteps);
            tmp_obj->addAttribute("BLOCKINFO", buf);

            sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            tmp_obj->addAttribute("READ_MODULE", buf);
        }
        delete tmp_obj;

        close(fp);
    }
}

void Application::writeobj(coDistributedObject *tmp_Object)
{
    coDoSet *set;
    coDoGeometry *geo;
    coDoRGBA *rgba;
    coDoLines *lin;
    coDoPoints *pts;
    char *gtype;
    USG_HEADER usg_h;
    STR_HEADER s_h;
    coDistributedObject *data_obj;
    coDistributedObject *do1;
    coDistributedObject *do2;
    coDistributedObject *do3;
    coDistributedObject *const *objs;
    int numsets, i, t1, t2, t3;
    data_obj = tmp_Object->createUnknown();
    if (data_obj != 0L)
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "SETELE") == 0)
        {
            set = (coDoSet *)data_obj;
            objs = set->getAllElements(&numsets);
            write(fp, gtype, 6);
            write(fp, &numsets, sizeof(int));
            for (i = 0; i < numsets; i++)
                writeobj(objs[i]);
            writeattrib(data_obj);
            delete set;
        }
        else if (strcmp(gtype, "GEOMET") == 0)
        {
            geo = (coDoGeometry *)data_obj;
            do1 = geo->getGeometry();
            do2 = geo->get_colors();
            do3 = geo->get_normals();
            t1 = geo->getGeometry_type();
            t2 = geo->get_color_attr();
            t3 = geo->get_normal_attr();
            write(fp, gtype, 6);
            write(fp, &do1, sizeof(int));
            write(fp, &do2, sizeof(int));
            write(fp, &do3, sizeof(int));
            write(fp, &t1, sizeof(int));
            write(fp, &t2, sizeof(int));
            write(fp, &t3, sizeof(int));
            if (do1)
                writeobj(do1);
            if (do2)
                writeobj(do2);
            if (do3)
                writeobj(do3);
            writeattrib(data_obj);
            delete geo;
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            mesh = (coDoUnstructuredGrid *)data_obj;
            mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
            mesh->getTypeList(&tl);
            mesh->getGridSize(&(usg_h.n_elem), &(usg_h.n_conn), &(usg_h.n_coord));
            write(fp, gtype, 6);
            write(fp, &usg_h, sizeof(usg_h));
            write(fp, el, usg_h.n_elem * sizeof(int));
            write(fp, tl, usg_h.n_elem * sizeof(int));
            write(fp, vl, usg_h.n_conn * sizeof(int));
            write(fp, x_coord, usg_h.n_coord * sizeof(float));
            write(fp, y_coord, usg_h.n_coord * sizeof(float));
            write(fp, z_coord, usg_h.n_coord * sizeof(float));
            writeattrib(data_obj);
            delete mesh;
        }
        else if (strcmp(gtype, "POINTS") == 0)
        {
            pts = (coDoPoints *)data_obj;
            pts->getAddresses(&x_coord, &y_coord, &z_coord);
            write(fp, gtype, 6);
            n_elem = pts->getNumPoints();
            write(fp, &n_elem, sizeof(int));
            write(fp, x_coord, n_elem * sizeof(float));
            write(fp, y_coord, n_elem * sizeof(float));
            write(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete pts;
        }
        else if (strcmp(gtype, "POLYGN") == 0)
        {
            pol = (coDoPolygons *)data_obj;
            pol->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
            usg_h.n_elem = pol->getNumPolygons();
            usg_h.n_conn = pol->getNumVertices();
            usg_h.n_coord = pol->getNumPoints();
            write(fp, gtype, 6);
            write(fp, &usg_h, sizeof(usg_h));
            write(fp, el, usg_h.n_elem * sizeof(int));
            write(fp, vl, usg_h.n_conn * sizeof(int));
            write(fp, x_coord, usg_h.n_coord * sizeof(float));
            write(fp, y_coord, usg_h.n_coord * sizeof(float));
            write(fp, z_coord, usg_h.n_coord * sizeof(float));
            writeattrib(data_obj);
            delete pol;
        }
        else if (strcmp(gtype, "LINES") == 0)
        {
            lin = (coDoLines *)data_obj;
            lin->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
            usg_h.n_elem = lin->getNumLines();
            usg_h.n_conn = lin->getNumVertices();
            usg_h.n_coord = lin->getNumPoints();
            write(fp, gtype, 6);
            write(fp, &usg_h, sizeof(usg_h));
            write(fp, el, usg_h.n_elem * sizeof(int));
            write(fp, vl, usg_h.n_conn * sizeof(int));
            write(fp, x_coord, usg_h.n_coord * sizeof(float));
            write(fp, y_coord, usg_h.n_coord * sizeof(float));
            write(fp, z_coord, usg_h.n_coord * sizeof(float));
            writeattrib(data_obj);
            delete lin;
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            tri = (coDoTriangleStrips *)data_obj;
            tri->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
            usg_h.n_elem = tri->getNumStrips();
            usg_h.n_conn = tri->getNumVertices();
            usg_h.n_coord = tri->getNumPoints();
            write(fp, gtype, 6);
            write(fp, &usg_h, sizeof(usg_h));
            write(fp, el, usg_h.n_elem * sizeof(int));
            write(fp, vl, usg_h.n_conn * sizeof(int));
            write(fp, x_coord, usg_h.n_coord * sizeof(float));
            write(fp, y_coord, usg_h.n_coord * sizeof(float));
            write(fp, z_coord, usg_h.n_coord * sizeof(float));
            writeattrib(data_obj);
            delete tri;
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid = (coDoRectilinearGrid *)data_obj;
            rgrid->getAddresses(&x_coord, &y_coord, &z_coord);
            rgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            write(fp, gtype, 6);
            write(fp, &s_h, sizeof(s_h));
            write(fp, x_coord, s_h.xs * sizeof(int));
            write(fp, y_coord, s_h.xs * sizeof(int));
            write(fp, z_coord, s_h.xs * sizeof(int));
            writeattrib(data_obj);
            delete rgrid;
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid = (coDoStructuredGrid *)data_obj;
            sgrid->getAddresses(&x_coord, &y_coord, &z_coord);
            sgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            write(fp, gtype, 6);
            write(fp, &s_h, sizeof(s_h));
            write(fp, x_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
            write(fp, y_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
            write(fp, z_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
            writeattrib(data_obj);
            delete sgrid;
        }
        else if (strcmp(gtype, "USTSDT") == 0)
        {
            us3d = (coDoFloat *)data_obj;
            us3d->getAddress(&z_coord);
            n_elem = us3d->getNumPoints();
            write(fp, gtype, 6);
            write(fp, &n_elem, sizeof(int));
            write(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete us3d;
        }
        else if (strcmp(gtype, "RGBADT") == 0)
        {
            rgba = (coDoRGBA *)data_obj;
            rgba->getAddress((int **)(&z_coord));
            n_elem = rgba->getNumElements();
            write(fp, gtype, 6);
            write(fp, &n_elem, sizeof(int));
            write(fp, z_coord, n_elem * sizeof(int));
            writeattrib(data_obj);
            delete rgba;
        }
        else if (strcmp(gtype, "USTVDT") == 0)
        {
            us3dv = (coDoVec3 *)data_obj;
            us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
            n_elem = us3dv->getNumPoints();
            write(fp, gtype, 6);
            write(fp, &n_elem, sizeof(int));
            write(fp, x_coord, n_elem * sizeof(float));
            write(fp, y_coord, n_elem * sizeof(float));
            write(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete us3dv;
        }
        else
        {
            Covise::sendError("ERROR: unsupported DataType");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'mesh_in'");
        return;
    }
}

void Application::writeattrib(coDistributedObject *tmp_Object)
{
    int numattrib, size, i;
    char **an, **at;
    numattrib = tmp_Object->get_all_attributes(&an, &at);
    size = sizeof(int);
    for (i = 0; i < numattrib; i++)
        size += strlen(an[i]) + strlen(at[i]) + 2;
    write(fp, &size, sizeof(int));
    write(fp, &numattrib, sizeof(int));
    for (i = 0; i < numattrib; i++)
    {
        write(fp, an[i], strlen(an[i]) + 1);
        write(fp, at[i], strlen(at[i]) + 1);
    }
}

void Application::readattrib(coDistributedObject *tmp_Object)
{
    int numattrib = 0, size = 0, i;
    char *an, *at;
    char *buf;
    read(fp, &size, sizeof(int));
    size -= sizeof(int);
    read(fp, &numattrib, sizeof(int));
    if (size > 0)
    {
        buf = new char[size];
        read(fp, buf, size);
        an = buf;
        for (i = 0; i < numattrib; i++)
        {
            at = an;
            while (*at)
                at++;
            at++;
            tmp_Object->addAttribute(an, at);
            an = at;
            while (*an)
                an++;
            an++;
        }
        delete[] buf;
    }
}

coDistributedObject *Application::readData(char *Name)
{
    coDoSet *set;
    coDoGeometry *geo = NULL;
    coDoRGBA *rgba;
    coDoLines *lin;
    coDoPoints *pts;
    char buf[300], Data_Type[7];
    USG_HEADER usg_h;
    STR_HEADER s_h;
    coDistributedObject **tmp_objs, *do1, *do2, *do3;
    int numsets, i, t1, t2, t3;

    read(fp, Data_Type, 6);
    Data_Type[6] = '\0';

    if (Mesh != NULL)
    {
        if (strcmp(Data_Type, "SETELE") == 0)
        {
            read(fp, &numsets, sizeof(int));
            tmp_objs = new coDistributedObject *[numsets];

            if (numsets == 1)
            {

                do1 = readData(Name);
                readattrib(do1);
                return (do1);
            }
            else
            {
                for (i = 0; i < numsets; i++)
                {
                    sprintf(buf, "%s_%d", Name, i);
                    tmp_objs[i] = readData(buf);
                }
                tmp_objs[i] = NULL;
                set = new coDoSet(Name, tmp_objs);
                if (!(set->objectOk()))
                {
                    Covise::sendError("ERROR: creation of SETELE object 'mesh' failed");
                    return (NULL);
                }
                for (i = 0; i < numsets; i++)
                {
                    delete tmp_objs[i];
                }
            }
            delete[] tmp_objs;
            readattrib(set);
            return (set);
        }
        else if (strcmp(Data_Type, "GEOMET") == 0)
        {
            read(fp, &do1, sizeof(int));
            read(fp, &do2, sizeof(int));
            read(fp, &do3, sizeof(int));
            read(fp, &t1, sizeof(int));
            read(fp, &t2, sizeof(int));
            read(fp, &t3, sizeof(int));
            if (do1)
            {
                sprintf(buf, "%s_Geo", Name);
                do1 = readData(buf);
            }
            if (do2)
            {
                sprintf(buf, "%s_Col", Name);
                do2 = readData(buf);
            }
            if (do3)
            {
                sprintf(buf, "%s_Norm", Name);
                do3 = readData(buf);
            }

            if (do1)
            {
                geo = new coDoGeometry(Name, do1);
                if (!(geo->objectOk()))
                {
                    Covise::sendError("ERROR: creation of GEOMET object 'mesh' failed");
                    return (NULL);
                }
                //geo->setGeometry(t1, do1);
            }
            if (do1 && do2)
            {
                geo->setColor(t2, do2);
            }
            if (do1 && do3)
            {
                geo->setNormal(t3, do3);
            }
            if (do1)
                delete do1;
            if (do2)
                delete do2;
            if (do3)
                delete do3;

            readattrib(geo);
            return (geo);
        }
        else if (strcmp(Data_Type, "UNSGRD") == 0)
        {
            read(fp, &usg_h, sizeof(usg_h));
            mesh = new coDoUnstructuredGrid(Name, usg_h.n_elem, usg_h.n_conn, usg_h.n_coord, 1);
            if (mesh->objectOk())
            {
                mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
                mesh->getTypeList(&tl);
                read(fp, el, usg_h.n_elem * sizeof(int));
                read(fp, tl, usg_h.n_elem * sizeof(int));
                read(fp, vl, usg_h.n_conn * sizeof(int));
                read(fp, x_coord, usg_h.n_coord * sizeof(float));
                read(fp, y_coord, usg_h.n_coord * sizeof(float));
                read(fp, z_coord, usg_h.n_coord * sizeof(float));
                readattrib(mesh);
                return (mesh);
            }
            else
            {
                Covise::sendError("ERROR: creation of UNSGRD object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "POLYGN") == 0)
        {
            read(fp, &usg_h, sizeof(usg_h));
            pol = new coDoPolygons(Name, usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            if (pol->objectOk())
            {
                pol->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                read(fp, el, usg_h.n_elem * sizeof(int));
                read(fp, vl, usg_h.n_conn * sizeof(int));
                read(fp, x_coord, usg_h.n_coord * sizeof(float));
                read(fp, y_coord, usg_h.n_coord * sizeof(float));
                read(fp, z_coord, usg_h.n_coord * sizeof(float));
                readattrib(pol);
                return (pol);
            }
            else
            {
                Covise::sendError("ERROR: creation of POLYGN object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "POINTS") == 0)
        {
            read(fp, &n_elem, sizeof(int));
            pts = new coDoPoints(Name, n_elem);
            if (pts->objectOk())
            {
                pts->getAddresses(&x_coord, &y_coord, &z_coord);
                read(fp, x_coord, n_elem * sizeof(float));
                read(fp, y_coord, n_elem * sizeof(float));
                read(fp, z_coord, n_elem * sizeof(float));
                readattrib(pts);
                return (pts);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "LINES") == 0)
        {
            read(fp, &usg_h, sizeof(usg_h));
            lin = new coDoLines(Name, usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            if (lin->objectOk())
            {
                lin->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                read(fp, el, usg_h.n_elem * sizeof(int));
                read(fp, vl, usg_h.n_conn * sizeof(int));
                read(fp, x_coord, usg_h.n_coord * sizeof(float));
                read(fp, y_coord, usg_h.n_coord * sizeof(float));
                read(fp, z_coord, usg_h.n_coord * sizeof(float));
                readattrib(lin);
                return (lin);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "TRIANG") == 0)
        {
            read(fp, &usg_h, sizeof(usg_h));
            tri = new coDoTriangleStrips(Name, usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            if (tri->objectOk())
            {
                tri->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                read(fp, el, usg_h.n_elem * sizeof(int));
                read(fp, vl, usg_h.n_conn * sizeof(int));
                read(fp, x_coord, usg_h.n_coord * sizeof(float));
                read(fp, y_coord, usg_h.n_coord * sizeof(float));
                read(fp, z_coord, usg_h.n_coord * sizeof(float));
                readattrib(tri);
                return (tri);
            }
            else
            {
                Covise::sendError("ERROR: creation of TRIANG object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "RCTGRD") == 0)
        {
            read(fp, &s_h, sizeof(s_h));
            rgrid = new coDoRectilinearGrid(Name, s_h.xs, s_h.ys, s_h.zs);
            if (rgrid->objectOk())
            {
                rgrid->getAddresses(&x_coord, &y_coord, &z_coord);
                read(fp, x_coord, s_h.xs * sizeof(float));
                read(fp, y_coord, s_h.ys * sizeof(float));
                read(fp, z_coord, s_h.zs * sizeof(float));
                readattrib(rgrid);
                return (rgrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of RCTGRD object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "STRGRD") == 0)
        {
            read(fp, &s_h, sizeof(s_h));
            sgrid = new coDoStructuredGrid(Name, s_h.xs, s_h.ys, s_h.zs);
            if (sgrid->objectOk())
            {
                sgrid->getAddresses(&x_coord, &y_coord, &z_coord);
                read(fp, x_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
                read(fp, y_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
                read(fp, z_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
                readattrib(sgrid);
                return (sgrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of STRGRD object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "USTSDT") == 0)
        {
            read(fp, &n_elem, sizeof(int));
            us3d = new coDoFloat(Name, n_elem);
            if (us3d->objectOk())
            {
                us3d->getAddress(&x_coord);
                read(fp, x_coord, n_elem * sizeof(float));
                readattrib(us3d);
                return (us3d);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTSDT object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "RGBADT") == 0)
        {
            read(fp, &n_elem, sizeof(int));
            rgba = new coDoRGBA(Name, n_elem);
            if (rgba->objectOk())
            {
                rgba->getAddress((int **)(&x_coord));
                read(fp, x_coord, n_elem * sizeof(int));
                readattrib(rgba);
                return (rgba);
            }
            else
            {
                Covise::sendError("ERROR: creation of RGBADT object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "USTVDT") == 0)
        {
            read(fp, &n_elem, sizeof(int));
            us3dv = new coDoVec3(Name, n_elem);
            if (us3dv->objectOk())
            {
                us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
                read(fp, x_coord, n_elem * sizeof(float));
                read(fp, y_coord, n_elem * sizeof(float));
                read(fp, z_coord, n_elem * sizeof(float));
                readattrib(us3dv);
                return (us3dv);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTVDT object 'mesh' failed");
                return (NULL);
            }
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct ");
        return (NULL);
    }
    return (NULL);
}
