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
#define __SYS_DIR_H__
// do not include sys/dir.h
extern "C" {
#include "hpss_api.h"
}

#include <appl/ApplInterface.h>
#include "RWHPSS.h"
#include <covise/covise_config.h>
int isHPSS = 0;

//=====================================================================
//
//=====================================================================
int myOpen(const char *file, int mode)
{
    char buf[800], *dirname, *covisepath;
    int fp, i;
    isHPSS = 0;
    if (file == NULL)
        return (-1);
    if (strncmp(file, "hpss://", 7) == 0)
    {
        isHPSS = 1;
        return hpss_Open((char *)file + 6, mode, 0660, NULL, NULL, NULL);
    }

    fp = ::open(file, mode, 0660);
    if (fp > 0)
        return (fp);

    if ((covisepath = getenv("COVISE_PATH")) == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: COVISE_PATH not defined!\n");
        return (-1);
    };

    dirname = strtok(strdup(covisepath), ":");
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::open(buf, mode, 0660);
        if (fp > 0)
            return (fp);
        for (i = strlen(dirname) - 2; i > 0; i--)
        {
            if (dirname[i] == '/')
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::open(buf, mode, 0660);
        if (fp > 0)
            return (fp);
        dirname = strtok(NULL, ":");
    }
    return (-1);
}

int myRead(int fildes, void *buf, size_t nbyte)
{
    if (isHPSS)
    {
        return hpss_Read(fildes, buf, nbyte);
    }
    return myRead(fildes, buf, nbyte);
}

int myWrite(int fildes, const void *buf, size_t nbyte)
{
    if (isHPSS)
    {
        return hpss_Write(fildes, (char *)buf, nbyte);
    }
    return myWrite(fildes, buf, nbyte);
}

int myClose(int fildes)
{
    if (isHPSS)
    {
        return hpss_Close(fildes);
    }
    return close(fildes);
}

int main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();

    return 0;
}

Application::Application(int argc, char *argv[])
{
    Mesh = NULL;
    Covise::set_module_description("Read OR Write COVISE Data");
    Covise::add_port(INPUT_PORT, "mesh_in", "coDoText|coDoPoints|coDoUnstructuredGrid|coDoUniformGrid|coDoRectilinearGrid|coDoStructuredGrid|coDoFloat|coDoVec3|coDoFloat|coDoVec3|coDoPolygons|coDoTriangleStrips|DO_Unstructured_V3D_Normals|coDoGeometry|coDoLines", "mesh_in");
    Covise::set_port_required("mesh_in", 0);
    Covise::add_port(OUTPUT_PORT, "mesh", "coDoText|coDoPoints|coDoUnstructuredGrid|coDoUniformGrid|coDoRectilinearGrid|coDoStructuredGrid|coDoFloat|coDoVec3|coDoFloat|coDoVec3|coDoPolygons|coDoTriangleStrips|DO_Unstructured_V3D_Normals|coDoGeometry|coDoLines", "mesh");
    Covise::add_port(PARIN, "grid_path", "Browser", "File path");
    Covise::set_port_default("grid_path", ". *.covise");
    Covise::init(argc, argv);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);

    // if RWCovise section in covise.config contains "NO_MAGIC" line, write no Magic
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
    CoviseConfig *config = new CoviseConfig("RWCovise");
    if (config->get_entry("RWCovise.NO_MAGIC"))
        useMagic = 0;
    else
        useMagic = 1;
    delete config;

// HW-dependent byteswapping setting

#ifdef BYTESWAP
    byte_swap = 1;
#else
    byte_swap = 0;
#endif

    // read input parameters and data object name
    char buf[300];
    coDistributedObject *tmp_obj;

    Covise::get_browser_param("grid_path", &grid_Path);

    Mesh = Covise::get_object_name("mesh");
    Mesh_in = Covise::get_object_name("mesh_in");
    if (Mesh_in != NULL)
    {
        // WRITE DATA

        // if ((fp = open(grid_Path,O_WRONLY|O_CREAT,0660)) <0)
        if ((fp = myOpen(grid_Path, O_WRONLY | O_CREAT)) < 0)
        {
            strcpy(buf, "ERROR: Can't open file >> ");
            strcat(buf, grid_Path);
            Covise::sendError(buf);
            return;
        }
        tmp_obj = new coDistributedObject(Mesh_in);

        // Write MAGIC
        if (useMagic)
        {
            if (byte_swap)
                myWrite(fp, "COV_BE", 6);
            else
                myWrite(fp, "COV_LE", 6);
        }
        // Write the object
        writeobj(tmp_obj);
        delete tmp_obj;
        myClose(fp);
    }
    else
    {
        // READ DATA

        if ((fp = myOpen(grid_Path, O_RDONLY)) < 0)
        {
            strcpy(buf, "ERROR: Can't open file >> ");
            strcat(buf, grid_Path);
            Covise::sendError(buf);
            return;
        }

        tmp_obj = readData(Mesh);
        delete tmp_obj;

        myClose(fp);
    }
}

void Application::writeobj(coDistributedObject *tmp_Object)
{
    coDoSet *set;
    coDoGeometry *geo;
    coDoRGBA *rgba;
    coDoLines *lin;
    coDoPoints *pts;
    coDoText *txt;
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
            myWrite(fp, gtype, 6);
            myWrite(fp, &numsets, sizeof(int));
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
            myWrite(fp, gtype, 6);
            myWrite(fp, &do1, sizeof(int));
            myWrite(fp, &do2, sizeof(int));
            myWrite(fp, &do3, sizeof(int));
            myWrite(fp, &t1, sizeof(int));
            myWrite(fp, &t2, sizeof(int));
            myWrite(fp, &t3, sizeof(int));
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
            myWrite(fp, gtype, 6);
            myWrite(fp, &usg_h, sizeof(usg_h));
            myWrite(fp, el, usg_h.n_elem * sizeof(int));
            myWrite(fp, tl, usg_h.n_elem * sizeof(int));
            myWrite(fp, vl, usg_h.n_conn * sizeof(int));
            myWrite(fp, x_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, y_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, z_coord, usg_h.n_coord * sizeof(float));
            writeattrib(data_obj);
            delete mesh;
        }
        else if (strcmp(gtype, "POINTS") == 0)
        {
            pts = (coDoPoints *)data_obj;
            pts->getAddresses(&x_coord, &y_coord, &z_coord);
            myWrite(fp, gtype, 6);
            n_elem = pts->getNumPoints();
            myWrite(fp, &n_elem, sizeof(int));
            myWrite(fp, x_coord, n_elem * sizeof(float));
            myWrite(fp, y_coord, n_elem * sizeof(float));
            myWrite(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete pts;
        }
        else if (strcmp(gtype, "DOTEXT") == 0)
        {
            char *data;
            txt = (coDoText *)data_obj;
            txt->getAddress(&data);
            myWrite(fp, gtype, 6);
            n_elem = txt->getTextLength();
            myWrite(fp, &n_elem, sizeof(int));
            myWrite(fp, data, n_elem);
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
            myWrite(fp, gtype, 6);
            myWrite(fp, &usg_h, sizeof(usg_h));
            myWrite(fp, el, usg_h.n_elem * sizeof(int));
            myWrite(fp, vl, usg_h.n_conn * sizeof(int));
            myWrite(fp, x_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, y_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, z_coord, usg_h.n_coord * sizeof(float));
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
            myWrite(fp, gtype, 6);
            myWrite(fp, &usg_h, sizeof(usg_h));
            myWrite(fp, el, usg_h.n_elem * sizeof(int));
            myWrite(fp, vl, usg_h.n_conn * sizeof(int));
            myWrite(fp, x_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, y_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, z_coord, usg_h.n_coord * sizeof(float));
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
            myWrite(fp, gtype, 6);
            myWrite(fp, &usg_h, sizeof(usg_h));
            myWrite(fp, el, usg_h.n_elem * sizeof(int));
            myWrite(fp, vl, usg_h.n_conn * sizeof(int));
            myWrite(fp, x_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, y_coord, usg_h.n_coord * sizeof(float));
            myWrite(fp, z_coord, usg_h.n_coord * sizeof(float));
            writeattrib(data_obj);
            delete tri;
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid = (coDoRectilinearGrid *)data_obj;
            rgrid->getAddresses(&x_coord, &y_coord, &z_coord);
            rgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            myWrite(fp, gtype, 6);
            myWrite(fp, &s_h, sizeof(s_h));
            myWrite(fp, x_coord, s_h.xs * sizeof(int));
            myWrite(fp, y_coord, s_h.ys * sizeof(int));
            myWrite(fp, z_coord, s_h.zs * sizeof(int));
            writeattrib(data_obj);
            delete rgrid;
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid = (coDoStructuredGrid *)data_obj;
            sgrid->getAddresses(&x_coord, &y_coord, &z_coord);
            sgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            myWrite(fp, gtype, 6);
            myWrite(fp, &s_h, sizeof(s_h));
            myWrite(fp, x_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
            myWrite(fp, y_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
            myWrite(fp, z_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
            writeattrib(data_obj);
            delete sgrid;
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            float x_min, y_min, z_min, x_max, y_max, z_max;

            ugrid = (coDoUniformGrid *)data_obj;
            ugrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            ugrid->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);

            myWrite(fp, gtype, 6);
            myWrite(fp, &s_h, sizeof(s_h));

            myWrite(fp, &x_min, sizeof(float));
            myWrite(fp, &x_max, sizeof(float));

            myWrite(fp, &y_min, sizeof(float));
            myWrite(fp, &y_max, sizeof(float));

            myWrite(fp, &z_min, sizeof(float));
            myWrite(fp, &z_max, sizeof(float));

            writeattrib(data_obj);
            delete ugrid;
        }
        else if (strcmp(gtype, "USTSDT") == 0)
        {
            us3d = (coDoFloat *)data_obj;
            us3d->getAddress(&z_coord);
            n_elem = us3d->getNumPoints();
            myWrite(fp, gtype, 6);
            myWrite(fp, &n_elem, sizeof(int));
            myWrite(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete us3d;
        }
        else if (strcmp(gtype, "RGBADT") == 0)
        {
            rgba = (coDoRGBA *)data_obj;
            rgba->getAddress((int **)(&z_coord));
            n_elem = rgba->getNumElements();
            myWrite(fp, gtype, 6);
            myWrite(fp, &n_elem, sizeof(int));
            myWrite(fp, z_coord, n_elem * sizeof(int));
            writeattrib(data_obj);
            delete rgba;
        }
        else if (strcmp(gtype, "USTVDT") == 0)
        {
            us3dv = (coDoVec3 *)data_obj;
            us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
            n_elem = us3dv->getNumPoints();
            myWrite(fp, gtype, 6);
            myWrite(fp, &n_elem, sizeof(int));
            myWrite(fp, x_coord, n_elem * sizeof(float));
            myWrite(fp, y_coord, n_elem * sizeof(float));
            myWrite(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete us3dv;
        }
        else if (strcmp(gtype, "STRSDT") == 0)
        {
            s3d = (coDoFloat *)data_obj;
            s3d->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            s3d->getAddress(&z_coord);
            n_elem = s_h.xs * s_h.ys * s_h.zs;
            myWrite(fp, gtype, 6);
            myWrite(fp, &s_h, sizeof(s_h));
            myWrite(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete s3d;
        }
        else if (strcmp(gtype, "STRVDT") == 0)
        {
            s3dv = (coDoVec3 *)data_obj;
            s3dv->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
            s3dv->getAddresses(&x_coord, &y_coord, &z_coord);
            n_elem = s_h.xs * s_h.ys * s_h.zs;
            myWrite(fp, gtype, 6);
            myWrite(fp, &s_h, sizeof(s_h));
            myWrite(fp, x_coord, n_elem * sizeof(float));
            myWrite(fp, y_coord, n_elem * sizeof(float));
            myWrite(fp, z_coord, n_elem * sizeof(float));
            writeattrib(data_obj);
            delete s3dv;
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

inline void swap_int(int &d)
{
    unsigned int &data = (unsigned int &)d;
    data = ((data & 0xff000000) >> 24)
           | ((data & 0x00ff0000) >> 8)
           | ((data & 0x0000ff00) << 8)
           | ((data & 0x000000ff) << 24);
}

inline void swap_int(int *d, int num)
{
    unsigned int *data = (unsigned int *)d;
    int i;
    //fprintf(stderr,"swapping %d integers\n", num);
    for (i = 0; i < num; i++)
    {
        //fprintf(stderr,"data=%d\n", *data);

        *data = (((*data) & 0xff000000) >> 24)
                | (((*data) & 0x00ff0000) >> 8)
                | (((*data) & 0x0000ff00) << 8)
                | (((*data) & 0x000000ff) << 24);
        //fprintf(stderr,"data=%d\n", *data);
        data++;
    }
}

// Not used
//inline void swap_float(float &d)
//{
//   unsigned int &data = (unsigned int &) d;
//   data =    ( (data & 0xff000000) >> 24 )
//           | ( (data & 0x00ff0000) >>  8 )
//           | ( (data & 0x0000ff00) <<  8 )
//           | ( (data & 0x000000ff) << 24 ) ;
//}

inline void swap_float(float *d, int num)
{
    unsigned int *data = (unsigned int *)d;
    int i;
    for (i = 0; i < num; i++)
    {
        *data = (((*data) & 0xff000000) >> 24)
                | (((*data) & 0x00ff0000) >> 8)
                | (((*data) & 0x0000ff00) << 8)
                | (((*data) & 0x000000ff) << 24);
        data++;
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
    myWrite(fp, &size, sizeof(int));
    myWrite(fp, &numattrib, sizeof(int));
    for (i = 0; i < numattrib; i++)
    {
        myWrite(fp, an[i], strlen(an[i]) + 1);
        myWrite(fp, at[i], strlen(at[i]) + 1);
    }
}

void Application::readattrib(coDistributedObject *tmp_Object)
{
    int numattrib = 0, size = 0, i;
    char *an, *at;
    char *buf;

    myRead(fp, &size, sizeof(int));
    if (byte_swap)
        swap_int(size);
    size -= sizeof(int);
    myRead(fp, &numattrib, sizeof(int));
    if (byte_swap)
        swap_int(numattrib);
    if (size > 0)
    {
        buf = new char[size];
        myRead(fp, buf, size);
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
    coDoGeometry *geo;
    coDoRGBA *rgba;
    coDoLines *lin;
    coDoPoints *pts;
    coDoText *txt;
    char buf[300], Data_Type[7];
    USG_HEADER usg_h;
    STR_HEADER s_h;
    coDistributedObject **tmp_objs, *do1, *do2, *do3;
    int numsets, i, t1, t2, t3;

    myRead(fp, Data_Type, 6);
    Data_Type[6] = '\0';

    // MAGIC check

    if (strcmp(Data_Type, "COV_BE") == 0) // The file is big-endian
    {
        byte_swap = 1 - byte_swap;
        myRead(fp, Data_Type, 6); // skip magic
    }
    else if (strcmp(Data_Type, "COV_LE") == 0) // The file is big-endian
        myRead(fp, Data_Type, 6); // skip magic

    if (Mesh != NULL)
    {
        if (strcmp(Data_Type, "SETELE") == 0)
        {
            myRead(fp, &numsets, sizeof(int));
            if (byte_swap)
                swap_int(numsets);

            tmp_objs = new coDistributedObject *[numsets + 1];

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
            delete[] tmp_objs;
            readattrib(set);
            return (set);
        }

        else if (strcmp(Data_Type, "GEOMET") == 0)
        {
            myRead(fp, &do1, sizeof(int));
            myRead(fp, &do2, sizeof(int));
            myRead(fp, &do3, sizeof(int));
            myRead(fp, &t1, sizeof(int));
            if (byte_swap)
                swap_int(t1);
            myRead(fp, &t2, sizeof(int));
            if (byte_swap)
                swap_int(t2);
            myRead(fp, &t3, sizeof(int));
            if (byte_swap)
                swap_int(t3);
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
            if (do2)
            {
                geo->setColor(t2, do2);
            }
            if (do3)
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

            //fprintf(stderr,"found USG\n");
            //if (byte_swap!=0)
            //   fprintf(stderr,"need swap\n");

            myRead(fp, &usg_h, sizeof(usg_h));
            if (byte_swap)
                swap_int((int *)&usg_h, sizeof(usg_h) / sizeof(int));

            //fprintf(stderr,"creating object with %d elements %d connections %d coordinates\n", usg_h.n_elem,usg_h.n_conn, usg_h.n_coord);
            //fprintf(stderr, "creating covise usg...");

            mesh = new coDoUnstructuredGrid(Name, usg_h.n_elem, usg_h.n_conn, usg_h.n_coord, 1);

            //fprintf(stderr, "...done\n");

            if (mesh->objectOk())
            {
                mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
                mesh->getTypeList(&tl);

                myRead(fp, el, usg_h.n_elem * sizeof(int));
                if (byte_swap)
                    swap_int(el, usg_h.n_elem);

                myRead(fp, tl, usg_h.n_elem * sizeof(int));
                if (byte_swap)
                    swap_int(tl, usg_h.n_elem);

                myRead(fp, vl, usg_h.n_conn * sizeof(int));
                if (byte_swap)
                    swap_int(vl, usg_h.n_conn);

                myRead(fp, x_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, usg_h.n_coord);

                myRead(fp, y_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, usg_h.n_coord);

                myRead(fp, z_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, usg_h.n_coord);

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
            myRead(fp, &usg_h, sizeof(usg_h));
            if (byte_swap)
                swap_int((int *)&usg_h, sizeof(usg_h) / sizeof(int));

            pol = new coDoPolygons(Name, usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            if (pol->objectOk())
            {
                pol->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);

                myRead(fp, el, usg_h.n_elem * sizeof(int));
                if (byte_swap)
                    swap_int(el, usg_h.n_elem);

                myRead(fp, vl, usg_h.n_conn * sizeof(int));
                if (byte_swap)
                    swap_int(vl, usg_h.n_conn);

                myRead(fp, x_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, usg_h.n_coord);

                myRead(fp, y_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, usg_h.n_coord);

                myRead(fp, z_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, usg_h.n_coord);

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
            myRead(fp, &n_elem, sizeof(int));
            if (byte_swap)
                swap_int(n_elem);
            pts = new coDoPoints(Name, n_elem);
            if (pts->objectOk())
            {
                pts->getAddresses(&x_coord, &y_coord, &z_coord);
                myRead(fp, x_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, usg_h.n_elem);

                myRead(fp, y_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, usg_h.n_elem);

                myRead(fp, z_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, usg_h.n_elem);

                readattrib(pts);
                return (pts);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "DOTEXT") == 0)
        {
            char *data;
            myRead(fp, &n_elem, sizeof(int));
            if (byte_swap)
                swap_int(n_elem);
            txt = new coDoText(Name, n_elem);
            if (txt->objectOk())
            {
                txt->getAddress(&data);
                myRead(fp, data, n_elem);

                readattrib(txt);
                return (txt);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "LINES") == 0)
        {
            myRead(fp, &usg_h, sizeof(usg_h));
            if (byte_swap)
                swap_int((int *)&usg_h, sizeof(usg_h) / sizeof(int));
            lin = new coDoLines(Name, usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            if (lin->objectOk())
            {
                lin->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);

                myRead(fp, el, usg_h.n_elem * sizeof(int));
                if (byte_swap)
                    swap_int(el, usg_h.n_elem);

                myRead(fp, vl, usg_h.n_conn * sizeof(int));
                if (byte_swap)
                    swap_int(vl, usg_h.n_conn);

                myRead(fp, x_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, usg_h.n_coord);

                myRead(fp, y_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, usg_h.n_coord);

                myRead(fp, z_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, usg_h.n_coord);

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
            myRead(fp, &usg_h, sizeof(usg_h));
            if (byte_swap)
                swap_int((int *)&usg_h, sizeof(usg_h) / sizeof(int));

            tri = new coDoTriangleStrips(Name, usg_h.n_coord, usg_h.n_conn, usg_h.n_elem);
            if (tri->objectOk())
            {
                tri->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
                myRead(fp, el, usg_h.n_elem * sizeof(int));
                if (byte_swap)
                    swap_int(el, usg_h.n_elem);

                myRead(fp, vl, usg_h.n_conn * sizeof(int));
                if (byte_swap)
                    swap_int(vl, usg_h.n_conn);

                myRead(fp, x_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, usg_h.n_coord);

                myRead(fp, y_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, usg_h.n_coord);

                myRead(fp, z_coord, usg_h.n_coord * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, usg_h.n_coord);

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
            myRead(fp, &s_h, sizeof(s_h));
            if (byte_swap)
                swap_int((int *)&s_h, sizeof(s_h) / sizeof(int));

            rgrid = new coDoRectilinearGrid(Name, s_h.xs, s_h.ys, s_h.zs);
            if (rgrid->objectOk())
            {
                rgrid->getAddresses(&x_coord, &y_coord, &z_coord);
                myRead(fp, x_coord, s_h.xs * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, s_h.xs);

                myRead(fp, y_coord, s_h.ys * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, s_h.ys);

                myRead(fp, z_coord, s_h.zs * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, s_h.zs);

                readattrib(rgrid);
                return (rgrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of RCTGRD object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "UNIGRD") == 0)
        {
            float x_min, y_min, z_min, x_max, y_max, z_max;

            myRead(fp, &s_h, sizeof(s_h));
            if (byte_swap)
                swap_int((int *)&s_h, sizeof(s_h) / sizeof(int));

            myRead(fp, &x_min, sizeof(float));
            if (byte_swap)
                swap_float(&x_min, 1);
            myRead(fp, &x_max, sizeof(float));
            if (byte_swap)
                swap_float(&x_max, 1);

            myRead(fp, &y_min, sizeof(float));
            if (byte_swap)
                swap_float(&y_min, 1);
            myRead(fp, &y_max, sizeof(float));
            if (byte_swap)
                swap_float(&y_max, 1);

            myRead(fp, &z_min, sizeof(float));
            if (byte_swap)
                swap_float(&z_min, 1);
            myRead(fp, &z_max, sizeof(float));
            if (byte_swap)
                swap_float(&z_max, 1);

            ugrid = new coDoUniformGrid(Name, s_h.xs, s_h.ys, s_h.zs,
                                        x_min, x_max, y_min, y_max, z_min, z_max);

            if (ugrid->objectOk())
            {
                readattrib(ugrid);
                return (ugrid);
            }
            else
            {
                Covise::sendError("ERROR: creation of UNIGRID object failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "STRGRD") == 0)
        {
            myRead(fp, &s_h, sizeof(s_h));
            if (byte_swap)
                swap_int((int *)&s_h, sizeof(s_h) / sizeof(int));

            sgrid = new coDoStructuredGrid(Name, s_h.xs, s_h.ys, s_h.zs);
            if (sgrid->objectOk())
            {
                sgrid->getAddresses(&x_coord, &y_coord, &z_coord);

                myRead(fp, x_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, s_h.xs * s_h.ys * s_h.zs);

                myRead(fp, y_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, s_h.xs * s_h.ys * s_h.zs);

                myRead(fp, z_coord, s_h.xs * s_h.ys * s_h.zs * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, s_h.xs * s_h.ys * s_h.zs);

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
            myRead(fp, &n_elem, sizeof(int));
            if (byte_swap)
                swap_int(n_elem);
            us3d = new coDoFloat(Name, n_elem);
            if (us3d->objectOk())
            {
                us3d->getAddress(&x_coord);

                myRead(fp, x_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, n_elem);

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
            myRead(fp, &n_elem, sizeof(int));
            if (byte_swap)
                swap_int(n_elem);

            rgba = new coDoRGBA(Name, n_elem);
            if (rgba->objectOk())
            {
                rgba->getAddress((int **)(&x_coord));
                myRead(fp, x_coord, n_elem * sizeof(int));
                if (byte_swap)
                    swap_int((int *)x_coord, n_elem);

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
            myRead(fp, &n_elem, sizeof(int));
            if (byte_swap)
                swap_int(n_elem);

            us3dv = new coDoVec3(Name, n_elem);
            if (us3dv->objectOk())
            {
                us3dv->getAddresses(&x_coord, &y_coord, &z_coord);
                myRead(fp, x_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, n_elem);

                myRead(fp, y_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, n_elem);

                myRead(fp, z_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, n_elem);

                readattrib(us3dv);
                return (us3dv);
            }
            else
            {
                Covise::sendError("ERROR: creation of USTVDT object 'mesh' failed");
                return (NULL);
            }
        }

        else if (strcmp(Data_Type, "STRSDT") == 0)
        {
            myRead(fp, &s_h, sizeof(s_h));
            if (byte_swap)
                swap_int((int *)&s_h, sizeof(s_h) / sizeof(int));

            n_elem = s_h.xs * s_h.ys * s_h.zs;
            s3d = new coDoFloat(Name, s_h.xs, s_h.ys, s_h.zs);
            if (s3d->objectOk())
            {
                s3d->getAddress(&x_coord);
                myRead(fp, x_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, n_elem);

                readattrib(s3d);
                return (s3d);
            }
            else
            {
                Covise::sendError("ERROR: creation of STRSDT object 'mesh' failed");
                return (NULL);
            }
        }
        else if (strcmp(Data_Type, "STRVDT") == 0)
        {
            myRead(fp, &s_h, sizeof(s_h));
            if (byte_swap)
                swap_int((int *)&s_h, sizeof(s_h) / sizeof(int));

            n_elem = s_h.xs * s_h.ys * s_h.zs;
            s3dv = new coDoVec3(Name, s_h.xs, s_h.ys, s_h.zs);

            if (s3dv->objectOk())
            {
                s3dv->getAddresses(&x_coord, &y_coord, &z_coord);

                myRead(fp, x_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(x_coord, n_elem);

                myRead(fp, y_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(y_coord, n_elem);

                myRead(fp, z_coord, n_elem * sizeof(float));
                if (byte_swap)
                    swap_float(z_coord, n_elem);

                readattrib(s3dv);
                return (s3dv);
            }
            else
            {
                Covise::sendError("ERROR: creation of STRVDT object 'mesh' failed");
                return (NULL);
            }
        }
        else
        {
            strcpy(buf, "ERROR: Reading file '");
            strcat(buf, grid_Path);
            strcat(buf, "', File does not seem to be in Covise Format");
            Covise::sendError(buf);
            return (NULL);
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct ");
        return (NULL);
    }
    return (NULL);
}
