/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: object manager class for COVISE renderer modules          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

#include <appl/RenderInterface.h>
#include "ObjectManager.h"
#include <util/coMaterial.h>

#include <do/coDoSet.h>
#include <do/coDoGeometry.h>
#include <do/coDoText.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <do/coDoSpheres.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoLines.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoUniformGrid.h>

#include <appl/RenderInterface.h>

/// whatever... copied from IVRenderer
coMaterialList materialList("metal");

#define INV_PER_VERTEX 0
#define INV_PER_FACE 1
#define INV_NONE 2
#define INV_OVERALL 3
#define INV_RGBA 4
#define INV_TEXTURE 5

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
//================================================================
// ObjectManager methods
//================================================================

ObjectManager::ObjectManager()
    : filename(NULL)
{
    anzset = 0;
    list = new ObjectList();
    gm = new GeometryManager();
    file_writing = 1;
};

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::addObject(char *object)
{
    const coDistributedObject *dobj = 0L;
    const coDistributedObject *colors = 0L;
    const coDistributedObject *normals = 0L;
    const coDoGeometry *geometry = 0L;
    const coDoText *Text = 0L;
    const coDistributedObject *data_obj;
    char *IvData;
    const char *gtype;
    int is_timestep = 0;

    //cerr << endl << "Start ObjectManager::addObject -> " << object << endl;

    data_obj = coDistributedObject::createFromShm(object);
    if (data_obj != 0L)
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "GEOMET") == 0)
        {
            geometry = (coDoGeometry *)data_obj;
            if (geometry->objectOk())
            {
                dobj = geometry->getGeometry();
                normals = geometry->getNormals();
                colors = geometry->getColors();
                gtype = dobj->getType();
                add_geometry(object, is_timestep, NULL, dobj, normals, colors, geometry);
            }
            delete geometry;
        }
        else if (strcmp(gtype, "DOTEXT") == 0)
        {
            Text = (const coDoText *)data_obj;
            Text->getAddress(&IvData);
            // gm->addIv(object,NULL,IvData,Text->getTextLength());
        }
        else
        {
            add_geometry(object, is_timestep, NULL, data_obj, NULL, NULL, NULL);
        }
    }

    // writing a wrml file
    if (file_writing && filename)
    {
        FILE *fp;
        if (strcmp("Web Connection", filename) == 0)
        {
            fp = fopen("out.wrl", "w");
        }
        else
            fp = fopen(filename, "w");

        if (fp)
        {
            fprintf(fp, "#VRML V2.0 utf8\n\n");
            objlist->write(fp);
            fclose(fp);
        }
        else
            CoviseRender::sendError("%s", strerror(errno));
    }

    //cerr << endl << "End ObjectManager::addObject -> " << object << endl;
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::deleteObject(char *name)
{

    int i, n;

    //printf("ObjectManager::deleteObject\n");
    //printf("\t object = %s\n", name);

    //cerr << endl << "Start ObjectManager::deleteObject -> " << name;

    for (i = 0; i < anzset; i++)
    {
        if (strcmp(setnames[i], name) == 0)
        {
            for (n = 0; n < elemanz[i]; n++)
            {
                deleteObject(elemnames[i][n]);
                delete elemnames[i][n];
            }
            delete elemnames[i];
            n = i;
            anzset--;
            while (n < (anzset))
            {
                elemanz[n] = elemanz[n + 1];
                elemnames[n] = elemnames[n + 1];
                setnames[n] = setnames[n + 1];
                n++;
            }
        }
    }

    remove_geometry(name);

    // writing a wrml file
    if (file_writing && filename)
    {
        FILE *fp;
        if (strcmp("Web Connection", filename) == 0)
            fp = fopen("out.wrl", "w");
        else
            fp = fopen(filename, "w");
        if (fp)
        {
            fprintf(fp, "#VRML V2.0 utf8\n\n");
            objlist->write(fp);
            fclose(fp);
        }
        else
            CoviseRender::sendError("%s", strerror(errno));
    }
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::remove_geometry(char *name)
{

    //cerr << endl << "ObjectManager::remove_geometry -> " << name;
    gm->remove_geometry(name);
}

//======================================================================
// create a color data object for a named color
//======================================================================
FILE *fp = NULL;
int isopen = 0;
static unsigned int create_named_color(const char *cname)
{
    int r = 255, g = 255, b = 255;
    unsigned int rgba;
    char line[80];
    char *tp, *token[15];
    unsigned char *chptr;
    int count;
    const int tmax = 15;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        count = 0;
        tp = strtok(line, " \t");
        for (count = 0; count < tmax && tp != NULL;)
        {
            token[count] = tp;
            tp = strtok(NULL, " \t");
            count++;
        }
        token[count] = NULL;
        if (count == 5)
        {
            strcat(token[3], " ");
            strcat(token[3], token[4]);
        }
        if (strstr(token[3], cname) != NULL)
        {
            r = atoi(token[0]);
            g = atoi(token[1]);
            b = atoi(token[2]);
            fseek(fp, 0L, SEEK_SET);
            break;
        }
    }
    fseek(fp, 0L, SEEK_SET);

    chptr = (unsigned char *)&rgba;
#ifdef BYTESWAP
    *chptr = (unsigned char)(255);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(r); // no transparency
#else
    *chptr = (unsigned char)(r);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(255); // no transparency
#endif
    return rgba;
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::add_geometry(const char *object, int is_timestep, const char *root, const coDistributedObject *geometry,
                                 const coDistributedObject *normals, const coDistributedObject *colors, const coDoGeometry *container)
{
    const coDistributedObject *const *dobjsg = NULL; // Geometry Set elements
    const coDistributedObject *const *dobjsc = NULL; // Color Set elements
    const coDistributedObject *const *dobjsn = NULL; // Normal Set elements
    const coDoVec3 *normal_udata = NULL;
    const coDoVec3 *color_udata = NULL;
    const coDoRGBA *color_pdata = NULL;
    const coDoPoints *points = NULL;
    const coDoSpheres *spheres = NULL;
    const coDoLines *lines = NULL;
    const coDoPolygons *poly = NULL;
    const coDoTriangleStrips *strip = NULL;
    const coDoUniformGrid *ugrid = NULL;
    const coDoRectilinearGrid *rgrid = NULL;
    const coDoStructuredGrid *sgrid = NULL;
    const coDoUnstructuredGrid *unsgrid = NULL;
    const coDoSet *set = NULL;
    // number of elements per geometry,color and normal set
    int no_elems = 0, no_c = 0, no_n = 0;
    int normalbinding = CO_NONE, colorbinding = CO_NONE;
    int colorpacking = CO_NONE;
    int vertexOrder = 0;
    int no_poly = 0;
    int no_strip = 0;
    int no_vert = 0;
    int no_points = 0;
    int no_lines = 0;
    int curset;
    int *v_l = 0, *l_l = 0, *el = 0, *vl = 0;
    int xsize, ysize, zsize;
    int minTimeStep = 0, maxTimeStep = -1;
    float xmax = 0, xmin = 0, ymax = 0, ymin = 0, zmax = 0, zmin = 0;
    float *rc = 0, *gc = 0, *bc = 0, *xn = 0, *yn = 0, *zn = 0;
    int *pc = NULL;
    float *x_c = 0, *y_c = 0, *z_c = 0;
    float *r_c = NULL;
    float transparency = 0.0;
    const char *gtype, *ntype, *ctype;
    const char *vertexOrderStr, *transparencyStr;
    const char *bindingType, *objName;
    char buf[300];
    const char *tstep_attrib = 0L;
    const char *feedback_info;
    is_timestep = 0;
    int i;
    unsigned int rgba;
    curset = anzset;
    coMaterial *material = NULL;

    gtype = geometry->getType();

    if (strcmp(gtype, "SETELE") == 0)
    {
        set = (coDoSet *)geometry;
        if (set != NULL)
        {
            // retrieve the whole set
            dobjsg = set->getAllElements(&no_elems);

            // look if it is a timestep series
            tstep_attrib = set->getAttribute("TIMESTEP");
            if (tstep_attrib != NULL)
            {
                //sscanf(tstep_attrib,"%d %d",&minTimeStep,&maxTimeStep);
                minTimeStep = 0;
                maxTimeStep = -1;
                //fprintf(stderr,"Found Attrib\n");
                is_timestep = 1;
            }

            // Uwe Woessner
            feedback_info = set->getAttribute("FEEDBACK");
            if (feedback_info)
            {
                this->addFeedbackButton(object, feedback_info);
            }
            if (normals != NULL)
            {
                ntype = normals->getType();
                if (strcmp(ntype, "SETELE") != 0)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: ...did not get a normal set");
                }
                else
                {
                    set = (coDoSet *)normals;
                    if (set != NULL)
                    {
                        // Get Set
                        dobjsn = set->getAllElements(&no_n);
                        if (no_n == no_elems)
                        {
                            print_comment(__LINE__, __FILE__, "... got normal set");
                        }
                        else
                        {
                            print_comment(__LINE__, __FILE__, "ERROR: number of normalelements does not match geometry set");
                            no_n = 0;
                        }
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: ...got bad normal set");
                    }
                }
            }
            if (colors != NULL)
            {
                ctype = colors->getType();
                if (strcmp(ctype, "SETELE") != 0)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: ...did not get a color set");
                }
                else
                {
                    set = (coDoSet *)colors;
                    if (set != NULL)
                    {
                        // Get Set
                        dobjsc = set->getAllElements(&no_c);
                        if (no_c == no_elems)
                        {
                            print_comment(__LINE__, __FILE__, "... got color set");
                        }
                        else
                        {
                            print_comment(__LINE__, __FILE__, "ERROR: number of colorelements does not match geometry set");
                            no_c = 0;
                        }
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: ...got bad color set");
                    }
                }
            }
            setnames[curset] = new char[strlen(object) + 1];
            strcpy(setnames[curset], object);
            elemanz[curset] = no_elems;
            elemnames[curset] = new char *[no_elems];
            gm->addGroup(object, root, is_timestep, minTimeStep, maxTimeStep);
            anzset++;
            if (is_timestep)
            {
                LObject *lob = new LObject("BeginTimeset", object, (CharBuffer *)NULL);
                lob->m_is_timestep = 1;
                lob->set_minmax(0, no_elems - 1);
                lob->set_real_root(root);
                objlist->append(lob);
                objlist->incr_no_sw();
            }
            else
            {
                LObject *lob = new LObject("Beginset", object, (CharBuffer *)NULL);
                lob->set_minmax(0, no_elems - 1);
                lob->set_real_root(root);
                lob->set_timestep(object);
                objlist->append(lob);
            }
            for (i = 0; i < no_elems; i++)
            {
                objName = dobjsg[i]->getAttribute("OBJECTNAME");
                if (objName == NULL)
                {
                    strcpy(buf, dobjsg[i]->getName());
                    objName = buf;
                }
                elemnames[curset][i] = new char[strlen(objName) + 1];
                strcpy(elemnames[curset][i], objName);
                if ((no_c > 0) && (no_n > 0))
                {
                    //cerr << "\nAdding (recursively) set geometry named " << object << " now"  << endl;
                    add_geometry(objName, is_timestep, object, dobjsg[i], dobjsn[i], dobjsc[i], container);
                }
                else if (no_c > 0)
                {
                    //cerr << "\nAdding (recursively) set geometry named " << object << " now"  << endl;
                    add_geometry(objName, is_timestep, object, dobjsg[i], NULL, dobjsc[i], container);
                }
                else if (no_n > 0)
                {
                    //cerr << "\nAdding (recursively) set geometry named " << object << " now"  << endl;
                    add_geometry(objName, is_timestep, object, dobjsg[i], dobjsn[i], NULL, container);
                }
                else
                {
                    //cerr << "\nAdding (recursively) set geometry named " << object << " now"  << endl;
                    add_geometry(objName, is_timestep, object, dobjsg[i], NULL, NULL, container);
                }
            }
            LObject *lob = new LObject("Endset", object, (CharBuffer *)NULL);
            objlist->append(lob);
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: ...got bad geometry set");
        }
    }

    /////////// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    else // not a set
    {
        /*****
         // check for Colorbinding attributes
             colorbinding=CO_NONE;
         if(colors != NULL)
         {
             bindingType=colors->getAttribute("COLOR_BINDING");

             if(bindingType==NULL) {
                      if (container->get_color_attr() == OVERALL)
                   colorbinding=CO_OVERALL;
                 else if (container->get_color_attr() == PER_VERTEX)
      colorbinding=CO_PER_VERTEX;
      else if (container->get_color_attr() == PER_FACE)
      colorbinding=CO_PER_FACE;
      }
      else if(strcmp(bindingType,"OVERALL")==NULL)
      colorbinding=CO_OVERALL;
      else if(strcmp(bindingType,"PER_VERTEX")==NULL)
      colorbinding=CO_PER_VERTEX;
      else if(strcmp(bindingType,"PER_FACE")==NULL)
      colorbinding=CO_PER_FACE;
      }

      // check for Normalbinding
      normalbinding = CO_NONE;
      if(normals != NULL)
      {

      bindingType=normals->getAttribute("NORMAL_BINDING");
      if(bindingType==NULL) {
      if (container->get_normal_attr() == OVERALL)
      normalbinding=CO_OVERALL;
      else if (container->get_normal_attr() == PER_VERTEX)
      normalbinding=CO_PER_VERTEX;
      else if (container->get_normal_attr() == PER_FACE)
      normalbinding=CO_PER_FACE;
      }
      else if(strcmp(bindingType,"PER_VERTEX")==NULL)
      normalbinding=CO_PER_VERTEX;
      else if(strcmp(bindingType,"PER_FACE")==NULL)
      normalbinding=CO_PER_FACE;
      else if(strcmp(bindingType,"OVERALL")==NULL)
      normalbinding=CO_OVERALL;
      }
      *****/

        feedback_info = geometry->getAttribute("FEEDBACK");
        if (feedback_info)
        {
            this->addFeedbackButton(object, feedback_info);
        }

        // check for VertexOrderStr
        vertexOrderStr = geometry->getAttribute("vertexOrder");
        if (vertexOrderStr == NULL)
            vertexOrder = 0;
        else
            vertexOrder = vertexOrderStr[0] - '0';
        /*
         // check for Transparency
         transparencyStr=geometry->getAttribute("TRANSPARENCY");
         transparency=0.0;
         if(transparencyStr!=NULL)
         {
             if((transparency=atof(transparencyStr))<0.0)
            transparency=0.0;
             if(transparency > 1.0)
            transparency=1.0;
         }

      if(normalbinding!=CO_NONE)
      {
      normal_data = (coDoVec3 *)normals;
      no_n = normal_data->getNumPoints();
      normal_data->getAddresses(&xn,&yn,&zn);
      }
      */
        if (colorbinding != CO_NONE)
        {

            ctype = colors->getType();
            if (strcmp(ctype, "USTVDT") == 0)
            {
                color_udata = (coDoVec3 *)colors;
                no_c = color_udata->getNumPoints();
                color_udata->getAddresses(&rc, &gc, &bc);
                colorpacking = CO_NONE;
            }
            else if (strcmp(ctype, "RGBADT") == 0)
            {
                color_pdata = (coDoRGBA *)colors;
                no_c = color_pdata->getNumPoints();
                color_pdata->getAddress(&pc);
                colorpacking = CO_RGBA;
            }
            else
            {
                colorbinding = CO_NONE;
                colorpacking = CO_NONE;
                print_comment(__LINE__, __FILE__, "ERROR: DataTypes other than structured and unstructured are not jet implemented");
                //   sendError("ERROR: DataTypes other than structured and unstructured are not jet implemented");
            }
            if (no_c == 0)
            {
                // sendWarning("WARNING: Data object 'Color' is empty");
                print_comment(__LINE__, __FILE__, "WARNING: Data object 'Color' is empty");
                colorbinding = CO_NONE;
                colorpacking = CO_NONE;
            }
        }
        //else if(colors==NULL)
        else if (no_c == 0)
        {
            bindingType = geometry->getAttribute("COLOR");
            if (bindingType != NULL)
            {
                colorbinding = CO_OVERALL;
                colorpacking = CO_RGBA;
                no_c = 1;
                // open ascii file for color names
                if (!isopen)
                {
                    fp = covise::CoviseRender::fopen("share/covise/rgb.txt", "r");
                    if (fp != 0L)
                        isopen = 1;
                }
                if (isopen)
                {
                    rgba = create_named_color(bindingType);
                    pc = (int *)&rgba;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        //// Now the geometrical primitives

        if (strcmp(gtype, "POLYGN") == 0)
        {
            poly = (coDoPolygons *)geometry;
            if (poly != NULL)
            {
                no_poly = poly->getNumPolygons();
                no_vert = poly->getNumVertices();
                no_points = poly->getNumPoints();
                poly->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
                print_comment(__LINE__, __FILE__, "... got polygons");
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad polygons");
                return;
            }
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            strip = (coDoTriangleStrips *)geometry;
            if (strip != NULL)
            {
                no_strip = strip->getNumStrips();
                no_vert = strip->getNumVertices();
                no_points = strip->getNumPoints();
                strip->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
                print_comment(__LINE__, __FILE__, "... got strips");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad strips");
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            ugrid = (coDoUniformGrid *)geometry;
            if (ugrid != NULL)
            {
                ugrid->getGridSize(&xsize, &ysize, &zsize);
                ugrid->getMinMax(&xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
                print_comment(__LINE__, __FILE__, "... got uniform grid");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad uniform grid");
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            unsgrid = (coDoUnstructuredGrid *)geometry;
            if (unsgrid != NULL)
            {
                unsgrid->getGridSize(&xsize, &ysize, &no_points);
                unsgrid->getAddresses(&el, &vl, &x_c, &y_c, &z_c);
                print_comment(__LINE__, __FILE__, "... got uniform grid");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad unstructured grid");
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid = (coDoRectilinearGrid *)geometry;
            if (rgrid != NULL)
            {
                rgrid->getGridSize(&xsize, &ysize, &zsize);
                rgrid->getAddresses(&x_c, &y_c, &z_c);
                print_comment(__LINE__, __FILE__, "... got rectilinear grid");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad rectilinear grid");
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid = (coDoStructuredGrid *)geometry;
            if (sgrid != NULL)
            {
                sgrid->getGridSize(&xsize, &ysize, &zsize);
                sgrid->getAddresses(&x_c, &y_c, &z_c);
                print_comment(__LINE__, __FILE__, "... got structured grid");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad structured grid");
        }
        else if (strcmp(gtype, "POINTS") == 0)
        {
            points = (coDoPoints *)geometry;
            if (points != NULL)
            {
                no_points = points->getNumPoints();
                points->getAddresses(&x_c, &y_c, &z_c);
                print_comment(__LINE__, __FILE__, "... got points");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad points");
        }

        else if (strcmp(gtype, "SPHERE") == 0)
        {
            spheres = (coDoSpheres *)geometry;
            if (spheres != NULL)
            {
                no_points = spheres->getNumSpheres();
                spheres->getAddresses(&x_c, &y_c, &z_c, &r_c);
                print_comment(__LINE__, __FILE__, "... got points");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad points");
        }

        else if (strcmp(gtype, "LINES") == 0)
        {
            lines = (coDoLines *)geometry;
            if (lines != NULL)
            {
                no_lines = lines->getNumLines();
                no_vert = lines->getNumVertices();
                no_points = lines->getNumPoints();
                lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
                print_comment(__LINE__, __FILE__, "...got lines");
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR: ...got bad lines");
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            print_comment(__LINE__, __FILE__, "WARNING: We should never get here");
            return;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: ...got unknown geometry");
            return;
        }

        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        ////// awe> copied parts from InvRenderer start

        ////////////////////////////////////
        // check for Transparency
        transparencyStr = geometry->getAttribute("TRANSPARENCY");
        transparency = 0.0;
        if (transparencyStr != 0)
        {
            fprintf(stderr, "Transparency: \"%s\"\n", transparencyStr);
            if ((transparency = (float)atof(transparencyStr)) < 0.0f)
                transparency = 0.0f;
            if (transparency > 1.0f)
                transparency = 1.0f;
        }

        ////////////////////////////////////
        /// check for Material
        const char *materialStr = geometry->getAttribute("MATERIAL");
        if (materialStr != NULL)
        {
            if (strncmp(materialStr, "MAT:", 4) == 0)
            {
                char *material_name;
                float ambientColor[3];
                float diffuseColor[3];
                float specularColor[3];
                float emissiveColor[3];
                float shininess;
                float transparency;

                material_name = new char[strlen(materialStr) + 1];
                int retval;
                retval = sscanf(materialStr, "%s %s %f%f%f%f%f%f%f%f%f%f%f%f%f%f",
                                material_name, material_name,
                                &ambientColor[0], &ambientColor[1], &ambientColor[2],
                                &diffuseColor[0], &diffuseColor[1], &diffuseColor[2],
                                &specularColor[0], &specularColor[1], &specularColor[2],
                                &emissiveColor[0], &emissiveColor[1], &emissiveColor[2],
                                &shininess, &transparency);
                if (retval != 16)
                {
                    std::cerr << "ObjectManager::add_geometry: sscanf failed" << std::endl;
                    return;
                }

                material = new coMaterial(material_name, ambientColor, diffuseColor, specularColor, emissiveColor, shininess, transparency);
                delete[] material_name;
            }
            else
            {
                material = materialList.get(materialStr);
                if (!material)
                {
                    char category[500];
                    int retval;
                    retval = sscanf(materialStr, "%s", category);
                    if (retval != 2)
                    {
                        std::cerr << "ObjectManager::add_geometry: sscanf failed" << std::endl;
                        return;
                    }
                    materialList.add(category);
                    material = materialList.get(materialStr);
                    if (!material)
                    {
                        fprintf(stderr, "Material %s not found!\n", materialStr);
                    }
                }
            }
        }

        ////////////////////////////////////
        /// Normals
        //if (normalbinding!=INV_NONE)
        if (normals)
        {
            ntype = normals->getType();

            // Normals in coDoVec3
            if (strcmp(ntype, "USTVDT") == 0)
            {
                normal_udata = (coDoVec3 *)normals;
                no_n = normal_udata->getNumPoints();
                normal_udata->getAddresses(&xn, &yn, &zn);
            }

            // Normals in unknown format
            else
                no_n = 0;

            /// now get this attribute junk done
            if (no_n == no_vert)
                normalbinding = PER_VERTEX;
            else if (no_n > 1)
                normalbinding = PER_FACE;
            else if (no_n == 1)
                normalbinding = OVERALL;
            else
                normalbinding = INV_NONE;
        }
        else
            normalbinding = INV_NONE;

        ////////////////////////////////////
        /// Colors
        no_c = 0;
        if (colors) /// Colors via 'normal' coloring
        {
            ctype = colors->getType();
            if (strcmp(ctype, "USTVDT") == 0)
            {
                color_udata = (coDoVec3 *)colors;
                no_c = color_udata->getNumPoints();
                color_udata->getAddresses(&rc, &gc, &bc);
                colorpacking = INV_NONE;
            }
            else if (strcmp(ctype, "RGBADT") == 0)
            {
                color_pdata = (coDoRGBA *)colors;
                no_c = color_pdata->getNumPoints();
#ifdef _IntIs64Bit
                ALLOC memory and copy buffers
#else
                color_pdata->getAddress((int **)&pc); // sgi64 has 32bit Integers
#endif
                    colorpacking = INV_RGBA;
            }
            else
            {
                colorbinding = INV_NONE;
                colorpacking = INV_NONE;
                print_comment(__LINE__, __FILE__, "ERROR: DataTypes other than structured and unstructured are not jet implemented");
                //   sendError("ERROR: DataTypes other than structured and unstructured are not jet implemented");
            }

            /// now get this attribute junk done
            if (no_c == no_points)
                colorbinding = INV_PER_VERTEX;
            else if (no_c > 1)
                colorbinding = INV_PER_FACE;
            else if (no_c == 1)
                colorbinding = INV_OVERALL;
            else
                colorbinding = INV_NONE;
        }
        else
        {
            colorbinding = INV_NONE;
            colorpacking = INV_NONE;
            print_comment(__LINE__, __FILE__, "ERROR: DataTypes other than structured and unstructured are not jet implemented");
            //   sendError("ERROR: DataTypes other than structured and unstructured are not jet implemented");
        }

        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        ////// awe> copied parts from InvRenderer ends here
        //
        // add object to scenegraph depending on type
        //
        if (strcmp(gtype, "UNIGRD") == 0)
            gm->addUGrid(object, root, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax,
                         no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                         no_n, normalbinding, xn, yn, zn, transparency, material);
        if (strcmp(gtype, "RCTGRD") == 0)
            gm->addRGrid(object, root, xsize, ysize, zsize, x_c, y_c, z_c,
                         no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                         no_n, normalbinding, xn, yn, zn, transparency, material);
        if (strcmp(gtype, "STRGRD") == 0)
            gm->addSGrid(object, root, xsize, ysize, zsize, x_c, y_c, z_c,
                         no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                         no_n, normalbinding, xn, yn, zn, transparency, material);
        if (strcmp(gtype, "POLYGN") == 0)
            gm->addPolygon(object, root, no_poly, no_vert,
                           no_points, x_c, y_c, z_c,
                           v_l, l_l,
                           no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                           no_n, normalbinding, xn, yn, zn, transparency,
                           vertexOrder, material);
        if (strcmp(gtype, "TRIANG") == 0)
            gm->addTriangleStrip(object, root, no_strip, no_vert,
                                 no_points, x_c, y_c, z_c,
                                 v_l, l_l,
                                 no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                 no_n, normalbinding, xn, yn, zn, transparency,
                                 vertexOrder, material);

        if (strcmp(gtype, "LINES") == 0)
        {
            // feedback_info + line  -> trace
            int isTrace = FALSE;
            if (feedback_info)
                isTrace = TRUE;

            gm->addLine(object, root, no_lines, no_vert, no_points,
                        x_c, y_c, z_c, v_l, l_l, no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                        no_n, normalbinding, xn, yn, zn, isTrace, material);
        }
        if ((strcmp(gtype, "POINTS") == 0) || (strcmp(gtype, "UNSGRD") == 0))
            gm->addPoint(object, root, no_points,
                         x_c, y_c, z_c, colorbinding, colorpacking, rc, gc, bc, pc, material);

        if (strcmp(gtype, "SPHERE") == 0)
            gm->addSphere(object, root, no_points,
                          x_c, y_c, z_c, r_c, colorbinding, colorpacking, rc, gc, bc, pc, material);
    }
}

/*______________________________________________________________________*/
void
ObjectManager::addFeedbackButton(const char *object, const char *feedback_info)
{
    //buttonSpecCell spec;
    int i;
    char tmp[200];
    (void)object;

    strcpy(tmp, feedback_info + 1);
    for (i = 0; i < strlen(tmp); i++)
    {
        if (tmp[i] == '\n')
        {
            tmp[i] = ' ';
            break;
        }
    }
    for (; i < strlen(tmp); i++)
    {
        if (tmp[i] == '\n')
        {
            tmp[i] = '\0';
            break;
        }
    }

    /*strcpy(spec.name, tmp);
   spec.actionType= BUTTON_SWITCH;
   spec.callback= &feedbackCallback;
   spec.calledClass= (void*)this;
   spec.state= 0.0;
   spec.dashed= FALSE;
   spec.userData= (void*) new char[strlen(feedback_info)+1];
   spec.group= 0;
   strcpy((char*)spec.userData, feedback_info);
   ((VRPinboard*)pinboard)->addButtonToMainMenu(&spec);*/
}

/*
void
ObjectManager::feedback(buttonSpecCell* spec)
{
        CoviseRender::set_feedback_info((char*)spec->userData);

   switch (CoviseRender::get_feedback_type())
   {
       case 'C':
      strcpy(this->currentFeedbackInfo, (char*)spec->userData);
      if (spec->state)
{
this->c_feedback= TRUE;
sceneGraph->setData(HAND_TYPE, (void*)HAND_PLANE);
}
else
{
this->c_feedback= FALSE;
sceneGraph->setData(HAND_TYPE, (void*)HAND_LINE);
}
break;

case 'T':
strcpy(this->currentFeedbackInfo, (char*)spec->userData);
if (spec->state)
{
this->t_feedback= TRUE;
sceneGraph->setData(HAND_TYPE, (void*)HAND_SPHERE);
}
else
{
this->t_feedback= FALSE;
sceneGraph->setData(HAND_TYPE, (void*)HAND_LINE);
}
break;

case 'I':
strcpy(this->currentFeedbackInfo, (char*)spec->userData);
if (spec->state)
{
this->i_feedback= TRUE;
sceneGraph->setData(HAND_TYPE, (void*)HAND_SPHERE);
}
else
{
this->i_feedback= FALSE;
sceneGraph->setData(HAND_TYPE, (void*)HAND_LINE);
}
break;

default:
printf("button %s unknown covise feedback!\n", spec->name);
break;
}
}

void
ObjectManager::feedbackCallback(void* objectManager, buttonSpecCell* spec)
{
((ObjectManager*)objectManager)->feedback(spec);
}*/

/*______________________________________________________________________*/
void
ObjectManager::set_write_file(int write_mode)
{
    file_writing = write_mode;
}
