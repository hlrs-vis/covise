/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: Write Polygondata to files (VRML, STL, ***)               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Bruno Burbaum                               **
 **                                 HLRS                                   **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  08.10.08  V0.1                                                  **
\**************************************************************************/

#undef VERBOSE

#include "WritePolygon.h"
#include <ctype.h>
#include <util/coviseCompat.h>
#include <util/coVector.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoGeometry.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// constructor
WritePolygon::WritePolygon(int argc, char *argv[])
    : coModule(argc, argv, "Write Stl or VRML97 files")
{

    // input port
    p_dataIn = addInputPort("DataIn",
                            "Geometry|IntArr|Polygons|Lines|Float|Vec3|UniformGrid|RectilinearGrid|TriangleStrips|StructuredGrid|UnstructuredGrid|Points|Vec3|Float|RGBA|USR_DistFenflossBoco",
                            "InputObjects");
    p_dataIn->setRequired(0);

    // select file name from browser
    p_filename = addFileBrowserParam("path", "Output File");
    p_filename->setValue(".", "*");
	
	p_fileFormat = addChoiceParam("type","File type");
    
    const char *formatChoices[] = { "Vrml97", "stl" , "stl_triangles"};
    p_fileFormat->setValue(3, formatChoices, 0);

    p_newFile = addBooleanParam("new", "Create new file");
	p_newFile->setInfo("Truncate file or append to existing file");
    p_newFile->setValue(true);
}

// write an objet to file
void
WritePolygon::writeObj(const char *offset, const coDistributedObject *new_data)
{
    // get object type
    const char *type = new_data->getType();
    char *sp = (char *)"   ";
    int i, j, k, l;

    // switch object types
    // Uwe:  Was ist wenn gettype versagt hat und type = leer ist
    if (strcmp(type, "LINES") == 0)
    {
        coDoLines *obj = (coDoLines *)new_data;
        int numE = obj->getNumLines();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();
        float *v[3];
        int *el, *cl, pos, next, counter = 0;
        const char **name, **val;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);

        fprintf(file, "%sLINES %d %d %d\n", offset, numE, numC, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
        fprintf(file, "%s%sCONN\n", offset, sp);

        pos = *(el);
        el++;
        next = *(el);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = pos; j < next; j++)
                fprintf(file, "%d ", *(cl)++);

            fprintf(file, "\n");
            pos = next;

            if (i == numE - 2)
                next = numC;
            else
            {
                el++;
                next = *(el);
            }
        }

        fprintf(file, "%s}\n", offset);
    }

    else if (strcmp(type, "POLYGN") == 0)
    {
        coDoPolygons *obj = (coDoPolygons *)new_data;
        int numE = obj->getNumPolygons();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();
        float *v[3];
        int *el, *cl, pos, next, counter = 0;
        const char **name, **val;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);
        fprintf(file, "# Write  %s POLYGONs: %d with Verticies:%d ? and Points: %d\n", offset, numE, numC, numV);
        //      fprintf (file, "%sPOLYGN %d %d %d\n", offset, numE, numC, numV);  //Uwe? Wozu benutzt Ihr offset (=Möglichkeit zum Auskomntieren)
        fprintf(file, "# OK 1 %s   {  \n", offset);
        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "#  %s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);
            fprintf(file, "\n");
        }
        fprintf(file, "Shape{                                 \n");
        fprintf(file, "	appearance Appearance               \n");
        fprintf(file, "	{ material Material {               \n");
        fprintf(file, "#			diffuseColor 0.8 0.1 0.1    \n");
        fprintf(file, "#			ambientIntensity 0.2        \n");
        fprintf(file, "#			emissiveColor 0.0 1.0 0.0   \n");
        fprintf(file, "#			specularColor 0.0 0.0 1.0   \n");
        fprintf(file, "#			shininess 0.2               \n");
        fprintf(file, "#			transparency 0.0            \n");
        fprintf(file, "	}}                                  \n");
        fprintf(file, "	geometry IndexedFaceSet {           \n");
        fprintf(file, "	coord Coordinate {                  \n");
        fprintf(file, "		point [                         \n");
        //	fprintf (file, "#  -1 -1 0,  // First                   \n");
        //	fprintf (file, "#   1 -1 0,                                \n");
        //	fprintf (file, "#   1  1 0,                               \n");
        //	fprintf (file, "#   0  0 1                                 \n");
        //	fprintf (file, "#   ] }                                   \n");

        fprintf(file, "# %s%sVERTEX\n", offset, sp);
        for (i = 0; i < numV - 1; i++) // not se last one, see below
        {
            fprintf(file, " ");
            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }
            fprintf(file, ",\n"); // every get a the end a "," m without the last one
        }

        // now the last one ,  without the last ","
        // fprintf (file, "%s%s%s", offset, sp, sp);
        for (j = 0; j < 3; j++)
        {
            if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12)) || (*v[j] == 0.0))
                fprintf(file, "%f ", *(v[j])++);
            else
                fprintf(file, "%e ", *(v[j])++);
        }
        fprintf(file, " \n"); // every get a the end a "," m without the last one
        fprintf(file, " ] } \n "); // End of Points

        fprintf(file, "coordIndex [       \n  "); // VRML
        fprintf(file, "#%s%sCONN\n", offset, sp); // old

        pos = *(el);
        el++;
        next = *(el);

        for (i = 0; i < numE; i++)
        {
            //   fprintf (file, "#  %s%s%s\n" , offset, sp, sp);
            for (j = pos; j < next; j++)
                fprintf(file, "%d  ", *(cl)++);
            fprintf(file, "-1, \n");
            pos = next;
            if (i == numE - 2)
                next = numC;
            else
            {
                el++;
                next = *(el);
            }
        }
        fprintf(file, "]                       \n");
        fprintf(file, "# color Color { color [1 0 0, 0 1 0, 0 0 1]} \n");
        fprintf(file, "# colorIndex [1 2 0 2  ] \n");
        fprintf(file, "# colorPerVertex TRUE    \n");
        fprintf(file, "#                        \n");
        fprintf(file, "  creaseAngle 20         \n");
        fprintf(file, "  solid TRUE             \n");
        fprintf(file, "  ccw TRUE               \n");
        fprintf(file, "  convex TRUE            \n");
        fprintf(file, "} }                      \n");
        fprintf(file, "# End Shape IndexFaceSet \n");
        fprintf(file, "# %s}\n", offset);
    }

    else if (strcmp(type, "TRIANG") == 0)
    {
        coDoTriangleStrips *obj = (coDoTriangleStrips *)new_data;
        int numV = obj->getNumPoints();
        int numC = obj->getNumVertices();
        int numS = obj->getNumStrips();
        float *v[3];
        int *sl, *cl, pos, next, counter = 0;
        const char **name, **val;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &sl);

        fprintf(file, "%sTRIANG %d %d %d\n", offset, numV, numC, numS);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
        fprintf(file, "%s%sCONN\n", offset, sp);

        pos = *(sl);
        sl++;
        next = *(sl);

        for (i = 0; i < numS; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = pos; j < next; j++)
                fprintf(file, "%d ", *(cl)++);

            fprintf(file, "\n");
            pos = next;

            if (i == numS - 2)
                next = numC;
            else
            {
                sl++;
                next = *(sl);
            }
        }

        fprintf(file, "%s}\n", offset);
    }
    //  -------------------------------
    else if (strcmp(type, "SETELE") == 0) //    "SETELE" ist geschlossen
    {
        int numElem;
        const coDoSet *obj = (const coDoSet *)new_data;
        const coDistributedObject *const *elem = obj->getAllElements(&numElem);
        const char **name, **val;
        char *space = (char *)"      ";
        int counter = 0;

        fprintf(file, "#2 SETELEM %d\n", numElem);
        fprintf(file, "#2 {\n");

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "#2 %sATTR %s %s\n", sp, *(name)++, *(val)++);

            fprintf(file, "#2 \n");
        }

        fprintf(file, "#2 %sELEM\n", sp);
        fprintf(file, "#2 %s{\n", sp);

        for (i = 0; i < numElem; i++)
        {
            fprintf(file, "#2 %s# elem number %d\n", space, i);
            writeObj(space, elem[i]);

            if (i != numElem - 1)
                fprintf(file, "#2 \n");
        }

        fprintf(file, "#2 %s}\n", sp);
        fprintf(file, "#2 }\n");
    }

    else if (strcmp(type, "GEOMET") == 0)
    {
        int has_colors, has_normals, has_texture;
        const coDoGeometry *obj = (const coDoGeometry *)new_data;

        const coDistributedObject *do1 = obj->getGeometry();
        const coDistributedObject *do2 = obj->getColors();
        const coDistributedObject *do3 = obj->getNormals();
        const coDistributedObject *do4 = obj->getTexture();

        const int falseVal = 0;
        const int trueVal = 1;
        if (do2)
            has_colors = trueVal;
        else
            has_colors = falseVal;
        if (do3)
            has_normals = trueVal;
        else
            has_normals = falseVal;
        if (do4)
            has_texture = trueVal;
        else
            has_texture = falseVal;

        fprintf(file, "#GEOMET\n");
        fprintf(file, "#{\n");

        const char **name, **val;
        int counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "#%sATTR %s %s\n", sp, *(name)++, *(val)++);

            fprintf(file, "#\n");
        }
        fprintf(file, "#%sHAS_COLORS    %d\n", sp, has_colors);
        fprintf(file, "#%sHAS_NORMALS   %d\n", sp, has_normals);
        fprintf(file, "#%sHAS_TEXTURES  %d\n", sp, has_texture);

        fprintf(file, "#%sELEM\n", sp);
        fprintf(file, "#%s{\n", sp);

        const char *space = "      ";
        writeObj(space, do1);
        if (do2)
            writeObj(space, do2);
        if (do3)
            writeObj(space, do3);
        if (do4)
            writeObj(space, do4);

        fprintf(file, "#3 %s}\n", sp);
        fprintf(file, "#3 }\n");
    }
    else if (strcmp(type, "POINTS") == 0)
    {
        const coDoPoints *obj = (const coDoPoints *)new_data;
        int numV = obj->getNumPoints();
        float *v[3];
        const char **name, **val;
        int counter = 0;

        obj->getAddresses(&v[0], &v[1], &v[2]);

        fprintf(file, "%sPOINTS %d\n", offset, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", offset);
    }
    /*   else if (strcmp (type, "SPHERE") == 0)
   {
      printf("in sphere");
      const coDoSpheres *obj = (const coDoSpheres *) new_data;
      int numV = obj->getNumSpheres ();
      float *v[3], *radius;
      char **name, **val;
      int counter = 0;

      obj->getAddresses (&v[0], &v[1], &v[2], &radius);

      fprintf (file, "%sSPHERE %d\n", offset, numV);
      fprintf (file, "%s{\n", offset);

      counter = obj->getAllAttributes (&name, &val);

      if (counter != 0)
      {
         for (i = 0; i < counter; i++)
            fprintf (file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

         fprintf (file, "\n");
      }

      fprintf (file, "%s%sVERTEX\n", offset, sp);

      for (i = 0; i < numV; i++)
      {
         fprintf (file, "%s%s%s", offset, sp, sp);

         for (j = 0; j < 3; j++)
         {
            if (((fabs (*v[j]) > 1e-6) && (fabs (*v[j]) < 1e12))
               || (*v[j] == 0.0))
               fprintf (file, "%f ", *(v[j])++);
            else
               fprintf (file, "%e ", *(v[j])++);
         }
         if (((fabs (*radius) > 1e-6) && (fabs (*radius) < 1e12)) || (*radius == 0.0))
            fprintf (file, "%f ", *radius);
         else
            fprintf (file, "%e ", *radius);
         fprintf (file, "\n");
      }

      fprintf (file, "%s}\n", offset);
   }*/

    else if (strcmp(type, "UNIGRD") == 0)
    {
        const coDoUniformGrid *obj = (const coDoUniformGrid *)new_data;
        int xSize, ySize, zSize;
        float xMin, yMin, zMin, xMax, yMax, zMax;
        const char **name, **val;
        int counter = 0;

        obj->getGridSize(&xSize, &ySize, &zSize);
        obj->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);

        fprintf(file, "%sUNIGRD %d %d %d %f %f %f %f %f %f\n",
                offset, xSize, ySize, zSize, xMin, xMax, yMin, yMax, zMin, zMax);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "INTARR") == 0)
    {
        const coDoIntArr *obj = (const coDoIntArr *)new_data;
        int numDim = obj->getNumDimensions();
        int size = obj->getSize();

        int *dataArr = obj->getAddress();

        fprintf(file, "%sINTARR %d  ", offset, numDim);

        for (i = 0; i < numDim; i++)
            fprintf(file, " %d", obj->getDimension(i));

        fprintf(file, "\n%s{\n", offset);

        const char **name, **val;
        int counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVALUES\n", offset, sp);
        for (i = 0; i < size; i++)
            fprintf(file, "%s%s%s%d\n", offset, sp, sp, dataArr[i]);
        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "RCTGRD") == 0)
    {
        const coDoRectilinearGrid *obj = (const coDoRectilinearGrid *)new_data;
        int xSize, ySize, zSize;
        float *vx, *vy, *vz;
        const char **name, **val;
        int counter = 0;

        obj->getGridSize(&xSize, &ySize, &zSize);
        obj->getAddresses(&vx, &vy, &vz);
        fprintf(file, "%sRCTGRD %d %d %d\n", offset, xSize, ySize, zSize);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < xSize; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);
            fprintf(file, "%f\n", vx[i]);
        }

        for (i = 0; i < ySize; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);
            fprintf(file, "%f\n", vy[i]);
        }

        for (i = 0; i < zSize; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);
            fprintf(file, "%f\n", vz[i]);
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "STRGRD") == 0)
    {
        const coDoStructuredGrid *obj = (const coDoStructuredGrid *)new_data;
        int xSize, ySize, zSize;
        float *v[3];
        const char **name, **val;
        int counter = 0;

        obj->getGridSize(&xSize, &ySize, &zSize);
        obj->getAddresses(&v[0], &v[1], &v[2]);

        fprintf(file, "%sSTRGRD %d %d %d\n", offset, xSize, ySize, zSize);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < xSize; i++)
            for (j = 0; j < ySize; j++)
                for (k = 0; k < zSize; k++)
                {
                    fprintf(file, "%s%s%s", offset, sp, sp);

                    for (l = 0; l < 3; l++)
                        if (((fabs(*v[l]) > 1e-6) && (fabs(*v[l]) < 1e12))
                            || (*v[l] == 0.0))
                            fprintf(file, "%f ", *(v[l])++);
                        else
                            fprintf(file, "%e ", *(v[l])++);

                    fprintf(file, "\n");
                }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "USTVDT") == 0)
    {
        const coDoVec3 *obj = (const coDoVec3 *)new_data;
        int numE = obj->getNumPoints();
        float *v[3];
        const char **name, **val;
        int counter = 0;

        obj->getAddresses(&v[0], &v[1], &v[2]);

        fprintf(file, "%sUSTVDT %d\n", offset, numE);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sDATA\n", offset, sp);

        for (i = 0; i < numE; i++)
            fprintf(file, "%s%s%s%f %f %f\n", offset, sp, sp,
                    *v[0]++, *v[1]++, *v[2]++);

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "USTSDT") == 0)
    {
        const coDoFloat *obj = (const coDoFloat *)new_data;
        int numE = obj->getNumPoints();
        float *v;
        const char **name, **val;
        int counter = 0;

        obj->getAddress(&v);

        fprintf(file, "%sUSTSDT %d\n", offset, numE);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sDATA\n", offset, sp);

        for (i = 0; i < numE; i++)
            fprintf(file, "%s%s%s%f\n", offset, sp, sp, *v++);

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "RGBADT") == 0)
    {
        const coDoRGBA *obj = (const coDoRGBA *)new_data;
        int numE = obj->getNumPoints();
        const char **name, **val;
        int counter = 0;

        int *data;

        obj->getAddress(&data);

        fprintf(file, "%sRGBADT %d\n", offset, numE);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sDATA\n", offset, sp);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s%d %d %d %d\n", offset, sp, sp,
                    ((*data) & 0xff000000) >> 24,
                    ((*data) & 0x00ff0000) >> 16,
                    ((*data) & 0x0000ff00) >> 8, ((*data) & 0x000000ff));
            data++;
        }
        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "UNSGRD") == 0)
    {
        const coDoUnstructuredGrid *obj = (const coDoUnstructuredGrid *)new_data;
        int numE, numC, numV, ht;
        float *v[3];
        int *el, *cl, *tl, counter = 0;
        const char **name, **val;

        obj->getGridSize(&numE, &numC, &numV);
        ht = obj->hasTypeList();
        obj->getAddresses(&el, &cl, &v[0], &v[1], &v[2]);
        obj->getTypeList(&tl);
        static const char *type[] = {
            "NONE", "BAR", "TRI", "QUA", "TET", "PYR",
            "PRI", "HEX", "POI"
        };

        fprintf(file, "%sUNSGRD %d %d %d\n", offset, numE, numC, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);

            fprintf(file, "\n");
        }

        fprintf(file, "\n");

        fprintf(file, "%s%sCONN\n", offset, sp);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            if (ht)
            {
                if (tl[i] > -1 && tl[i] < 9)
                    if (strncmp(type[tl[i]], "10", 2) == 0)
                        //if (type[tl[i]] == (char *)"10")
                        fprintf(file, "%s ", type[8]);
            }
            else
                fprintf(file, "%s ", type[tl[i]]);

            for (j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++)
                fprintf(file, "%i ", cl[j]);

            fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", offset);
    }
    else
    {

        sendError("ERROR: Object type not supported by COVISE: '%s'",
                  type);
        return;
    }
}

// write an objet to file
void
WritePolygon::writeSTLObj(const char *offset, const coDistributedObject *new_data)
{
    // get object type
    const char *type = new_data->getType();
    char *sp = (char *)"   ";
    int i;

    if (strcmp(type, "POLYGN") == 0)
    {
        coDoPolygons *obj = (coDoPolygons *)new_data;
        int numE = obj->getNumPolygons();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();
        float *v[3];
        int *el, *cl, counter = 0;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);
        fprintf(file, "solid %s\n",obj->getName());
        if(outputtype == TYPE_STL)
        {
            for (int p = 0; p < numE; p++) // loop over all polygons
            {
                
                coVector a(v[0][cl[el[p]]],v[1][cl[el[p]]],v[2][cl[el[p]]]);
                coVector b(v[0][cl[el[p]+1]],v[1][cl[el[p]+1]],v[2][cl[el[p]+1]]);
                coVector c(v[0][cl[el[p]+2]],v[1][cl[el[p]+2]],v[2][cl[el[p]+2]]);
                coVector ba = b-a;
                coVector ca = c-a;
                coVector n = ba.cross(ca);
                n.normalize();
                fprintf(file, "facet normal %f %f %f\n",n[0],n[1],n[2]);
                fprintf(file, "outer loop\n");
                int numv;
                if(p == numE-1)
                {
                    numv = numC - el[p];
                }
                else
                    numv = el[p+1] - el[p];
                for (int j = 0; j < numv; j++)
                {
                    fprintf(file, "vertex %f %f %f\n", v[0][cl[el[p]+j]],v[1][cl[el[p]+j]],v[2][cl[el[p]+j]]);
                }
                fprintf(file, "endloop\n");
                fprintf(file, "endfacet\n");
            }
        }
        else
        {
            for (int p = 0; p < numE; p++) // loop over all polygons
            {
                coVector a(v[0][cl[el[p]]],v[1][cl[el[p]]],v[2][cl[el[p]]]);
                coVector b(v[0][cl[el[p]+1]],v[1][cl[el[p]+1]],v[2][cl[el[p]+1]]);
                coVector c(v[0][cl[el[p]+2]],v[1][cl[el[p]+2]],v[2][cl[el[p]+2]]);
                coVector ba = b-a;
                coVector ca = c-a;
                coVector n = ba.cross(ca);
                n.normalize();
                int numv;
                if(p == numE-1)
                {
                    numv = numC - el[p];
                }
                else
                    numv = el[p+1] - el[p];
                for (int j = 1; j < numv-1; j++)
                {
                    fprintf(file, "facet normal %f %f %f\n",n[0],n[1],n[2]);
                    fprintf(file, "outer loop\n");
                    fprintf(file, "vertex %f %f %f\n", v[0][cl[el[p]]],v[1][cl[el[p]]],v[2][cl[el[p]]]);
                    fprintf(file, "vertex %f %f %f\n", v[0][cl[el[p]+j]],v[1][cl[el[p]+j]],v[2][cl[el[p]+j]]);
                    fprintf(file, "vertex %f %f %f\n", v[0][cl[el[p]+j+1]],v[1][cl[el[p]+j+1]],v[2][cl[el[p]+j+1]]);

                    fprintf(file, "endloop\n");
                    fprintf(file, "endfacet\n");
                }
            }
        }
        fprintf(file, "endsolid %s\n",obj->getName());
    }

    else if (strcmp(type, "TRIANG") == 0)
    {
        coDoTriangleStrips *obj = (coDoTriangleStrips *)new_data;
        int numV = obj->getNumPoints();
        int numC = obj->getNumVertices();
        int numS = obj->getNumStrips();
        float *v[3];
        int *sl, *cl, counter = 0;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &sl);

        fprintf(file, "solid %s\n",obj->getName());
        for (int p = 0; p < numS; p++) // loop over all polygons
        {
            int numv;
            if(p == numS-1)
            {
                numv = numC - sl[p];
            }
            else
                numv = sl[p+1] - sl[p];
            for (int j = 1; j < numv-1; j++)
            {
                
                coVector a(v[0][cl[sl[p]]],v[1][cl[sl[p]]],v[2][cl[sl[p]]]);
                coVector b(v[0][cl[sl[p]+j]],v[1][cl[sl[p]+j]],v[2][cl[sl[p]+j]]);
                coVector c(v[0][cl[sl[p]+j+1]],v[1][cl[sl[p]+j+1]],v[2][cl[sl[p]+j+1]]);
                coVector ba = b-a;
                coVector ca = c-a;
                coVector n = ba.cross(ca);
                n.normalize();
                fprintf(file, "facet normal %f %f %f\n",n[0],n[1],n[2]);
                fprintf(file, "outer loop\n");
                fprintf(file, "vertex %f %f %f\n", a);
                fprintf(file, "vertex %f %f %f\n", b);
                fprintf(file, "vertex %f %f %f\n", c);
                fprintf(file, "endloop\n");
                fprintf(file, "endfacet\n");
            }
        }
        fprintf(file, "endsolid %s\n",obj->getName());
    }
    //  -------------------------------
    else if (strcmp(type, "SETELE") == 0) //    "SETELE" ist geschlossen
    {
        int numElem;
        const coDoSet *obj = (const coDoSet *)new_data;
        const coDistributedObject *const *elem = obj->getAllElements(&numElem);
        char *space = (char *)"      ";
        int counter = 0;

    
        for (i = 0; i < numElem; i++)
        {
            writeSTLObj(space, elem[i]);

        }

    }

    else if (strcmp(type, "GEOMET") == 0)
    {
        int has_colors, has_normals, has_texture;
        const coDoGeometry *obj = (const coDoGeometry *)new_data;

        const coDistributedObject *do1 = obj->getGeometry();
        const coDistributedObject *do2 = obj->getColors();
        const coDistributedObject *do3 = obj->getNormals();
        const coDistributedObject *do4 = obj->getTexture();

        const int falseVal = 0;
        const int trueVal = 1;
        if (do2)
            has_colors = trueVal;
        else
            has_colors = falseVal;
        if (do3)
            has_normals = trueVal;
        else
            has_normals = falseVal;
        if (do4)
            has_texture = trueVal;
        else
            has_texture = falseVal;

        const char *space = "";
        writeSTLObj(space, do1);
        if (do2)
            writeSTLObj(space, do2);
        if (do3)
            writeSTLObj(space, do3);
        if (do4)
            writeSTLObj(space, do4);

    }
}

// write writeFileBegin  to file
void WritePolygon::writeFileBegin()

{
    if (outputtype == 0)
    {
        fprintf(file, "#VRML V2.0 utf8                   \n");
        fprintf(file, "#  Covise Exporter via WritePolygon, B.Burbaum  \n");
        fprintf(file, "#  WorldInfo {                     \n");
        fprintf(file, "#  info [     ]                    \n");
        fprintf(file, "#  title \"  \" }                  \n");
        fprintf(file, "                                   \n");
        fprintf(file, "   NavigationInfo {                \n");
        fprintf(file, "      avatarSize [0.25, 1.6, 0.75] \n");
        fprintf(file, "      headlight TRUE               \n");
        fprintf(file, "      speed 1.0                    \n");
        fprintf(file, "      type \"WALK\"                \n");
        fprintf(file, "      visibilityLimit 0.0          \n");
        fprintf(file, "      }                            \n");
        fprintf(file, "                                   \n");
    }
    if (outputtype == 1) // 2 = stl
    {
    }
}

// write writeFileEnd  to file
void WritePolygon::writeFileEnd()
{
    if (outputtype == 0)
    {
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, " #   VRML Schluss                  \n");
    }
    if (outputtype == 1) // 2 = stl
    {
    }
}

// compute callback
int
WritePolygon::compute(const char *)
{

    // ???   const char *DataIn = Covise::get_object_name ("DataIn");

    //2   Nur noch im File schreiben daher kann //2 als auch //1 entfallen
    //2    if (DataIn != NULL)
    //2   {
    //2

    // write data to file
    const char *dataPath = p_filename->getValue();

    //Covise::get_browser_param ("path", &dataPath);

    if (dataPath == NULL)
    {

        // invalid data path
        sendError("ERROR: Could not get filename");
        return STOP_PIPELINE;
    }

    int newFile = p_newFile->getValue();

    file = Covise::fopen(dataPath, (newFile ? "w" : "a"));
#ifdef _WIN32
    if (!file)
#else
    if ((!file) || (fseek(file, 0, SEEK_CUR)))
#endif
    {

        // can't create this file
        sendError("ERROR: Could not create file");
        return STOP_PIPELINE;
    }

    // get input data
    char *inputName = Covise::get_object_name("DataIn");

    if (inputName == NULL)
    {

        // can't create object
        Covise::
            sendError("ERROR: Could not create object name for 'dataIn'");

        // close file
        fclose(file);
        return STOP_PIPELINE;
    }

    currentObject = coDistributedObject::createFromShm(inputName);
    if (currentObject == NULL)
    {

        // object creation failed
        sendError("ERROR: createUnknown() failed for data");

        // close file
        fclose(file);
        return STOP_PIPELINE;
    }

    // write FileBegin
    outputtype = p_fileFormat->getValue();
    writeFileBegin();

    // write obj in vrml format
    if(outputtype == TYPE_WRL)
        writeObj("", currentObject);
    else
        writeSTLObj("", currentObject);

    // clean up
    delete currentObject;

    // write FileEnd
    writeFileEnd();

    // close file
    fclose(file);

    return STOP_PIPELINE;
}

// destructor
WritePolygon::~WritePolygon()
{
}

MODULE_MAIN(IO, WritePolygon)
