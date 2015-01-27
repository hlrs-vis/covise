/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Interpolation from Cell Data to Vertex Data               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Werner                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "WriteASC.h"
#include <util/coviseCompat.h>
#include <do/coDoColormap.h>
#include <do/coDoText.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoLines.h>
#include <do/coDoPoints.h>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>

#undef VERBOSE

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

void Application::writeObject(const coDistributedObject *new_data,
                              FILE *file,
                              int verbose,
                              int indent)
{
    int i;
    char sp[1025];
    if (indent)
    {
        for (i = 0; i < 4 * indent; i++)
            if (i & 3)
                sp[i] = ' ';
            else
                sp[i] = '|';
    }
    sp[4 * indent] = '\0';

    if (verbose)
    {
        fprintf(file, "\n%s-----------------------------------"
                      "-----------------------------------\n"
                      "%sObject name: '%s'\n",
                sp, sp, new_data->getName());
        fprintf(file, "%sType string: '%s': ", sp, new_data->getType());
    }
    else
        fprintf(file, "%sObject '%s', Type '%s': ",
                sp, new_data->getName(), new_data->getType());

    const char *type = new_data->getType();

    const char **attrNames, **attrValues;
    int numAttr = new_data->getAllAttributes(&attrNames, &attrValues);
    fprintf(file, "\nAttributes\n");
    for (i = 0; i < numAttr; i++)
    {
        fprintf(file, "%20s %s\n", attrNames[i], attrValues[i]);
    }

    ////////////////////////////// SETELE /////////////////////////////

    if (strcmp(type, "SETELE") == 0)
    {
        int numElem;
        const coDoSet *set = (const coDoSet *)new_data;
        const coDistributedObject *const *elem = set->getAllElements(&numElem);
        if (verbose)
        {
            fprintf(file, "coDoSet\n%s\n", sp);
            fprintf(file, "%sNumber of Elements: %d\n%s\n%s",
                    sp, numElem, sp, sp);
            for (i = 0; i < numElem; i++)
                fprintf(file, "%sType: %s   Name %s\n", sp,
                        elem[i]->getType(), elem[i]->getName());
            for (i = 0; i < numElem; i++)
                writeObject(elem[i], file, verbose, indent + 1);
        }
        else
        {
            fprintf(file, "coDoSet, %d elements\n", numElem);
            for (i = 0; i < numElem; i++)
                writeObject(elem[i], file, verbose, indent + 1);
            fprintf(file, "%s------ coDoSet %s finished ------", sp,
                    new_data->getName());
        }
    }

    ////////////////////////////// USTSDT /////////////////////////////

    else if (strcmp(type, "USTSDT") == 0)
    {
        const coDoFloat *obj = (const coDoFloat *)new_data;
        int numValues = obj->getNumPoints();
        fprintf(file, "Scalar USG Data, %d Elements\n", numValues);
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            float *data;
            obj->getAddress(&data);
            for (i = 0; i < numValues; i++)
                fprintf(file, "%s%7i: %15f\n", sp, i, *data++);
        }
    }

    ////////////////////////////// USTSDT /////////////////////////////

    else if (strcmp(type, "USTSDT") == 0)
    {
        const coDoFloat *obj = (const coDoFloat *)new_data;
        int numValues = obj->getNumPoints();
        fprintf(file, "Scalar USG Data, %d Elements\n", numValues);
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            float *data;
            obj->getAddress(&data);
            for (i = 0; i < numValues; i++)
                fprintf(file, "%s%7i: %15f\n", sp, i, *data++);
        }
    }

    ////////////////////////////// USTSTD /////////////////////////////

    else if (strcmp(type, "USTSTD") == 0)
    {
        const coDoVec2 *obj = (const coDoVec2 *)new_data;
        int numValues = obj->getNumPoints();
        fprintf(file, "Scalar 2D USG Data, %d Elements\n", numValues);
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            float *data[2];
            obj->getAddresses(&data[0], &data[1]);
            for (i = 0; i < numValues; i++)
                fprintf(file, "%s%7i: %15f %15f\n", sp,
                        i, *data[0]++, *data[1]++);
        }
    }

    ////////////////////////////// RGBDAT /////////////////////////////

    else if (strcmp(type, "RGBADT") == 0)
    {
        const coDoRGBA *obj = (const coDoRGBA *)new_data;
        int numValues = obj->getNumPoints();
        fprintf(file, "RGBA Data, %d Elements\n", numValues);
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
#ifdef _INT_IS_64
#error XXXXXX
#endif
            int *data;
            obj->getAddress(&data);
            unsigned char *cVal = (unsigned char *)data;
            for (i = 0; i < numValues; i++)
            {
                fprintf(file, "%s%7i: %5d %5d %5d %5d\n", sp, i,
                        (int)cVal[0], (int)cVal[1], (int)cVal[2], (int)cVal[3]);

                cVal += 4;
            }
        }
    }

    ////////////////////////////// UNIGRD /////////////////////////////

    else if (strcmp(type, "UNIGRD") == 0)
    {

        const coDoUniformGrid *obj = (const coDoUniformGrid *)new_data;
        int x, y, z;
        obj->getGridSize(&x, &y, &z);
        fprintf(file, "Uniform Grid, (%d, %d, %d)-dimensional\n", x, y, z);
        if (verbose)
        {
            float ix, ax, iy, ay, iz, az;
            obj->getMinMax(&ix, &ax, &iy, &ay, &iz, &az);
            fprintf(file, "x: %f -> %f\n", ix, ax);
            fprintf(file, "y: %f -> %f\n", iy, ay);
            fprintf(file, "z: %f -> %f\n", iz, az);
        }
    }

    ////////////////////////////// INTARR /////////////////////////////

    else if (strcmp(type, "INTARR") == 0)
    {

        const coDoIntArr *obj = (const coDoIntArr *)new_data;
        int numDim = obj->getNumDimensions();
        fprintf(file, "%sInteger Array, %d-dimensional: [", sp, numDim);
        int i;
        for (i = 0; i < numDim; i++)
        {
            fprintf(file, "%d", obj->getDimension(i));
            if (i < numDim - 1)
                fprintf(file, ",");
        }
        fprintf(file, "]\n");
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            int *data;
            obj->getAddress(&data);
            for (i = 0; i < obj->getSize(); i++)
            {
                if ((i & 7) == 0)
                    fprintf(file, "\n%s%7i: ", sp, i);
                fprintf(file, "%15i", *data++);
            }
            fprintf(file, "\n");
        }
    }

    ////////////////////////////// DOTEXT /////////////////////////////

    else if (strcmp(type, "DOTEXT") == 0) // Text objects
    {
        const coDoText *obj = (const coDoText *)new_data;
        int numBytes = obj->getTextLength();
        char *data;
        obj->getAddress(&data);
        fprintf(file, "coDoText, %d bytes", numBytes);
        if (verbose)
        {
            fprintf(file, "-----------------------\n");
            if (fwrite(data, 1, numBytes, file) != numBytes)
            {
                fprintf(stderr, "WriteASC::writeObject: fwrite failed\n");
            }
            fprintf(file, "-----------------------\n");
        }
    }

    ////////////////////////////// USTVDT /////////////////////////////

    else if (strcmp(type, "USTVDT") == 0) // VECTOR DATA
    {

        const coDoVec3 *obj = (const coDoVec3 *)new_data;
        int numValues = obj->getNumPoints();
        fprintf(file, "Vector USG Data, %d Elements", numValues);
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            float *data[3];
            obj->getAddresses(&data[0], &data[1], &data[2]);
            for (i = 0; i < numValues; i++)
                fprintf(file, "%s%7i: %15f %15f %15f\n", sp,
                        i, *data[0]++, *data[1]++, *data[2]++);
        }
    }

    ////////////////////////////// USTTDT /////////////////////////////
    else if (strcmp(type, "USTTDT") == 0) // TENSOR DATA
    {
        const coDoTensor *obj = (const coDoTensor *)new_data;
        int numValues = obj->getNumPoints();
        const char *strType = obj->getTensorCharType();
        fprintf(file, "Tensor (%s) USG Data, %d Elements", strType, numValues);
        if (verbose && obj->objectOk())
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            float *data;
            obj->getAddress(&data);
            int j;
            for (i = 0; i < numValues; i++)
            {
                fprintf(file, "%s%7i:", sp, i);
                for (j = 0; j < obj->dimension(); ++j)
                    fprintf(file, " %15f", *(data++));
                fputc('\n', file);
            }
        }
    }

    ////////////////////////////// USTVDT /////////////////////////////

    else if (strcmp(type, "POINTS") == 0) // VECTOR DATA
    {

        const coDoPoints *obj = (const coDoPoints *)new_data;
        int numValues = obj->getNumPoints();
        fprintf(file, "Points Data, %d Elements", numValues);
        if (verbose)
        {
            fprintf(file, "%s\n%s\n", sp, sp);
            float *data[3];
            obj->getAddresses(&data[0], &data[1], &data[2]);
            for (i = 0; i < numValues; i++)
                fprintf(file, "%s%7i: %15f %15f %15f\n", sp,
                        i, *data[0]++, *data[1]++, *data[2]++);
        }
    }

    ////////////////////////////// UNSGRD /////////////////////////////

    else if (strcmp(type, "UNSGRD") == 0)
    {

        const coDoUnstructuredGrid *obj = (const coDoUnstructuredGrid *)new_data;
        int numE, numC, numV, ht;
        obj->getGridSize(&numE, &numC, &numV);
        ht = obj->hasTypeList();

        if (verbose)
        {
            fprintf(file, "Unstructured Grid\n%s\n", sp);
            fprintf(file, "%sNo of Elements:    %8i\n", sp, numE);
            fprintf(file, "%sNo of Connections: %8i\n", sp, numC);
            fprintf(file, "%sNo of Vertices:    %8i\n", sp, numV);
            fprintf(file, "%sHas type list:     %s\n%s\n", sp,
                    (ht ? "TRUE" : "FALSE"), sp);

            int *el, *cl, *tl, j;
            float *v[3];
            obj->getAddresses(&el, &cl, &v[0], &v[1], &v[2]);
            obj->getTypeList(&tl);

            static const char *type[] = { "NONE", "BAR", "TRI", "QUAD", "TET", "PYR", "PRI", "HEX" };

            if (verbose)
            {
                fprintf(file, "%sCell List:\n%s\n", sp, sp);
                for (i = 0; i < numE; i++)
                {
                    fprintf(file, "%s%6i: [%6i-%6i]", sp, i, el[i], ((i < numE - 1) ? el[i + 1] : numC) - 1);
                    if (ht)
                    {
                        if (tl[i] < 0 || tl[i] > 7)
                            fprintf(file, " ILL  ");
                        else
                            fprintf(file, " %4s ", type[tl[i]]);
                    }
                    for (j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++)
                        fprintf(file, "  %6i ", cl[j]);
                    fprintf(file, "%s\n", sp);
                }

                fprintf(file, "%sVertex List:\n%s\n", sp, sp);
                for (i = 0; i < numV; i++)
                {
                    fprintf(file, "%s%7i:", sp, i);
                    for (j = 0; j < 3; j++)
                        if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                            || (*v[j] == 0.0))
                            fprintf(file, "%s %15f ", sp, *(v[j])++);
                        else
                            fprintf(file, "%s %15e ", sp, *(v[j])++);
                    fprintf(file, "%s\n", sp);
                }
            }
            fprintf(file, "%s\n", sp);
        }
        else
            fprintf(file, "UnStrGrid: %d Elem, %d Conn, %d Vert,%s Typelist\n", numE, numC, numV, (ht ? "" : " No"));
    }

    ////////////////////////////// POLYGN /////////////////////////////

    else if (strcmp(type, "POLYGN") == 0)
    {

        const coDoPolygons *obj = (const coDoPolygons *)new_data;

        int numE = obj->getNumPolygons();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();

        if (verbose)
        {

            fprintf(file, "Polygon list\n%s\n", sp);
            fprintf(file, "%sNo of Polygons:    %8i\n", sp, numE);
            fprintf(file, "%sNo of Connections: %8i\n", sp, numC);
            fprintf(file, "%sNo of Vertices:    %8i\n", sp, numV);

            int *el, *cl, j;
            float *v[3];
            obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);

            fprintf(file, "%sPolygon List:\n%s\n", sp, sp);
            for (i = 0; i < numE; i++)
            {
                fprintf(file, "%s%6i: [%6i-%6i]", sp, i, el[i], ((i < numE - 1) ? el[i + 1] : numC) - 1);
                for (j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++)
                    fprintf(file, "%s %6i ", sp, cl[j]);
                fprintf(file, "%s\n", sp);
            }
            fprintf(file, "%s\n", sp);

            fprintf(file, "%sVertex List:\n%s\n", sp, sp);
            for (i = 0; i < numV; i++)
            {
                fprintf(file, "%s%7i:", sp, i);
                for (j = 0; j < 3; j++)
                    if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                        || (*v[j] == 0.0))
                        fprintf(file, "%s %15f ", sp, *(v[j])++);
                    else
                        fprintf(file, "%s %15e ", sp, *(v[j])++);
                fprintf(file, "%s\n", sp);
            }
            fprintf(file, "%s\n", sp);
        }
        else
            fprintf(file, "%sPolygon List: %d Poly, %d Conn, %d Vert", sp, numE, numC, numV);
    }
    ////////////////////////////// Lines /////////////////////////////

    else if (strcmp(type, "LINES") == 0)
    {

        const coDoLines *obj = (const coDoLines *)new_data;

        int numE = obj->getNumLines();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();

        if (verbose)
        {

            fprintf(file, "Line list\n%s\n", sp);
            fprintf(file, "No of Lines:       %8i\n", numE);
            fprintf(file, "No of Connections: %8i\n", numC);
            fprintf(file, "No of Vertices:    %8i\n", numV);

            int *el, *cl, j;
            float *v[3];
            obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);

            fprintf(file, "%sLine List:\n%s\n", sp, sp);
            for (i = 0; i < numE; i++)
            {
                fprintf(file, "%s%6i: [%6i-%6i]", sp, i, el[i], ((i < numE - 1) ? el[i + 1] : numC) - 1);
                for (j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++)
                    fprintf(file, " %6i ", cl[j]);
                fprintf(file, "\n");
            }
            fprintf(file, "\n");

            fprintf(file, "%sVertex List:\n%s\n", sp, sp);
            for (i = 0; i < numV; i++)
            {
                fprintf(file, "%s%7i:", sp, i);
                for (j = 0; j < 3; j++)
                    if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                        || (*v[j] == 0.0))
                        fprintf(file, " %15f ", *(v[j])++);
                    else
                        fprintf(file, " %15e ", *(v[j])++);
                fprintf(file, "\n");
            }
            fprintf(file, "%s\n", sp);
        }
        else
            fprintf(file, "%sPolyList: %d Poly, %d Conn, %d Vert", sp, numE, numC, numV);
    }

    //////////////////// Colormap /////////////////////

    else if (strcmp(type, "COLMAP") == 0)
    {
        const coDoColormap *cmap = (const coDoColormap *)new_data;

        fprintf(file, "%sColormap:\n%s\n", sp, sp);
        fprintf(file, "%sNumber of steps: %8i\n", sp, cmap->getNumSteps());
        fprintf(file, "%sMin/Max:         %8f %8f\n", sp, cmap->getMin(), cmap->getMax());
        fprintf(file, "%sName:            '%s'\n", sp, cmap->getMapName());

        if (verbose)
        {
            fprintf(file, "%s\n", sp);
            float *map = cmap->getAddress();
            for (i = 0; i < cmap->getNumSteps(); i++)
            {
                fprintf(file, "%s x=%6.3f : %6.3f %6.3f %6.3f %6.3f\n", sp,
                        map[4], map[0], map[1], map[2], map[3]);
                map += 5;
            }
        }
    }
    //////////////////// Framework for all types /////////////////////

    else if (strcmp(type, "XYZABC") == 0)
    {
    }

    else
    {
        Covise::sendError("%sSorry, WriteASC doesn't support type %s", sp, type);
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
