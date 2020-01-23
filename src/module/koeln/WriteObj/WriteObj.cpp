/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "WriteObj.h"
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

void Application::writeObject(const coDistributedObject *geomObj,
                              const coDistributedObject *colorObj,
                              FILE *file,
                              bool singleLines,
                              int &vertexOffset)
{
    const char *type = geomObj->getType();

    const char **attrNames, **attrValues;
    int numAttr = geomObj->getAllAttributes(&attrNames, &attrValues);

    if (strcmp(type, "SETELE") == 0)
    {
        int numElem = 0;
        const coDoSet *set = (const coDoSet *)geomObj;
        const coDistributedObject *const *elem = set->getAllElements(&numElem);

        const coDoSet *colorSet = (const coDoSet *)colorObj;
        const coDistributedObject *const *colorElem = NULL;
        int numColorElem = 0;
        if (colorSet != NULL)
            colorElem = colorSet->getAllElements(&numColorElem);
        for (int i = 0; i < numElem; i++) {
            const coDistributedObject *col = NULL;
            if (colorElem != NULL && i < numColorElem)
                col = colorElem[i];
            writeObject(elem[i], col, file, singleLines, vertexOffset);
        }
    }
    else if (strcmp(type, "LINES") == 0)
    {
        const coDoLines *obj = (const coDoLines *)geomObj;

        const coDoRGBA *rgba = (const coDoRGBA *)colorObj;

        size_t numE = obj->getNumLines();
        size_t numC = obj->getNumVertices();
        size_t numV = obj->getNumPoints();
        size_t numRGBA = rgba != NULL ? rgba->getNumPoints() : 0;

        int *el, *cl;
        float *v[3];
        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);
        int *cols;
        if (rgba)
            rgba->getAddress(&cols);

        // Write out vertices and (optionally) vertex colors
        for (size_t i = 0; i < numV; ++i)
        {
            fprintf(file, "v %f %f %f", v[0][i], v[1][i], v[2][i]);
            if (rgba != NULL) {
                unsigned col(cols[i]);
                //unsigned a = col & 0xFF;
                unsigned b = (col >> 8) & 0xFF;
                unsigned g = (col >> 16) & 0xFF;
                unsigned r = (col >> 24) & 0xFF;
                fprintf(file, " %f %f %f", r/255.f, g/255.f, b/255.f);
            }
            fprintf(file, "\n");
        }

        // Write out line segments
        for (size_t i = 0; i < numE; ++i)
        {
            if (singleLines)
            {
                // Each line segment ends up on a single line, like:
                // l 1 2
                // l 2 3
                // l 3 4
                // ...
                size_t numS = (i < numE - 1) ? el[i + 1] : numC;
                for (size_t j = el[i]; j < numS-1; ++j) {
                    fprintf(file, "l %d %d\n", vertexOffset+cl[j]+1, vertexOffset+cl[j+1]+1);
                }
            }
            else
            {
                fprintf(file, "l");
                for (size_t j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++) {
                    fprintf(file, " %d", vertexOffset+cl[j]+1);
                }
                fprintf(file, "\n");
            }
        }

        vertexOffset += numV;
    }
    else if (strcmp(type, "POLYGN") == 0)
    {
        const coDoPolygons *obj = (const coDoPolygons *)geomObj;

        const coDoRGBA *rgba = (const coDoRGBA *)colorObj;

        size_t numE = obj->getNumPolygons();
        size_t numC = obj->getNumVertices();
        size_t numV = obj->getNumPoints();
        size_t numRGBA = rgba != NULL ? rgba->getNumPoints() : 0;

        int *el, *cl;
        float *v[3];
        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);
        int *cols;
        if (rgba)
            rgba->getAddress(&cols);

        // Write out vertices and (optionally) vertex colors
        for (size_t i = 0; i < numV; ++i)
        {
            fprintf(file, "v %f %f %f", v[0][i], v[1][i], v[2][i]);
            if (rgba != NULL) {
                unsigned col(cols[i]);
                //unsigned a = col & 0xFF;
                unsigned b = (col >> 8) & 0xFF;
                unsigned g = (col >> 16) & 0xFF;
                unsigned r = (col >> 24) & 0xFF;
                fprintf(file, " %f %f %f", r/255.f, g/255.f, b/255.f);
            }
            fprintf(file, "\n");
        }

        // Write out triangle fans
        for (size_t i = 0; i < numE; ++i)
        {
            fprintf(file, "f");
            for (size_t j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++) {
                fprintf(file, " %d", vertexOffset+cl[j]+1);
            }
            fprintf(file, "\n");
        }

        vertexOffset += numV;
    }
    else
    {
        Covise::sendError("Sorry, WriteObj doesn't support type %s", type);
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
    int singleLines;
    Covise::get_boolean_param("singleLines", &singleLines);

    FILE *file = Covise::fopen(path, (newFile ? "w" : "a"));
    if ((!file) || (fseek(file, 0, SEEK_CUR)))
    {
        Covise::sendError("Could not create file");
        return;
    }

    // ========================== Get input data ======================

    // Geometry
    char *geomName = Covise::get_object_name("dataIn0");
    if (geomName == NULL)
    {
        Covise::sendError("Error creating object name for 'dataIn0'");
        fclose(file);
        return;
    }

    const coDistributedObject *geomObj = coDistributedObject::createFromShm(geomName);
    if (geomObj == NULL)
    {
        Covise::sendError("createFromShm() failed for data");
        fclose(file);
        return;
    }

    // Colors
    char *colorName = Covise::get_object_name("dataIn1");
    const coDistributedObject *colorObj = NULL;
    if (colorName != NULL)
    {
        colorObj = coDistributedObject::createFromShm(colorName);
    }

    int vertexOffset = 0;
    writeObject(geomObj, colorObj, file, singleLines, vertexOffset);
    delete geomObj;
    delete colorObj;
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
