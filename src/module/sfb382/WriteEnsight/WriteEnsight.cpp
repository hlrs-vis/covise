/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2001 RUS  **
 **                                                                        **
 ** Description: Write coDoLines in Ensight ASCII data format               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 06.04.01                                                 **
\**************************************************************************/

#include <stdio.h>
#include <api/coModule.h>
#include "WriteEnsight.h"

/// Constructor
coWriteEnsight::coWriteEnsight()
{
    // Create parameters:
    pa_filename = addFileBrowserParam("filename", "ASCII data file (.dat)");
    pa_filename->setValue("/mnt/pro/cod/sfb382/", "*.geo");

    // Create ports:
    po_lines = addInputPort("lines", "coDoSet", "Lines");
    po_lines->setInfo("Line data.");
    po_colors = addInputPort("colors", "coDoSet", "Colors for lines");
    po_colors->setInfo("Colors for lines.");
}

/// Destructor
coWriteEnsight::~coWriteEnsight()
{
}

/// Called before application terminates.
void coWriteEnsight::quit()
{
}

void coWriteEnsight::writeGeometryFile(FILE *fp)
{
    coDistributedObject *tmpObj; // temporary object
    coDoSet *set; // set
    coDoLines *lines; // lines
    coDistributedObject *const *setElements; // set elements
    int i, j, k;
    int numLines, numVertices, numPoints; // line attributes
    int numBars;
    int *ll, *vl;
    int vi; // vertex index
    float *x, *y, *z;

    // Get pointer to set of lines:
    tmpObj = po_lines->getCurrentObject();
    if (!tmpObj->isType("SETELE"))
    {
        cerr << "Set expected." << endl;
        const char *t = tmpObj->getType();
        cerr << "Found: " << t << endl;
        return;
    }
    set = (coDoSet *)tmpObj;
    setElements = set->getAllElements(&timesteps);

    // Process time steps:
    cerr << "Processing " << timesteps << " time steps for geometry file" << endl;
    for (i = 0; i < timesteps; ++i)
    {
        if (setElements[i]->isType("LINES"))
        {
            // Write file header:
            if (timesteps > 1)
                fputs("BEGIN TIME STEP\n", fp);
            fputs("File created by Covise module WriteEnsight\n", fp);
            fprintf(fp, "Line data for time step %d\n", i);
            fputs("node id assign\n", fp);
            fputs("element id assign\n", fp);

            // Get line data:
            lines = (coDoLines *)setElements[i];
            numLines = lines->getNumLines();
            numVertices = lines->getNumVertices();
            numPoints = lines->getNumPoints();
            lines->getAddresses(&x, &y, &z, &vl, &ll);

            // Write point coordinates to file:
            fputs("coordinates\n", fp);
            fprintf(fp, "%8d\n", numPoints);
            for (j = 0; j < numPoints; ++j)
                fprintf(fp, "%12.5e%12.5e%12.5e\n", x[j], y[j], z[j]);

            // Write lines to file (each line is a part):
            vi = 0;
            for (j = 0; j < numLines; ++j)
            {
                fprintf(fp, "part %d\n", j + 1);
                fprintf(fp, "line %d-%d\n", i + 1, j + 1);
                fputs("bar2\n", fp);
                numBars = ((j < numLines - 1) ? ll[j + 1] : numVertices) - ll[j] - 1;
                fprintf(fp, "%8d\n", numBars);
                for (k = 0; k < numBars; ++k)
                {
                    fprintf(fp, "%8d%8d\n", vl[vi] + 1, vl[vi + 1] + 1);
                    ++vi;
                }
                ++vi;
            }
            if (timesteps > 1)
                fputs("END TIME STEP\n", fp);
        }
    }
    delete[] setElements;
}

/// Write variables file containing scalar values for lines,
/// to be interpreted as colors.
void coWriteEnsight::writeVariablesFile(FILE *fp)
{
    coDistributedObject *tmpObj; // temporary object
    coDoSet *set; // set
    coDoLines *lines; // lines
    coDoFloat *colors; // colors
    coDistributedObject *const *setLines; // lines of set
    coDistributedObject *const *setColors; // colors of set
    int i, j;
    int numLines, numPoints; // line attributes
    int numColors;
    int *ll, *vl;
    float *x, *y, *z;
    float *col;

    // Get pointer to set of lines:
    tmpObj = po_lines->getCurrentObject();
    if (!tmpObj->isType("SETELE"))
    {
        cerr << "Set expected." << endl;
        const char *t = tmpObj->getType();
        cerr << "Found: " << t << endl;
        return;
    }
    set = (coDoSet *)tmpObj;
    setLines = set->getAllElements(&timesteps);

    // Get pointer to set of colors:
    tmpObj = po_colors->getCurrentObject();
    if (!tmpObj->isType("SETELE"))
    {
        cerr << "Set expected." << endl;
        const char *t = tmpObj->getType();
        cerr << "Found: " << t << endl;
        return;
    }
    set = (coDoSet *)tmpObj;
    setColors = set->getAllElements(&timesteps);

    // Process time steps:
    cerr << "Processing " << timesteps << " time steps for variables file" << endl;
    for (i = 0; i < timesteps; ++i)
    {
        if (setLines[i]->isType("LINES") && setColors[i]->isType("USTSDT"))
        {
            // Write file header:
            if (timesteps > 1)
                fputs("BEGIN TIME STEP\n", fp);
            fprintf(fp, "Colors for time step %d\n", i);

            // Get line data:
            lines = (coDoLines *)setLines[i];
            numLines = lines->getNumLines();
            numPoints = lines->getNumPoints();
            lines->getAddresses(&x, &y, &z, &vl, &ll);

            // Get colors data:
            colors = (coDoFloat *)setColors[i];
            numColors = colors->getNumPoints();
            colors->getAddress(&col);

            // Write colors to file (each line is a part):
            if (numColors == numPoints)
            {
                for (j = 0; j < numLines; ++j)
                {
                    fprintf(fp, "part %d\n", j + 1);
                    fputs("bar2\n", fp);
                    fprintf(fp, "%12.5e\n", (float)(col[ll[j]]));
                }
            }
            else
                cerr << "Data integrity error" << endl;
            if (timesteps > 1)
                fputs("END TIME STEP\n", fp);
        }
    }
    delete[] setColors;
    delete[] setLines;
}

/// Write case file for Ensight 6 and up
void coWriteEnsight::writeCaseFile(FILE *fp, char *geofile, char *varfile)
{
    int i;
    char gfile[256];
    char vfile[256];

    strcpyTail(gfile, geofile, '/');
    strcpyTail(vfile, varfile, '/');
    fputs("FORMAT\n", fp);
    fputs("type: ensight\n\n", fp);
    fputs("GEOMETRY\n", fp);
    fprintf(fp, "model:           1   %s", gfile);
    if (timesteps > 1)
        fputs(" 1", fp);
    fputs("\n\nVARIABLE\n", fp);
    fprintf(fp, "scalar per element: 1  Color  %s\n", vfile);

    if (timesteps > 1)
    {
        fputs("\nTIME\n", fp);
        fputs("time set:        1\n", fp);
        fprintf(fp, "number of steps: %d\n", timesteps);
        fputs("time values:", fp);
        for (i = 0; i < timesteps; ++i)
        {
            fprintf(fp, " %d", i);
            if (i > 0 && (i % 10) == 0) // lines must be limited to 79 characters
                fputs("\n", fp);
        }
        fputs("\n", fp);
        fputs("\nFILE\n", fp);
        fputs("file set:        1\n", fp);
        fprintf(fp, "number of steps: %d\n", timesteps);
    }
}

//----------------------------------------------------------------------------
/** Copies the tail string after the last occurrence of a given character.
    Example: str="local/testfile.dat", c='/' => suffix="testfile.dat"
    @param suffix <I>allocated</I> space for the found string
    @param str    source string
    @param c      character after which to copy characters
    @return result in suffix, empty string if c was not found in str
*/
void coWriteEnsight::strcpyTail(char *suffix, const char *str, char c)
{
    int i, j;

    // Search for c in pathname:
    i = strlen(str) - 1;
    while (i >= 0 && str[i] != c)
        --i;

    // Extract tail string:
    if (i < 0) // c not found?
        strcpy(suffix, "");
    else
    {
        for (j = i + 1; j < (int)strlen(str); ++j)
            suffix[j - i - 1] = str[j];
        suffix[j - i - 1] = '\0';
    }
}

int coWriteEnsight::compute()
{
    FILE *fp;
    const char *basename;
    char geofile[256];
    char casefile[256];
    char varfile[256];

    // Get parameters from covise
    basename = pa_filename->getValue();

    // Create line data file:
    strcpy(geofile, basename);
    strcat(geofile, ".geo");
    cout << "Writing geometry file " << geofile << endl;
    fp = Covise::fopen(geofile, "w");
    if (!fp)
    {
        Covise::sendError("Cannot create geometry file.");
        return 0;
    }
    writeGeometryFile(fp);
    fclose(fp);

    // Create variables file:
    strcpy(varfile, basename);
    strcat(varfile, ".var");
    cout << "Writing variables file " << varfile << endl;
    fp = Covise::fopen(varfile, "w");
    if (!fp)
    {
        Covise::sendError("Cannot create variables file.");
        return 0;
    }
    writeVariablesFile(fp);
    fclose(fp);

    // Create case file:
    strcpy(casefile, basename);
    strcat(casefile, ".case");
    cout << "Writing case file " << casefile << endl;
    fp = Covise::fopen(casefile, "w");
    if (!fp)
    {
        Covise::sendError("Cannot create case file.");
        return 0;
    }
    writeCaseFile(fp, geofile, varfile);
    fclose(fp);

    return 0;
}

/// Startup routine
int main(int argc, char *argv[])
{
    coWriteEnsight *application = new coWriteEnsight();
    application->start(argc, argv);
    return 0;
}
