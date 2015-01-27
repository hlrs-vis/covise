/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "Testbed.h"

#include <util/coviseCompat.h>

//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    Testbed *application = new Testbed(argc, argv);

    // let the user know about the possible types
    cerr << endl;
    cerr << "=== Testbed: list of possible values for outputMesh and outputData ===" << endl;
    cerr << "===                   and description of possible choices       ===" << endl;
    cerr << "- outputMesh" << endl;
    cerr << "   cuboid   cuboid with center at vertex and" << endl;
    cerr << "        u/v/w hexahedrons in the respective directions" << endl;
    cerr << "   points   u points arranged in a circle around vertex" << endl;
    cerr << "        with radius 1.0" << endl;
    cerr << "   lines   some lines with scalar-data" << endl;

    cerr << endl;
    cerr << "- outputData" << endl;
    cerr << "   circular   circular velocity field" << endl;
    cerr << "       scalars representing something usefull" << endl;

    // and back to covise
    application->run();

    // done
    return 1;
}

void Testbed::compute(const char *)
{
    char *objNames[5];

    // get the parameters
    // changed this to a choice parameter (Uwe)
    //Covise::get_scalar_param( "outputMesh", &outputMesh );
    //Covise::get_scalar_param( "outputData", &outputData );
    Covise::get_choice_param("outputMesh", &outputMesh);
    Covise::get_choice_param("outputData", &outputData);
    outputMesh--;
    outputData--;
    Covise::get_scalar_param("u", &u);
    Covise::get_scalar_param("v", &v);
    Covise::get_scalar_param("w", &w);
    Covise::get_vector_param("vertex", 0, &vertex[0]);
    Covise::get_vector_param("vertex", 1, &vertex[1]);
    Covise::get_vector_param("vertex", 2, &vertex[2]);

    // get object names
    objNames[0] = Covise::get_object_name("mesh");
    objNames[1] = Covise::get_object_name("vectorData_pe");
    objNames[2] = Covise::get_object_name("vectorData_pp");
    objNames[3] = Covise::get_object_name("scalarData_pe");
    objNames[4] = Covise::get_object_name("scalarData_pp");

    // now build the output
    buildOutput(objNames);

    // done
    return;
}

void Testbed::buildOutput(char **objNames)
{
    switch (outputMesh)
    {
    case 1: // points in a circle
    {
        coDoPoints *points = NULL;
        coDoFloat *sd = NULL;
        coDoVec3 *vd = NULL;

        float *x, *y, *z;
        float *d1, *d2, *d3, *d;
        int i;
        float w, c;

        points = new coDoPoints(objNames[0], u);
        points->getAddresses(&x, &y, &z);

        sd = new coDoFloat(objNames[3], u);
        vd = new coDoVec3(objNames[1], u);
        sd->getAddress(&d);
        vd->getAddresses(&d1, &d2, &d3);

        // build points
        w = 2.0 * M_PI / ((float)(u));
        for (i = 0; i < u; i++)
        {
            c = w * ((float)i);

            x[i] = vertex[0] + cos(c);
            y[i] = vertex[1] + sin(c);
            z[i] = vertex[2];

            d[i] = (float)i;

            d1[i] = cos(c + M_PI / 2.0);
            d2[i] = sin(c + M_PI / 2.0);
            d3[i] = 0.0;
        }

        // clean up
        delete points;
        delete sd;
        delete vd;
    }
    break;
    case 2: // lines
    {
        int numCircle;
        coDoLines *lines = NULL;
        coDoFloat *dpv, *dpl;

        float *x, *y, *z;
        float *dv, *dl;
        int *vl, *ll;

        int i, n;
        float c, w;

        numCircle = 10;
        lines = new coDoLines(objNames[0], numCircle + 10, numCircle + 10, 6);
        lines->getAddresses(&x, &y, &z, &vl, &ll);

        dpv = new coDoFloat(objNames[3], numCircle + 10);
        dpv->getAddress(&dv);
        dpl = new coDoFloat(objNames[4], numCircle + 10);
        dpl->getAddress(&dl);

        // first linestrip builds a circle
        ll[0] = 0;
        w = 2.0 * M_PI / ((float)numCircle + 1);
        for (i = 0; i < numCircle; i++)
        {
            c = w * ((float)i);

            x[i] = vertex[0] + cos(c);
            y[i] = vertex[1] + sin(c);
            z[i] = vertex[2];

            dv[i] = ((float)i);
            vl[i] = i;
        }
        dl[0] = 1.0;

        // and 5 parallel lines
        n = 1;
        for (i = numCircle; i < numCircle + 10; i += 2)
        {
            ll[n] = i;
            n++;
            //fprintf(stderr, "ll[%d] = %d\n", i-numCircle+10, i);

            vl[i] = i;
            vl[i + 1] = i + 1;

            dv[i] = ((float)i);
            dv[i + 1] = ((float)i + 1);
            dl[i] = ((float)i);

            x[i] = vertex[0] + 0.3 + ((float)i) / 10.0;
            y[i] = vertex[1];
            z[i] = vertex[2];

            x[i + 1] = vertex[0] + 0.3 + ((float)i) / 10.0;
            y[i + 1] = vertex[1] + 2.0 - ((float)i) / 20.0;
            z[i + 1] = vertex[2];
        }

        // clean up
        delete lines;
        delete dpl;
        delete dpv;
    }
    break;

    default: // unsupported or cuboid
    {
        coDoUnstructuredGrid *grid = NULL;
        coDoFloat *sd = NULL;
        coDoVec3 *vd = NULL;

        float *x, *y, *z;
        int *tl, *el, *cl;
        float *d1, *d2, *d3, *d;
        int i, j, k, o;
        //float c;
        float t1, t2, t3;
        int p1, p2, p3;

        float ex, ey, ez;
        //float as, ac;

        grid = new coDoUnstructuredGrid(objNames[0], u * v * w, (u * v * w) * 8, (u + 1) * (v + 1) * (w + 1), 1);
        grid->getAddresses(&el, &cl, &x, &y, &z);
        grid->getTypeList(&tl);

        sd = new coDoFloat(objNames[3], (u + 1) * (v + 1) * (w + 1));
        vd = new coDoVec3(objNames[1], (u + 1) * (v + 1) * (w + 1));
        sd->getAddress(&d);
        vd->getAddresses(&d1, &d2, &d3);

        ex = 1.0 / ((float)u);
        ey = 1.0 / ((float)v);
        ez = 1.0 / ((float)w);

        // compute vertices (and data)
        o = 0;
        for (i = 0; i < u + 1; i++)
        {
            for (j = 0; j < v + 1; j++)
            {
                for (k = 0; k < w + 1; k++)
                {
                    t1 = ((float)i) * ex;
                    t2 = ((float)j) * ey;
                    t3 = ((float)k) * ez;

                    x[o] = vertex[0] + t1;
                    y[o] = vertex[1] + t2;
                    z[o] = vertex[2] + t3;

                    t1 -= 0.5;
                    t2 -= 0.5;
                    t3 -= 0.5;

                    // circular flow
                    d1[o] = -t2;
                    d2[o] = t1;
                    d3[o] = t3 * 5;

                    d[o] = t1 * t1 + t2 * t2;

                    o++;
                }
            }
        }

        // build hexahedrons
        // element/typelist
        for (i = 0; i < u * v * w; i++)
        {
            el[i] = i * 8;
            tl[i] = TYPE_HEXAGON;
        }
        // connectivities
        o = 0;
        for (i = 0; i < u; i++)
        {
            for (j = 0; j < v; j++)
            {
                p1 = (w + 1) * (v + 1);
                p2 = (w + 1);
                p3 = 1;

                for (k = 0; k < w; k++)
                {
                    cl[o] = i * p1 + j * p2 + k * p3;
                    cl[o + 1] = i * p1 + j * p2 + (k + 1) * p3;
                    cl[o + 2] = (i + 1) * p1 + j * p2 + (k + 1) * p3;
                    cl[o + 3] = (i + 1) * p1 + j * p2 + k * p3;
                    cl[o + 4] = i * p1 + (j + 1) * p2 + k * p3;
                    cl[o + 5] = i * p1 + (j + 1) * p2 + (k + 1) * p3;
                    cl[o + 6] = (i + 1) * p1 + (j + 1) * p2 + (k + 1) * p3;
                    cl[o + 7] = (i + 1) * p1 + (j + 1) * p2 + k * p3;
                    o += 8;
                }
            }
        }
    }
    break;
    }

    // done
    return;
}

/*

   char *file14path, *file15path, *file23path;

   FILE *vertexFile, *boundaryFile, *cellFile;

   long numStarCells, numStarVertices, numStarBound;
   long numVertOut;
   float *xCoord, *yCoord, *zCoord;
   int *vl, *pl;

float *xStarVert, *yStarVert, *zStarVert;
long *vert1, *vert2, *vert3, *vert4;
long *vertUsed;

float x, y, z;
int n;
long id;
long v1, v2, v3, v4, w;
char bfr[1024];
char *objName;

long i, o;

coDoPolygons *boundaryOut;

// get filenames
Covise::get_browser_param( "vertexfile", &file15path );
Covise::get_browser_param( "boundaryfile", &file23path );
Covise::get_browser_param( "cellfile", &file14path );

// read in all vertices
vertexFile = fopen( file15path, "r" );
if( !vertexFile )
{
Covise::sendError( "vertexFile not found !" );
return;
}

// ?!? where can we get these values from
numStarCells = 400000;
numStarVertices = 800000;
numStarBound = 200000;

// alloc mem
xStarVert = new float[numStarVertices];
yStarVert = new float[numStarVertices];
zStarVert = new float[numStarVertices];

// and read in
numStarVertices = 0;
while( !feof(vertexFile) )
{
fscanf( vertexFile, "%d %e %e %e\n", &id, &x, &y, &z );
if( id>0 )
{
if( id>numStarVertices+1 )
// skip unused vertices (starcd suckz)
numStarVertices = id-1;

xStarVert[numStarVertices] = x;
yStarVert[numStarVertices] = y;
zStarVert[numStarVertices] = z;
numStarVertices++;
}
}

// info
cerr << "found " << numStarVertices << " vertices" << endl;

// close file
fclose( vertexFile );

// read in the boundary
boundaryFile = fopen( file23path, "r" );
if( !boundaryFile )
{
Covise::sendError( "boundaryFile not found !" );
return;
}

// alloc mem
vert1 = new long[numStarBound];
vert2 = new long[numStarBound];
vert3 = new long[numStarBound];
vert4 = new long[numStarBound];

int counter[30];
for( i=0; i<30; i++ )
counter[i] = 0;

// and read in
numStarBound = 0;
while( !feof(boundaryFile) )
{
fscanf( boundaryFile, "%d %d %d %d %d %d %d %s\n", &id, &v1, &v2, &v3, &v4, &i, &w, &bfr );
if( id>0 )
{
{
vert1[numStarBound] = v1-1;
vert2[numStarBound] = v2-1;
vert3[numStarBound] = v3-1;
vert4[numStarBound] = v4-1;
}
numStarBound++;

counter[i]++;
}
}

for( i=0; i<30; i++ )
fprintf(stderr, "%d: %d\n", i, counter[i]);

// info
cerr << "found " << numStarBound << " boundary elements" << endl;

// close file
fclose( boundaryFile );

// get really used vertices
vertUsed = new long[numStarVertices];
for( i=0; i<numStarVertices; i++ )
vertUsed[i] = 0;
for( i=0; i<numStarBound; i++ )
{
vertUsed[ vert1[i] ] = 1;
vertUsed[ vert2[i] ] = 1;
vertUsed[ vert3[i] ] = 1;
vertUsed[ vert4[i] ] = 1;
}

// build a dlink-list
numVertOut = 0;
for( i=0; i<numStarVertices; i++ )
if( vertUsed[i] )
{
vertUsed[i] = numVertOut;
numVertOut++;
}
else
vertUsed[i] = -1;

// create objects
objName = Covise::get_object_name( "boundary" );
boundaryOut = new coDoPolygons( objName, numVertOut, numStarBound*4, numStarBound );
boundaryOut->getAddresses( &xCoord, &yCoord, &zCoord, &vl, &pl );

// and load data
// polygonList is simple
for( i=0; i<numStarBound; i++ )
pl[i] = i*4;
// coordinates also simple
for( i=0; i<numStarVertices; i++ )
if( vertUsed[i]!=-1 )
{
xCoord[ vertUsed[i] ] = xStarVert[i];
yCoord[ vertUsed[i] ] = yStarVert[i];
zCoord[ vertUsed[i] ] = zStarVert[i];
}
// now vertexList
for( i=0; i<numStarBound; i++ )
{
o = i*4;
vl[o] = vertUsed[vert1[i]];
vl[o+1] = vertUsed[vert2[i]];
vl[o+2] = vertUsed[vert3[i]];
vl[o+3] = vertUsed[vert4[i]];
}
// we should use 2sided lighting
boundaryOut->addAttribute( "vertexOrder", "2" );

// done
delete boundaryOut;

// clean up
delete[] xStarVert;
delete[] yStarVert;
delete[] zStarVert;
delete[] vert1;
delete[] vert2;
delete[] vert3;
delete[] vert4;

// done
return;
}

*/
