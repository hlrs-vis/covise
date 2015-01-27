/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                   	      (C)2008 HLRS  **
**                                                                        **
** Description: READ Idea (MPA)                                           **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <ctype.h>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include "ReadIdea.h"

#include "defs.h"
#include "decl.h"
#include "edge.h"
#include "dc.h"

extern point *p_array;

ReadIdea::ReadIdea(int argc, char *argv[])
    : coModule(argc, argv, "Idea Reader")
{
    // the output ports
    p_mesh = addOutputPort("mesh", "UnstructuredGrid", "the grid");
    p_intensity = addOutputPort("intensity", "Float", "the intensity");
    p_polygons = addOutputPort("polygons", "Polygons", "2D polygons");

    // parameters

    // input file
    p_mainFile = addFileBrowserParam("main_File", "main file");
    p_mainFile->setValue("/data/mpa/frick", "*.txt");

    // other params
    const char *method_labels[] = { "ultrasound", "radar" };
    p_method = addChoiceParam("method", "measurement method (not implemented yet)");
    p_method->setValue(2, method_labels, 0);

    p_freqStart = addFloatParam("lowest_frequency", "frequency min to use");
    p_freqStart->setValue(0.);

    p_freqEnd = addFloatParam("highest_frequency", "frequency max to use");
    p_freqEnd->setValue(100000.);

    p_zScale = addFloatParam("frequency_scale", "frequency scale (z-direction scale factor)");
    p_zScale->setValue(1.);

    p_zMax = addFloatParam("z_max", "z max value. Only used if != 0.0");
    p_zMax->setValue(0.);

    p_minAngle = addFloatParam("min_angle", "minimum angle for outer triangles (if angle is smaller, triangles are removed)");
    p_minAngle->setValue(15.);

    p_normalize = addBooleanParam("normalize", "normalize intensities after building the average?");
    p_normalize->setValue(false);

    p_inverse_frequency = addBooleanParam("inverse_frequencies", "use reciprocal value of frequency?");
    p_inverse_frequency->setValue(false);

    //p_getZLevel = addFloatParam("zlevel_of_freq", "actual z-Level of a given frequency is printed on console");
    //p_getZLevel->setValue(0);
}

ReadIdea::~ReadIdea()
{
}

// =======================================================

int ReadIdea::compute(const char *)
{
    // open the file
    //int i,j;
    float *xCoord2D, *yCoord2D, *zCoord2D; // coordinate lists
    //const char ***fileNames;  // the filenames where or data is stored
    std::vector<std::vector<string> > fileNames;

    int *nMeasurements; // measurements done per node

    int nCoord2D; // number of vertices list (=number of measurement points)

    int maxMeasures;

    int *elem, *conn, *tl; // element, connectivity and type list

    coDoUnstructuredGrid *grid = NULL;
    coDoPolygons *polys = NULL;

    int nElem3D; // number of elements
    int nConn3D; // number of edges (length of connectivity list)
    int nCoord3D; // number of nodes, frequency is on z-axis

    float *xCoord3D;
    float *yCoord3D;
    float *zCoord3D;

    coDoFloat *intensity_out = NULL;

    // for delaunay triangulation
    edge *l_cw, *r_ccw;
    point **p_sorted, **p_temp;

    // open mainfile
    ifstream mainfile;
    mainfile.open(p_mainFile->getValue());

    if (mainfile.is_open())
    {
        // get number of measure points (count number of lines)
        string t;
        int lineCount = 0;

        while (getline(mainfile, t, '\n'))
        {
            ++lineCount;
        }
        nCoord2D = lineCount / 4;

        cerr << endl;
        cerr << "number of nodes is " << nCoord2D << endl;
        ;

        if (nCoord2D == 0)
        {
            cerr << "somethings's not OK with your input file." << endl;
            return STOP_PIPELINE;
        }

        mainfile.clear();
        mainfile.seekg(0, ios::beg);

        getline(mainfile, t, '\n');

        size_t found;
        found = t.rfind("Measurement");

        string t2 = t.substr(found + 12);

        sscanf(t2.c_str(), "%d", &maxMeasures);
        fprintf(stderr, "maximum number of measurements is %d\n", maxMeasures);

        // read coordinates and measurement filenames
        // ...

        xCoord2D = new float[nCoord2D];
        yCoord2D = new float[nCoord2D];
        zCoord2D = new float[nCoord2D];
        nMeasurements = new int[nCoord2D];

        fileNames.resize(nCoord2D);

        // for triangulation
        alloc_memory(nCoord2D);

        for (int i = 0; i < nCoord2D; i++)
        {
            int dummy;

            getline(mainfile, t, '\n'); // Exp-Line

            sscanf(t.c_str(), "%d %f %f", &dummy, &xCoord2D[i], &yCoord2D[i]);
            p_array[i].x = xCoord2D[i];
            p_array[i].y = yCoord2D[i];

            zCoord2D[i] = 0.;

            //cerr << "xCoord2D[" << i << "]=" << xCoord2D[i] << ", yCoord2D[" << i << "]=" << yCoord2D[i] << endl;

            getline(mainfile, t, '\n'); // Sig-Line
            getline(mainfile, t, '\n'); // Spec-Line

            int count = 0;
            int pos = 0;
            while (pos != string::npos)
            {
                pos = t.find(".dat", pos);
                if (pos != string::npos)
                {
                    ++count;
                    pos += 4;
                }
            }
            nMeasurements[i] = count;
            fileNames[i].resize(nMeasurements[i]);

            count = 0;
            char *buf;
            buf = strdup(t.c_str());
            strtok(buf, "\t");

            for (int k = 0; k < 2; k++)
            {
                strtok(NULL, "\t");
            }
            for (int j = 0; j < nMeasurements[i]; j++)
            {
                fileNames[i][j].append(strtok(NULL, "\t"));
            }
            getline(mainfile, t, '\n'); // Stat-Line
        }

        mainfile.close();
    }
    else
    {
        sendError("could not open File %s", p_mainFile->getValue());
        return STOP_PIPELINE;
    }

    // Initialise entry edge pointers
    for (int i = 0; i < nCoord2D; i++)
        p_array[i].entry_pt = NULL;

    // Sort for delaunay
    p_sorted = (point **)malloc((unsigned)nCoord2D * sizeof(point *));
    if (p_sorted == NULL)
        sendError("triangulate: not enough memory\n");
    p_temp = (point **)malloc((unsigned)nCoord2D * sizeof(point *));
    if (p_temp == NULL)
        sendError("triangulate: not enough memory\n");
    for (int i = 0; i < nCoord2D; i++)
        p_sorted[i] = p_array + i;
    merge_sort(p_sorted, p_temp, 0, nCoord2D - 1);

    free((char *)p_temp);

    // Triangulate
    divide(p_sorted, 0, nCoord2D - 1, &l_cw, &r_ccw);

    free((char *)p_sorted);

    // get bounding box
    float xmin = FLT_MAX;
    float xmax = -FLT_MAX;
    float ymin = FLT_MAX;
    float ymax = -FLT_MAX;
    for (int i = 0; i < nCoord2D; i++)
    {
        if (xCoord2D[i] < xmin)
            xmin = xCoord2D[i];
        else if (xCoord2D[i] > xmax)
            xmax = xCoord2D[i];

        if (yCoord2D[i] < ymin)
            ymin = yCoord2D[i];
        else if (yCoord2D[i] > ymax)
            ymax = yCoord2D[i];
    }
    //cerr << "xmin=" << xmin << ", xmax=" << xmax << ", ymin=" << ymin << ", ymax=" << ymax << endl;

    // set zmax as average between dx and dy
    float zmax;
    if (p_zMax->getValue() != 0.0)
    {
        zmax = p_zMax->getValue();
    }
    else
    {
        zmax = 0.5 * ((ymax - ymin) + (xmax - xmin));
    }

    // get number of frequency-values in spec-File
    // we assume that this number is constant in all our spec files
    string file = p_mainFile->getValue();
    string path = file.erase(file.rfind("/") + 1);
    path += "Analysed spectra/";

    string caseName = p_mainFile->getValue();
    caseName.erase(caseName.rfind("-"));
    caseName.erase(0, caseName.rfind("-") + 1);
    cerr << "caseName is " << caseName << endl;

    ifstream specfile;
    file = path + caseName + "-" + fileNames[0][0];
    specfile.open(file.c_str());

    //cerr << "trying to open " << file << endl;

    int nFreqs = 0;
    float fr_max, fr_min;

    if (specfile.is_open())
    {
        nFreqs = 0;
        string t;
        float fr, in, fr_old = -1.;

        // get fr_min
        getline(specfile, t, '\n');
        sscanf(t.c_str(), "%f %f", &fr_min, &in);
        specfile.seekg(0, ios::beg);

        while (getline(specfile, t, '\n'))
        {
            sscanf(t.c_str(), "%f %f", &fr, &in);
            if (fr == fr_old) // repeating frequency: measurement seems to be stopped
                break;
            ++nFreqs;
            fr_old = fr;
        }
        cerr << "number of frequencies is " << nFreqs << endl;
        fr_max = fr;
    }
    else
    {
        sendError("could not open File %s", file.c_str());
        return STOP_PIPELINE;
    }
    specfile.close();

    // maybe we want to read just an interval of all frequencies
    int kstart, kend;

    //fprintf(stderr,"fr_min=%8.5f\n",fr_min);
    //fprintf(stderr,"fr_max=%8.5f\n",fr_max);

    kstart = (int)(p_freqStart->getValue() / (fr_max - fr_min) * nFreqs);
    kend = (int)(p_freqEnd->getValue() / (fr_max - fr_min) * nFreqs);

    if (kstart < 0)
    {
        kstart = 0;
        sendWarning("p_freqStart too small. Resetting it to 0.0");
        p_freqStart->setValue(0.0);
    }
    if (kstart > (nFreqs - 1))
    {
        kstart = 0;
        sendWarning("p_freqStart too big. Resetting it to 0.0");
        p_freqStart->setValue(0.0);
    }
    if (kend < 0)
    {
        kend = nFreqs - 1;
        sendWarning("p_freqEnd too small. Resetting it to %f", fr_max);
        p_freqEnd->setValue(fr_max);
    }
    if (kend > (nFreqs - 1))
    {
        kend = nFreqs - 1;
        sendWarning("p_freqEnd too big. Resetting it to %f", fr_max);
        p_freqEnd->setValue(fr_max);
    }

    //fprintf(stderr,"kstart=%d\n",kstart);
    //fprintf(stderr,"kend=%d\n",kend);

    // read intensities
    float *intensity;
    // we read all the data, build the average, normalize and then
    // just use the interval we're interested in: kstart ... kend
    int nFreqs_total = nFreqs;
    nFreqs = kend - kstart;
    intensity_out = new coDoFloat(p_intensity->getObjName(), nCoord2D * nFreqs_total);
    intensity_out->getAddress(&intensity);
    memset(intensity, 0, nCoord2D * nFreqs_total * sizeof(float));

    float intense;

    char buf[500];

    float factor;
    float dummy;

    FILE *datafile;

    for (int i = 0; i < nCoord2D; i++)
    {
        factor = 1. / nMeasurements[i];

        for (int j = 0; j < nMeasurements[i]; j++)
        {
            file = path + caseName + "-" + fileNames[i][j];

            if ((datafile = fopen(file.c_str(), "r")) <= 0)
            {
                sendError("ERROR: can't open file: %s", file.c_str());
                return STOP_PIPELINE;
            }
            else
            {
                int k;
                for (k = 0; k < nFreqs_total; k++)
                {
                    fgets(buf, 500, datafile);
                    sscanf(buf, "%f %f", &dummy, &intense);
                    intensity[k * nCoord2D + i] += factor * intense; // we build the average
                }
            }

            fclose(datafile);
        }
    }

    // now we have all our values in intensity array
    // let's check for max values for normalization
    float *maxval = new float[nCoord2D];
    memset(maxval, 0, nCoord2D * sizeof(float));

    for (int i = 0; i < nCoord2D; i++)
        for (int k = 0; k < nFreqs_total; k++)
            if (intensity[k * nCoord2D + i] > maxval[i])
                maxval[i] = intensity[k * nCoord2D + i];

    // normalize after building the average
    if (p_normalize->getValue())
    {
        for (int i = 0; i < nCoord2D; i++)
        {
            factor = 1. / maxval[i];
            // in the same step just take frequencies from kstart to kend
            for (int j = kstart; j < kend; j++)
            {
                intensity[(j - kstart) * nCoord2D + i] = factor * intensity[j * nCoord2D + i];
            }
        }
    }
    delete[] maxval;

    /*
for (int i=0;i<nCoord2D*nFreqs;i++)
{
   if (intensity[i]<.0)
      cerr << "intensity[" << i << "]<0." << ", i=" << i << endl;
   
}
*/
    //cerr << "intensity[" << savenr << "]=" << intensity[savenr] << endl;

    // build 3D mesh
    // polygons are extendes to prisms

    int nz = kend - kstart + 1;
    nCoord3D = nCoord2D * nz; // number of nodes, frequency is on z-axis

    xCoord3D = new float[nCoord3D];
    yCoord3D = new float[nCoord3D];
    zCoord3D = new float[nCoord3D];

    float *zlevel = new float[nz + 1];
    float freqstep = (fr_max - fr_min) / (nz - 1);
    float f0 = fr_min + kstart * freqstep;

    //float zLevel_freq = p_getzLevel->getValue();

    if (p_inverse_frequency->getValue())
    {

        for (int i = 0; i < nFreqs + 1; i++)
        {
            zlevel[i] = 1. / (f0 + (i + 1) * freqstep) * zmax * (f0 + freqstep) * p_zScale->getValue();
        }
        for (int i = 0; i < nFreqs + 1; i++)
        {
            zlevel[i] -= zlevel[nFreqs];
        }
    }
    else
    {
        float dz = zmax / (nz - 1) * p_zScale->getValue();
        for (int i = 0; i < nz; i++)
        {
            zlevel[i] = i * dz;
        }
    }

    float v[3];

    for (int j = 0; j < nz; j++)
    {
        for (int i = 0; i < nCoord2D; i++)
        {
            v[0] = xCoord2D[i];
            v[1] = yCoord2D[i];
            v[2] = zlevel[j]; //j*dz;
            xCoord3D[j * nCoord2D + i] = v[0];
            yCoord3D[j * nCoord2D + i] = v[1];
            zCoord3D[j * nCoord2D + i] = v[2];
        }
    }

    int nElem2D;
    int *conn2D;

    // remove triangles at outer edges (condition: 2 angles < min_angle)
    removeOuterEdges(nCoord2D);
    conn2D = get_triangles(nCoord2D, &nElem2D);

    cerr << "triangulated " << nElem2D << " polygons" << endl;

    nElem3D = nElem2D * (nz - 1);
    nConn3D = nElem3D * 6; // we have a mesh of prisms (6 nodes / prism)

    cerr << "nElem3D=" << nElem3D << endl;

    elem = new int[nElem3D];
    conn = new int[nConn3D];
    tl = new int[nElem3D];
    int nodes[3];

    int *elem2D = new int[nElem2D];
    int elemnr;

    for (int i = 0; i < nElem2D; i++)
    {
        nodes[0] = conn2D[3 * i + 0];
        nodes[1] = conn2D[3 * i + 1];
        nodes[2] = conn2D[3 * i + 2];

        //cerr << "i=" << i << ": " << nodes[0] << ", " << nodes[1] << ", " << nodes[2] << endl;

        elem2D[i] = 3 * i;

        for (int k = 0; k < nz - 1; k++)
        {
            elemnr = i * (nz - 1) + k;
            //cerr << "elemnnr=" << elemnr << "i=" << i << ", k=" << k << endl;
            tl[elemnr] = 6; //TYPE_PRISM;
            elem[elemnr] = 6 * elemnr;

            conn[6 * elemnr] = nodes[0] + k * nCoord2D;
            conn[6 * elemnr + 1] = nodes[1] + k * nCoord2D;
            conn[6 * elemnr + 2] = nodes[2] + k * nCoord2D;

            conn[6 * elemnr + 3] = nodes[0] + (k + 1) * nCoord2D;
            conn[6 * elemnr + 4] = nodes[1] + (k + 1) * nCoord2D;
            conn[6 * elemnr + 5] = nodes[2] + (k + 1) * nCoord2D;
        }
    }

    grid = new coDoUnstructuredGrid(p_mesh->getObjName(), nElem3D, nConn3D, nCoord3D, elem, conn, xCoord3D, yCoord3D, zCoord3D, tl);
    p_mesh->setCurrentObject(grid);

    polys = new coDoPolygons(p_polygons->getObjName(), nCoord2D, xCoord2D, yCoord2D, zCoord2D, nElem2D * 3, conn2D, nElem2D, elem2D);
    p_polygons->setCurrentObject(polys);

    if (p_method->getValue() == ULTRASOUND)
    {
        // do something
        sendInfo("reading ultrasound data\n");
    }
    else
    {
        // do something else ...
        sendInfo("reading radar data\n");
    }

    intensity_out->setSize((nFreqs + 1) * nCoord2D);
    p_intensity->setCurrentObject(intensity_out);

    delete[] nMeasurements;

    // free triangulator memory
    free_memory();

    return SUCCESS;
}

int *ReadIdea::get_triangles(int n, int *n_tria)
{
    edge *e_start, *e, *next;
    point *u, *v, *w;
    int i;
    point *t;

    int n_triangles = 0;

    int *triangles = new int[2 * 3 * n]; // we assume that n points do not produce more than 2*n triangles

    for (i = 0; i < n; i++)
    {
        u = &p_array[i];
        e_start = e = u->entry_pt;
        do
        {
            v = Other_point(e, u);
            if (u < v)
            {
                next = Next(e, u);
                w = Other_point(next, u);

                if (u < w)
                    if (Identical_refs(Next(next, w), Prev(e, v)))
                    {
                        // Triangle
                        if (v > w)
                        {
                            t = v;
                            v = w;
                            w = t;
                        }
                        //if (printf("%d %d %d\n", u - p_array, v - p_array, w - p_array) == EOF)
                        //    sendError("Error printing results\n");
                        if (n_triangles > 2 * n)
                        {
                            fprintf(stderr, "ooops, not enough memory allocated for triangles!\n");
                            *n_tria = n_triangles;
                            return triangles;
                        }
                        triangles[3 * n_triangles + 0] = u - p_array;
                        triangles[3 * n_triangles + 1] = v - p_array;
                        triangles[3 * n_triangles + 2] = w - p_array;
                        //cerr << "triangle " << n_triangles << ": " << u - p_array << ", " << v - p_array << ", " << w - p_array << endl;
                        n_triangles++;
                    }
            }

            // Next edge around u
            e = Next(e, u);
        } while (!Identical_refs(e, e_start));
    }

    *n_tria = n_triangles;

    return (triangles);
}

void ReadIdea::removeOuterEdges(int n)
{
    edge *e_start, *e, *e2;
    point *u, *v, *w;
    int i;
    float min_angle = cos(p_minAngle->getValue() / 180.0 * M_PI);
    bool removed;
    do
    {
        removed = false;
        for (i = 0; i < n; i++)
        {
            u = &p_array[i];
            e_start = e = u->entry_pt;
            if (Next(Next(e, u), u) != e_start) // only remove one if we have more than two edges
            {
                do
                {
                    e2 = Next(e, u);
                    v = Other_point(e, u);
                    w = Other_point(e2, u);
                    if (!isConnected(v, w)) // this edge is an outer edge
                    {
                        if (checkAndRemove(u, e, Prev(e, u), min_angle))
                        {
                            removed = true;
                            if (e == e_start)
                                break;
                        }
                        else if (checkAndRemove(u, e2, Next(e, u), min_angle))
                        {
                            removed = true;
                            if (e == e_start)
                                break;
                        }
                    }
                    e = e2;
                } while (e != e_start);
            }
        }
    } while (removed);
}

MODULE_MAIN(IO, ReadIdea)
