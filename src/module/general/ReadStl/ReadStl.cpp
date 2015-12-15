/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadSTL
//
// This module interpolates data values from Cell to Vertex
// based data representation
//
// Initial version: 2002-07-17 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:
// 18.05.2004 sl: triangles may be separated when their orientation
//                is too different. This makes it possible
//                to produce a better normal interpolation.
//                At the moment it is assumed a consistent orientation
//                of the triangles with that of their neighbours.
//                This assumption
//                is known to be violated by some exotic stl datasets.

#include "ReadStl.h"
#include <do/coDoData.h>
#include <alg/coFeatureLines.h>

#ifndef _WIN32
#include <inttypes.h>
#endif
#include <functional>

#include <iterator>

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>
//#include <util/coviseCompat.h>

// checkPoly glues together elements having nodes with the same
// coordintes. Put it another way, it suppresses nodes with the
// same coordinates. It is assumed that all polygons are triangles
// The arrays are used for input and output.
static void checkPoly(vector<int> &connList,
                      vector<float> &xcoord,
                      vector<float> &ycoord,
                      vector<float> &zcoord);

/**
 * Answer whether this is a binary file and rewind afterwards.
 * Answer false if fi is empty.
 */
static bool isBinary(FILE *fi)
{
    bool bin = false;
    int k = 0;

    if (feof(fi)) // empty
        return false;

    while (k < 1000 && !bin)
    {
        enum
        {
            CR = 13,
            NL = 10
        };
        char c = getc(fi);
        if (feof(fi))
            break;
        if (!isprint(c) && c != '\n' && c != NL && c != CR)
            bin = true;
        k++;
    }
    rewind(fi);

    return bin;
}

/**
 * Answer whether the file is empty.
 *
 * Postcondition: fi is rewinded.
 */
static bool isEmpty(FILE *fi)
{
    bool answer = true;
    rewind(fi);
    getc(fi);
    if (feof(fi))
        answer = true;
    else
        answer = false;

    rewind(fi);
    return answer;
}

// remove trailing path from filename
inline const char *coBasename(const char *str)
{
    const char *lastslash = strrchr(str, '/');
    if (lastslash)
        return lastslash + 1;
    else
        return str;
}

// Byte-Swapping for Byte-Order changes
inline void
byteSwap(int no_points, void *buffer)
{
    int i;
    unsigned int *i_buffer = (unsigned int *)(buffer);
    for (i = 0; i < no_points; i++)
    {
        unsigned &val = i_buffer[i];
        val = ((val & 0xff000000) >> 24)
              | ((val & 0x00ff0000) >> 8)
              | ((val & 0x0000ff00) << 8)
              | ((val & 0x000000ff) << 24);
    }
}

// Module set-up in Constructor
ReadStl::ReadStl(int argc, char *argv[])
    : coModule(argc, argv, "Read STL")
{
    // file browser parameter
    p_filename = addFileBrowserParam("file_path", "Data file path");
    p_filename->setValue("data/nofile.stl", "*.stl;*.STL/*");

    // Choice for File Types
    const char *choLabels[] = { "Autodetect", "ASCII", "Intel", "Unix" };
    p_format = addChoiceParam("Format", "Select STL format");
    p_format->setValue(4, choLabels, 0);

    // Choice for Coloring Types
    const char *colLabels[] = { "Autodetect", "Magics style", "VisCAM style" };
    p_colType = addChoiceParam("color_type", "Select coloring type");
    p_colType->setValue(3, colLabels, 0);

    // set color of object
    p_color = addStringParam("color", "color");
    p_color->setValue("white");

    // activate automatic FixUsg ?
    p_removeDoubleVert = addBooleanParam("RemoveDoubleVertices", "Remove double Vertices");
    p_removeDoubleVert->setValue(1);

    // show feature lines and smooth surfaces (when p_removeDoubleVert is true)
    p_showFeatureLines = addBooleanParam("ShowFeatureLines", "Show feature lines");
    p_showFeatureLines->setValue(1);

    // angle for feature edge definition
    p_angle = addFloatSliderParam("FeatureAngle", "Feature angle");
    p_angle->setValue(10.0, 30.0, 30.0);

    // flip output normals
    p_flipNormals = addBooleanParam("FlipNormals", "Flip output normals");
    p_flipNormals->setValue(0);
    
    // auto colors
    p_autoColors = addBooleanParam("AutoColors", "Automaticallz color solids (CAD colors)");
    p_autoColors->setValue(0);

    // Output ports
    p_polyOut = addOutputPort("mesh", "Polygons", "Polygons");
    p_normOut = addOutputPort("Normals", "Vec3", "velocity data");
    p_linesOut = addOutputPort("Feature_lines", "Lines", "Feature lines");

    // to be added later for coloured binary files
    p_colorOut = addOutputPort("colors", "RGBA", "color data");

    // set default values
    d_format = STL_NONE;
    d_file = NULL;
}

ReadStl::~ReadStl()
{
}

// param callback read header again after all changes
void
ReadStl::param(const char *paraName, bool inMapLoading)
{
    if (inMapLoading)
        return;

    if (strcmp(paraName, p_filename->getName()) == 0)
        readHeader();
    else if (strcmp(paraName, p_removeDoubleVert->getName()) == 0)
    {
        if (p_removeDoubleVert->getValue())
        {
            p_showFeatureLines->enable();
            if (p_showFeatureLines->getValue())
            {
                p_angle->enable();
            }
            else
            {
                p_angle->disable();
            }
        }
        else
        {
            p_showFeatureLines->disable();
            p_angle->disable();
        }
    }
    else if (strcmp(paraName, p_showFeatureLines->getName()) == 0)
    {
        if (p_showFeatureLines->getValue())
        {
            p_angle->enable();
        }
        else
        {
            p_angle->disable();
        }
    }
}

/**
 * Put the respective objects to the ports.
 *
 * We want to lump as much as possible covise calls together.
 */
void
ReadStl::outputObjects(
    vector<float> &x, vector<float> &y, vector<float> &z,
    vector<int> &connList, vector<int> &elemList,
    vector<float> &nx, vector<float> &ny, vector<float> &nz,
    vector<float> &lx, vector<float> &ly, vector<float> &lz,
    vector<int> &vl, vector<int> &ll, vector<Color> &colors)
{
    coDoPolygons *poly = new coDoPolygons(p_polyOut->getObjName(),
                                          x.size(),
                                          (x.size() > 0) ? &x[0] : NULL,
                                          (y.size() > 0) ? &y[0] : NULL,
                                          (z.size() > 0) ? &z[0] : NULL,
                                          connList.size(),
                                          (connList.size() > 0) ? &connList[0] : NULL,
                                          elemList.size(),
                                          (elemList.size() > 0) ? &elemList[0] : NULL);
    poly->addAttribute("vertexOrder", "2");
    if(p_autoColors->getValue())
    {
#define NUMCOLORS 7
        static const char *color[NUMCOLORS] = { "white", "red", "magenta", "blue", "cyan", "green", "yellow" };
	colorNumber++;
        poly->addAttribute("COLOR", color[colorNumber%NUMCOLORS]);
    }
    else
    {
    poly->addAttribute("COLOR", p_color->getValue());
    }

    coDoVec3 *norm = new coDoVec3(
        p_normOut->getObjName(),
        nx.size(),
        (nx.size() > 0) ? &nx[0] : NULL,
        (ny.size() > 0) ? &ny[0] : NULL,
        (nz.size() > 0) ? &nz[0] : NULL);
    coDoLines *lines = new coDoLines(p_linesOut->getObjName(),
                                     lx.size(),
                                     (lx.size() > 0) ? &lx[0] : NULL,
                                     (ly.size() > 0) ? &ly[0] : NULL,
                                     (lz.size() > 0) ? &lz[0] : NULL,
                                     vl.size(),
                                     (vl.size() > 0) ? &vl[0] : NULL,
                                     ll.size(),
                                     (ll.size() > 0) ? &ll[0] : NULL);

    coDoRGBA *rgba = NULL;
    if (colors.size() > 0)
    {
        rgba = new coDoRGBA(p_colorOut->getObjName(), elemList.size());
    }
    else
    {
        rgba = new coDoRGBA(p_colorOut->getObjName(), 0);
    }
    for (int i = 0; i < colors.size(); i++)
        rgba->setFloatRGBA(i, colors[i].r, colors[i].g, colors[i].b, colors[i].a);

    p_polyOut->setCurrentObject(poly);
    p_normOut->setCurrentObject(norm);
    p_linesOut->setCurrentObject(lines);
    p_colorOut->setCurrentObject(rgba);
}

// Read a binary STL file
int ReadStl::readBinary()
{
    // read Header
    char desc[80];
    if (fread(desc, 80, 1, d_file) != 1)
    {
        cerr << "ReadStl::readBinary: fread1 failed" << endl;
    }

    // replace 0 in string (viscam)
    for (int i = 0; i < 78; i++)
    {
        if (desc[i] == 0)
        {
            desc[i] = 32;
        }
    }
    // parse for COLOR information in Header
    int color_pos = string(desc).find(string("COLOR="), 0);
    Color baseColor;

    if (colType == AUTO && color_pos != string::npos)
    {
        baseColor.r = desc[color_pos + 6] / 256.;
        baseColor.g = desc[color_pos + 7] / 256.;
        baseColor.b = desc[color_pos + 8] / 256.;
        baseColor.a = desc[color_pos + 8] / 256.;

        if (string(desc).find(string("MATERIAL="), 0) != string::npos)
        {
            colType = MAGICS;
            sendInfo("Autodetected coloring type MAGICS");
        }
        else
        {
            colType = VISCAM;
            sendInfo("Autodetected coloring type VISCAM");
        }
    }
    else if (colType == AUTO)
    {
        colType = ONE;
        sendInfo("Autodetected STL as uncolored");
    }

    int numFacets;
    if (fread(&numFacets, 4, 1, d_file) != 1)
    {
        cerr << "ReadStl::readBinary: fread2 failed" << endl;
    }

    if (d_format == STL_BINARY_BS)
        byteSwap(1, &numFacets);

    // Polygon object
    vector<int> elemList, connList;
    vector<float> x, y, z;

    // Normals object
    vector<float> nx, ny, nz;

    // Color object
    vector<Color> color;

    // now read the file
    int i, j;
    Color tmpColor;

    for (i = 0; i < numFacets; i++)
    {
        if (fread(&facet, 50, 1, d_file) != 1)
        {
            cerr << "ReadStl::readBinary: fread3 failed" << endl;
        }
        if (d_format == STL_BINARY_BS)
        {
            byteSwap(12, &facet);
        }

        for (j = 0; j < 3; j++)
        {
            x.push_back(facet.vert[j].x);
            y.push_back(facet.vert[j].y);
            z.push_back(facet.vert[j].z);

            nx.push_back(facet.norm.x);
            ny.push_back(facet.norm.y);
            nz.push_back(facet.norm.z);
        }
        elemList.push_back(3 * i);

        // add reading of colors here...
        if (colType == MAGICS)
        {
            if ((facet.colors >> 15) == 0)
            {
                tmpColor.b = ((facet.colors & 0x7c00) >> 10) / 31.;
                tmpColor.g = ((facet.colors & 0x03e0) >> 5) / 31.;
                tmpColor.r = (facet.colors & 0x001f) / 31.;
                tmpColor.a = 1.;
                color.push_back(tmpColor);
            }
            else
            {
                color.push_back(baseColor);
            }
        }
        else if (colType == VISCAM)
        {
            if ((facet.colors >> 15) == 0) // materialise
            {
                tmpColor.b = ((facet.colors & 0x7c00) >> 10) / 31.;
                tmpColor.g = ((facet.colors & 0x03e0) >> 5) / 31.;
                tmpColor.r = (facet.colors & 0x001f) / 31.;
                tmpColor.a = 1.;
            }
            else
            {
                tmpColor.r = ((facet.colors & 0x7c00) >> 10) / 31.;
                tmpColor.g = ((facet.colors & 0x03e0) >> 5) / 31.;
                tmpColor.b = (facet.colors & 0x001f) / 31.;
                tmpColor.a = 1.;
            }
            color.push_back(tmpColor);
        }
    }

    // flip normals, if requested
    if (p_flipNormals->getValue())
    {
        unsigned int i;
        for (i = 0; i < nx.size(); i++)
        {
            nx[i] *= -1.0f;
            ny[i] *= -1.0f;
            nz[i] *= -1.0f;
        }
    }

    // trivial connList: all triangles are independent
    for (i = 0; i < numFacets * 3; i++)
        connList.push_back(i);

    // normals may be per vertex or per cell...
    // if per vertex -> make them cell-based!!!
    if (1 || nx.size() == 3 * elemList.size())
    {
        vector<float> l_nx, l_ny, l_nz;
        unsigned int e;
        for (e = 0; e < elemList.size(); ++e)
        {
            int c_beg = elemList[e];
            int n0 = connList[c_beg];
            l_nx.push_back(nx[n0]);
            l_ny.push_back(ny[n0]);
            l_nz.push_back(nz[n0]);
        }
        l_nx.swap(nx);
        l_ny.swap(ny);
        l_nz.swap(nz);
    }

    // lines
    vector<int> ll, vl;
    vector<float> lx, ly, lz;

    if (p_removeDoubleVert->getValue())
    {
        checkPoly(connList, x, y, z);
        if (p_showFeatureLines->getValue())
        {
            float ang_rad = (float)(M_PI * p_angle->getValue() / 180.0);
            vector<int> dll, dvl;
            vector<float> dlx, dly, dlz;
            coFeatureLines::cutPoly(cos(ang_rad), elemList, connList, x, y, z, nx, ny, nz,
                                    ll, vl, lx, ly, lz, dll, dvl, dlx, dly, dlz);
            // we proceed to produce both feature and domain lines
            // in a single object
            std::transform(dll.begin(), dll.end(), dll.begin(),
                           std::bind2nd(std::plus<int>(), vl.size()));
            std::transform(dvl.begin(), dvl.end(), dvl.begin(),
                           std::bind2nd(std::plus<int>(), lx.size()));
            std::copy(dll.begin(), dll.end(), std::back_inserter(ll));
            std::copy(dvl.begin(), dvl.end(), std::back_inserter(vl));
            std::copy(dlx.begin(), dlx.end(), std::back_inserter(lx));
            std::copy(dly.begin(), dly.end(), std::back_inserter(ly));
            std::copy(dlz.begin(), dlz.end(), std::back_inserter(lz));
        }
    }

    // group here covise-library calls
    outputObjects(x, y, z, connList, elemList,
                  nx, ny, nz,
                  lx, ly, lz, vl, ll, color);
    rewind(d_file);
    return CONTINUE_PIPELINE;
}

// taken from old ReadStl module: 2-Pass reading
int ReadStl::readASCII()
{
    if (isEmpty(d_file))
    {
        // put the empty object to port
        vector<float> x_coord, y_coord, z_coord;
        vector<int> vl, el;
        vector<float> x_n, y_n, z_n;
        vector<float> lx, ly, lz;
        vector<int> cl, ll;
        vector<Color> dummy;
        outputObjects(x_coord, y_coord, z_coord,
                      vl, el,
                      x_n, y_n, z_n,
                      lx, ly, lz,
                      cl, ll, dummy);
        return CONTINUE_PIPELINE;
    }

    char buf[600], *cbuf, tb1[600];

    
    // 1st pass: count sizes
    int n_coord = 0;
    int n_elem = 0;
    int n_normals = 0;

    while (!feof(d_file))
    {
        if (fgets(buf, sizeof(buf), d_file) == NULL)
        {
            cerr << "ReadStl::readASCII: fgets1 failed" << endl;
        }
        cbuf = buf;
        while (*cbuf == ' ' || *cbuf == '\t')
        {
            cbuf++;
        }
        if (tolower(*cbuf) == 'v')
            n_coord++;
        if (tolower(*cbuf) == 'f')
        {
            while (*cbuf != ' ' && *cbuf != '\t' && *cbuf != '\0')
            {
                cbuf++;
            }
            while (*cbuf == ' ' || *cbuf == '\t')
            {
                cbuf++;
            }
            n_elem++;
        }
        if (tolower(*cbuf) == 'n')
            n_normals++;
    }
    // 2nd pass: read actual files
    rewind(d_file);
    vector<float> x_n, y_n, z_n;
    vector<float> x_coord, y_coord, z_coord;
    vector<int> vl, el;
    vector<Color> colors;
    
    vector<Color> colorArray;
    Color c;
    c.r=1;c.g=0;c.b=0;c.a=1;
    colorArray.push_back(c);
    c.r=1;c.g=1;c.b=0;c.a=1;
    colorArray.push_back(c);
    c.r=1;c.g=1;c.b=1;c.a=1;
    colorArray.push_back(c);
    c.r=0;c.g=1;c.b=1;c.a=1;
    colorArray.push_back(c);
    c.r=0;c.g=1;c.b=0;c.a=1;
    colorArray.push_back(c);
    c.r=0;c.g=0;c.b=1;c.a=1;
    colorArray.push_back(c);
    
    n_coord = 0;
    while (!feof(d_file))
    {
        if (fgets(buf, sizeof(buf), d_file) == NULL)
        {
            cerr << "ReadStl::readASCII: fgets2 failed" << endl;
        }
        cbuf = buf;
        while (*cbuf == ' ' || *cbuf == '\t')
        {
            cbuf++;
        }
        if (tolower(*cbuf) == 's')
        {
	    colorNumber++;
	}
        if (tolower(*cbuf) == 'v')
        {
            float x, y, z;
            if (sscanf(cbuf, "%s %f %f %f", tb1, &x, &y, &z) != 4)
            {
                cerr << "ReadStl::readASCII: sscanf1 failed" << endl;
            }
            x_coord.push_back(x);
            y_coord.push_back(y);
            z_coord.push_back(z);
            vl.push_back(n_coord);
            n_coord++;
	    
	    if (p_autoColors->getValue())
	    {
	       colors.push_back(colorArray[colorNumber%colorArray.size()]);
	    }
        }
        if (tolower(*cbuf) == 'f')
        {
            while (*cbuf != ' ' && *cbuf != '\t' && *cbuf != '\0')
            {
                cbuf++;
            }
            while (*cbuf == ' ' || *cbuf == '\t')
            {
                cbuf++;
            }
            el.push_back(n_coord);
        }
        if (tolower(*cbuf) == 'n')
        {
            float fx_n, fy_n, fz_n;
            if (sscanf(cbuf, "%s %f %f %f", tb1, &fx_n, &fy_n, &fz_n) != 4)
            {
                cerr << "ReadStl::readASCII: sscanf2 failed" << endl;
            }
            x_n.push_back(fx_n);
            y_n.push_back(fy_n);
            z_n.push_back(fz_n);
        }
    }

    // this probably never happens in the ascii case!???
    if (x_n.size() == 3 * el.size())
    {
        vector<float> l_nx, l_ny, l_nz;
        unsigned int e;
        for (e = 0; e < el.size(); ++e)
        {
            int c_beg = el[e];
            int n0 = vl[c_beg];
            l_nx.push_back(x_n[n0]);
            l_ny.push_back(y_n[n0]);
            l_nz.push_back(z_n[n0]);
        }
        l_nx.swap(x_n);
        l_ny.swap(y_n);
        l_nz.swap(z_n);
    }

    // lines
    vector<int> ll, cl;
    vector<float> lx, ly, lz;

    if (p_removeDoubleVert->getValue())
    {
        checkPoly(vl, x_coord, y_coord, z_coord);
        if (p_showFeatureLines->getValue())
        {
            float ang_rad = (float)(M_PI * p_angle->getValue() / 180.0);
            vector<int> dll, dcl;
            vector<float> dlx, dly, dlz;
            coFeatureLines::cutPoly(cos(ang_rad), el, vl, x_coord, y_coord, z_coord, x_n, y_n, z_n,
                                    ll, cl, lx, ly, lz, dll, dcl, dlx, dly, dlz);
            // we proceed to produce both feature and domain lines
            // in a single object
            std::transform(dll.begin(), dll.end(), dll.begin(),
                           std::bind2nd(std::plus<int>(), cl.size()));
            std::transform(dcl.begin(), dcl.end(), dcl.begin(),
                           std::bind2nd(std::plus<int>(), lx.size()));
            std::copy(dll.begin(), dll.end(), std::back_inserter(ll));
            std::copy(dcl.begin(), dcl.end(), std::back_inserter(cl));
            std::copy(dlx.begin(), dlx.end(), std::back_inserter(lx));
            std::copy(dly.begin(), dly.end(), std::back_inserter(ly));
            std::copy(dlz.begin(), dlz.end(), std::back_inserter(lz));
        }
    }

    // flip normals, if requested
    if (p_flipNormals->getValue())
    {
        unsigned int i;
        for (i = 0; i < x_n.size(); i++)
        {
            x_n[i] *= -1.0f;
            y_n[i] *= -1.0f;
            z_n[i] *= -1.0f;
        }
    }

#if 0
   cerr << x_coord.size() << endl;
   cerr << y_coord.size() << endl;
   cerr << z_coord.size() << endl;
   cerr << vl.size() << endl;
   cerr << el.size() << endl;
   cerr << x_n.size() << endl;
   cerr << y_n.size() << endl;
   cerr << z_n.size() << endl;
   cerr << cl.size() << endl;
   cerr << ll.size() << endl;
#endif
    // group here covise-library calls
    outputObjects(x_coord, y_coord, z_coord, vl, el,
                  x_n, y_n, z_n,
                  lx, ly, lz, cl, ll, colors);

    rewind(d_file);
    return CONTINUE_PIPELINE;
}

int ReadStl::compute(const char *)
{
    colorNumber=0;
    // not opened any file yet ?
    if (!d_file || d_format == STL_NONE)
        readHeader();

    // Now, this must be an error:
    //     No message, readHeader already cries if problems occur
    if (!d_file || d_format == STL_NONE)
        return STOP_PIPELINE;

    colType = (ColType)p_colType->getValue();

    int result;
    if (d_format == STL_ASCII)
        result = readASCII();
    else
        result = readBinary();

    // add filename as an attribute
    coDistributedObject *obj = p_polyOut->getCurrentObject();
    if (obj)
    {
        std::string filename(p_filename->getValue());
        if (filename.length() > 40)
        {
            size_t pos = filename.rfind("/");
            if (pos != std::string::npos)
            {
                filename = "..." + filename.substr(pos);
            }
        }
#if 0
        pos = filename.rfind(".");
        if (pos != std::string::npos)
        {
            filename = filename.substr(0, pos);
        }
#endif
        obj->addAttribute("OBJECTNAME", filename.c_str());
    }

    return result;
}

// utility functions
void ReadStl::readHeader()
{
    if (d_file)
        fclose(d_file);
    d_format = STL_NONE;

    const char *fileName = p_filename->getValue();

    // Try to open file
    d_file = fopen(fileName, "rb");
    if (!d_file)
    {
        sendError("Could not read %s: %s", fileName, strerror(errno));
        return;
    }

    // skip over header and read number of facets
    int numFacets;

    struct stat statRec;
    fstat(fileno(d_file), &statRec);

    // Binary: find byte-order
    switch (p_format->getValue())
    {
    case 0: // AUTO: try to guess
    {
        if (isBinary(d_file))
        {
            fseek(d_file, 80, SEEK_SET);
            if (fread(&numFacets, 4, 1, d_file) != 1)
            {
                cerr << "ReadStl::readHader: fread1 failed" << endl;
            }

            if (numFacets > 10000000 || numFacets < 0)
            {
                byteSwap(1, &numFacets);
                d_format = STL_BINARY_BS;
                sendInfo("Autodetected '%s' as binary STL, byte-swap required",
                         coBasename(fileName));
            }
            else
            {
                d_format = STL_BINARY;
                sendInfo("Autodetected '%s' as binary STL, no byte-swap required",
                         coBasename(fileName));
            }
        }
        else
        {
            d_format = STL_ASCII;
            sendInfo("Autodetected '%s' as ASCII STL", coBasename(fileName));
            return; // do not check sizes - not possible for ASCII
        }
        break;
    }

    case 1: // User set: Ascii
    {
        d_format = STL_ASCII;
        return;
    }

    case 2: // User set: Intel byte-Order
    {
        fseek(d_file, 80, SEEK_SET);
        if (fread(&numFacets, 4, 1, d_file) != 1)
        {
            cerr << "ReadStl::readHader: fread3 failed" << endl;
        }
#ifdef BYTESWAP
        d_format = STL_BINARY;
#else
        d_format = STL_BINARY_BS;
        byteSwap(1, &numFacets);
#endif
        break;
    }

    case 3: // User set: Unix byte-Order
    {
        fseek(d_file, 80, SEEK_SET);
        if (fread(&numFacets, 4, 1, d_file) != 1)
        {
            cerr << "ReadStl::readHader: fread4 failed" << endl;
        }
#ifdef BYTESWAP
        d_format = STL_BINARY_BS;
        byteSwap(1, &numFacets);
#else
        d_format = STL_BINARY;
#endif
        break;
    }

    default:
    {
        sendError("Choice %s returned strange value", p_format->getName());
        d_format = STL_NONE;
        fclose(d_file);
        return;
    }
    }

    /// Important: at least try to sort out buggy files: no "magic" in file
    if (statRec.st_size != numFacets * 50 + 84)
    {
        d_format = STL_NONE;
        fclose(d_file);
        sendError("File %s no STL file, other format or truncated", fileName);
        return;
    }

    rewind(d_file);
    return;
}

/////////////////////////////////////////////////////////////////////
//
//   CheckUSG auf internen Feldern -> es kann nur kleiner werden...
//   Code von CellToVert kopiert.
//
//   Returns Map newCellId [ oldCellID ] -> to be deleted by prog
//
/////////////////////////////////////////////////////////////////////

/////// find the bounding box

inline void boundingBox(const vector<float> &x,
                        const vector<float> &y,
                        const vector<float> &z,
                        int *c, int n,
                        float *bbx1, float *bby1, float *bbz1,
                        float *bbx2, float *bby2, float *bbz2)
{
    int i;
    float cx, cy, cz;

    *bbx1 = *bbx2 = x[c[0]];
    *bby1 = *bby2 = y[c[0]];
    *bbz1 = *bbz2 = z[c[0]];

    for (i = 0; i < n; i++)
    {
        cx = x[c[i]];
        cy = y[c[i]];
        cz = z[c[i]];

        // x
        if (cx < *bbx1)
            *bbx1 = cx;
        else if (cx > *bbx2)
            *bbx2 = cx;

        // y
        if (cy < *bby1)
            *bby1 = cy;
        else if (cy > *bby2)
            *bby2 = cy;

        // z
        if (cz < *bbz1)
            *bbz1 = cz;
        else if (cz > *bbz2)
            *bbz2 = cz;
    }
    return;
}

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////

inline int getOctant(float x, float y, float z, float ox, float oy, float oz)
{
    int r;

    // ... the origin

    if (x < ox) // behind

        if (y < oy) // below
            if (z < oz) // left
                r = 6;
            else // right
                r = 7;
        else // above
            if (z < oz) // left
            r = 4;
        else // right
            r = 5;

    else // in front of
        if (y < oy) // below
        if (z < oz) // left
            r = 2;
        else // right
            r = 3;
    else // above
        if (z < oz) // left
        r = 0;
    else // right
        r = 1;

    // done
    return (r);
}

inline void getOctantBounds(int o, float ox, float oy, float oz,
                            float bbx1, float bby1, float bbz1,
                            float bbx2, float bby2, float bbz2,
                            float *bx1, float *by1, float *bz1,
                            float *bx2, float *by2, float *bz2)
{
    switch (o)
    {
    case 0: // left, above, front
        *bx1 = bbx1;
        *by1 = oy;
        *bz1 = oz;
        *bx2 = ox;
        *by2 = bby2;
        *bz2 = bbz2;
        break;
    case 1: // right, above, front
        *bx1 = ox;
        *by1 = oy;
        *bz1 = oz;
        *bx2 = bbx2;
        *by2 = bby2;
        *bz2 = bbz2;
        break;
    case 2: // left, below, front
        *bx1 = bbx1;
        *by1 = bby1;
        *bz1 = oz;
        *bx2 = ox;
        *by2 = oy;
        *bz2 = bbz2;
        break;
    case 3: // right, below, front
        *bx1 = ox;
        *by1 = bby1;
        *bz1 = oz;
        *bx2 = bbx2;
        *by2 = oy;
        *bz2 = bbz2;
        break;
    case 4: // left, above, behind
        *bx1 = bbx1;
        *by1 = oy;
        *bz1 = bbz1;
        *bx2 = ox;
        *by2 = bby2;
        *bz2 = oz;
        break;
    case 5: // right, above, behind
        *bx1 = ox;
        *by1 = oy;
        *bz1 = bbz1;
        *bx2 = bbx2;
        *by2 = bby2;
        *bz2 = oz;
        break;
    case 6: // left, below, behind
        *bx1 = bbx1;
        *by1 = bby1;
        *bz1 = bbz1;
        *bx2 = ox;
        *by2 = oy;
        *bz2 = oz;
        break;
    case 7: // right, below, behind
        *bx1 = ox;
        *by1 = bby1;
        *bz1 = bbz1;
        *bx2 = bbx2;
        *by2 = oy;
        *bz2 = oz;
        break;
    }

    return;
}

void computeCell(const vector<float> &xcoord,
                 const vector<float> &ycoord,
                 const vector<float> &zcoord,
                 int *coordInBox, int numCoordInBox,
                 float bbx1, float bby1, float bbz1,
                 float bbx2, float bby2, float bbz2,
                 int maxCoord, int *replBy, int &numCoordToRemove)
{
    int i, j;
    int v, w;
    float rx, ry, rz;
    float obx1 = 0.f, oby1 = 0.f, obz1 = 0.f;
    float obx2 = 0.f, oby2 = 0.f, obz2 = 0.f;
    int numCoordInCell[8];
    int *coordInCell[8];

    // too many Coords in my box -> split octree box deeper
    if (numCoordInBox > maxCoord)
    {
        // yes we have
        rx = (bbx1 + bbx2) / 2.0f;
        ry = (bby1 + bby2) / 2.0f;
        rz = (bbz1 + bbz2) / 2.0f;

        // go through the coordinates and sort them in the right cell

        for (i = 0; i < 8; i++)
        {
            coordInCell[i] = new int[numCoordInBox];
            numCoordInCell[i] = 0;
        }

        for (i = 0; i < numCoordInBox; i++)
        {
            v = coordInBox[i];
            w = getOctant(xcoord[v], ycoord[v], zcoord[v], rx, ry, rz);
            coordInCell[w][numCoordInCell[w]] = v;
            numCoordInCell[w]++;
        }

        // we're recursive - hype
        for (i = 0; i < 8; i++)
        {
            if (numCoordInCell[i])
            {
                if (numCoordInCell[i] > numCoordInBox / 4)
                {
                    // we decide to compute the new BoundingBox instead of
                    // just splitting the parent-Box
                    boundingBox(xcoord, ycoord, zcoord, coordInCell[i],
                                numCoordInCell[i], &obx1, &oby1, &obz1,
                                &obx2, &oby2, &obz2);
                }
                else
                    getOctantBounds(i, rx, ry, rz, bbx1, bby1, bbz1,
                                    bbx2, bby2, bbz2,
                                    &obx1, &oby1, &obz1, &obx2, &oby2, &obz2);

                computeCell(xcoord, ycoord, zcoord, coordInCell[i],
                            numCoordInCell[i], obx1, oby1, obz1,
                            obx2, oby2, obz2,
                            (numCoordInCell[i] == numCoordInBox) ? numCoordInCell[i] : maxCoord,
                            replBy, numCoordToRemove);
            }
            delete[] coordInCell[i];
        }
    }

    //// directly compare in box
    else if (numCoordInBox > 1)
    {
        // check these vertices
        for (i = 0; i < numCoordInBox - 1; i++)
        {
            v = coordInBox[i];
            rx = xcoord[v];
            ry = ycoord[v];
            rz = zcoord[v];
            // see if this one is doubled
            for (j = i + 1; j < numCoordInBox; j++)
            {
                w = coordInBox[j];

                if (xcoord[w] == rx && // @@@@ add distance fkt here if necessary
                    ycoord[w] == ry && zcoord[w] == rz)
                {
                    // this one is double
                    if (v < w)
                        replBy[w] = v;
                    else
                        replBy[v] = w;
                    numCoordToRemove++;
                    // break out
                    j = numCoordInBox;
                }
            }
        }
    }

    // done
    return;
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

static void computeReplaceLists(int num_coord, int *replBy,
                                int *&src2filt, int *&filt2src)
{
    int i, k;

    // now unlink the temporary list
    for (i = 0; i < num_coord; i++)
    {
        k = replBy[i];
        if (k >= 0)
        {
            // this one will be replaced, so unlink the list
            while (replBy[k] >= 0)
                k = replBy[k];

            if (replBy[k] == -1)
                // remove this one
                replBy[i] = -1;
            else
                // replace this one
                replBy[i] = k;
        }
    }

    // allocate mem
    src2filt = new int[num_coord]; // original vertex i is replaced by s2f[i]

    // forward filter
    int numFiltered = 0;
    for (i = 0; i < num_coord; i++)
    {
        // vertex untouched
        if (replBy[i] == -2)
        {
            src2filt[i] = numFiltered;
            numFiltered++;
        }
        // vertex replaced: replacer < replacee
        else if (replBy[i] >= 0)
        {
            src2filt[i] = src2filt[replBy[i]];
        }
        else
        {
            src2filt[i] = -1;
        }
    }

    // backward filter
    filt2src = new int[numFiltered];
    for (i = 0; i < num_coord; i++)
        if (src2filt[i] >= 0)
            filt2src[src2filt[i]] = i;

    // done
    return;
}

static void checkPoly(vector<int> &connList,
                      vector<float> &xcoord,
                      vector<float> &ycoord,
                      vector<float> &zcoord)
{

    int i, num_conn, num_coord;

    num_conn = connList.size();
    num_coord = xcoord.size();

    /// create a Replace-List
    int *replBy = new int[num_coord];
    for (i = 0; i < num_coord; i++)
        replBy[i] = -2;

    int numCoordToRemove = 0;

    int *coordInBox = new int[num_coord];
    int numCoordInBox = 0;

    // the "starting" cell contains all USED coordinates
    // clear all flags -> no coordinates used at all
    for (i = 0; i < num_coord; i++)
        coordInBox[i] = 0;

    for (i = 0; i < num_conn; i++)
        coordInBox[connList[i]] = 1;

    // now remove the unused coordinates
    for (i = 0; i < num_coord; i++)
    {
        if (coordInBox[i])
        {
            // this one is used
            coordInBox[numCoordInBox] = i;
            numCoordInBox++;
        }
        else
        {
            // unused coordinate
            replBy[i] = -1;
            numCoordToRemove++;
        }
    }

    float bbx1, bby1, bbz1;
    float bbx2, bby2, bbz2;

    // find the bounding box
    boundingBox(xcoord, ycoord, zcoord, coordInBox, numCoordInBox,
                &bbx1, &bby1, &bbz1, &bbx2, &bby2, &bbz2);

    const int maxCoord = 50; // elements for direct sort

    computeCell(xcoord, ycoord, zcoord,
                coordInBox, numCoordInBox, bbx1, bby1, bbz1,
                bbx2, bby2, bbz2, maxCoord, replBy, numCoordToRemove);

    // partially clean up
    delete[] coordInBox;

    // obly if we found vertices to remove...
    if (numCoordToRemove)
    {
        // compute the lists of replacements (both directions)
        int *src2filt, *filt2src;
        computeReplaceLists(num_coord, replBy, src2filt, filt2src);
        delete[] replBy;
        replBy = NULL;

        ////// ---------- Filter Grid ----------
        int newNumCoord = num_coord - numCoordToRemove;
        int newIdx = 0;

        // skip thru first changed idx (we DO replace at least one, see above)
        while (filt2src[newIdx] == newIdx)
            newIdx++;

        // and now replace coordinates
        vector<float> l_xcoord(xcoord.begin(), xcoord.begin() + newIdx);
        vector<float> l_ycoord(ycoord.begin(), ycoord.begin() + newIdx);
        vector<float> l_zcoord(zcoord.begin(), zcoord.begin() + newIdx);

        do
        {
            int oldIdx = filt2src[newIdx];
            l_xcoord.push_back(xcoord[oldIdx]);
            l_ycoord.push_back(ycoord[oldIdx]);
            l_zcoord.push_back(zcoord[oldIdx]);
            newIdx++;
        } while (newIdx < newNumCoord);

        l_xcoord.swap(xcoord);
        l_ycoord.swap(ycoord);
        l_zcoord.swap(zcoord);

        // and replace connectivities
        vector<int> l_connList(num_conn);
        for (i = 0; i < num_conn; i++)
            l_connList[i] = src2filt[connList[i]];
        l_connList.swap(connList);

        // clean up
        delete[] src2filt;
        delete[] filt2src;
    }
    delete[] replBy;
}

MODULE_MAIN(IO, ReadStl)
