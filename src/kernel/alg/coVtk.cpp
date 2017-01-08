/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef HAVE_VTK
#include <vtkVersion.h>
#include <vtkDataSet.h>
#include <vtkDataArray.h>
#include <vtkDataSetAttributes.h>

#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkCharArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkShortArray.h>
#include <vtkUnsignedShortArray.h>
#include <vtkIntArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkIdTypeArray.h>

#include <vtkAlgorithm.h>
#include <vtkInformation.h>
#include <vtkUnstructuredGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredPoints.h>
#include <vtkImageData.h>
#include <vtkUniformGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredPoints.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>

#include <vtkCompositeDataSet.h>
#if VTK_MAJOR_VERSION < 6
#include <vtkTemporalDataSet.h>
#define HAVE_VTK_TEMP
#endif
#include <vtkMultiPieceDataSet.h>
#endif

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoGeometry.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>
#include <do/coDoSet.h>

#include "coVtk.h"

using namespace covise;

#define CHECK_AND_RETURN(obj)                                                               \
    if (!obj->objectOk())                                                                   \
    {                                                                                       \
        std::cerr << __FILE__ << ":" << __LINE__ << ": " << "objectOk failed" << std::endl; \
        delete obj;                                                                         \
        return NULL;                                                                        \
    }

#ifdef HAVE_VTK
static coDoGrid *vtkUGrid2Covise(const coObjInfo &info, vtkUnstructuredGrid *vugrid)
{
    int ncoord = vugrid->GetNumberOfPoints();
    int nelem = vugrid->GetNumberOfCells();
    vtkCellArray *vcellarray = vugrid->GetCells();
    int nconn = vcellarray->GetNumberOfConnectivityEntries() - nelem;

    coDoUnstructuredGrid *cugrid = new coDoUnstructuredGrid(info, nelem, nconn, ncoord, 1);
    CHECK_AND_RETURN(cugrid);
    float *xc, *yc, *zc;
    int *elems, *connlist, *typelist;
    cugrid->getAddresses(&elems, &connlist, &xc, &yc, &zc);
    cugrid->getTypeList(&typelist);

    for (int i = 0; i < ncoord; ++i)
    {
        xc[i] = vugrid->GetPoint(i)[0];
        yc[i] = vugrid->GetPoint(i)[1];
        zc[i] = vugrid->GetPoint(i)[2];
    }

    vtkUnsignedCharArray *vtypearray = vugrid->GetCellTypesArray();
    for (int i = 0; i < nelem; ++i)
    {
        switch (vtypearray->GetValue(i))
        {
        case VTK_VERTEX:
        case VTK_POLY_VERTEX:
            typelist[i] = TYPE_POINT;
            break;
        case VTK_LINE:
        case VTK_POLY_LINE:
            typelist[i] = TYPE_BAR;
            break;
        case VTK_TRIANGLE:
            typelist[i] = TYPE_TRIANGLE;
            break;
        case VTK_QUAD:
            typelist[i] = TYPE_QUAD;
            break;
        case VTK_TETRA:
            typelist[i] = TYPE_TETRAHEDER;
            break;
        case VTK_HEXAHEDRON:
            typelist[i] = TYPE_HEXAEDER;
            break;
        case VTK_WEDGE:
            typelist[i] = TYPE_PRISM;
            break;
        case VTK_PYRAMID:
            typelist[i] = TYPE_PYRAMID;
            break;
        case VTK_POLYHEDRON:
            typelist[i] = TYPE_POLYHEDRON;
            break;
        default:
            typelist[i] = 0;
            break;
        }
    }

    vcellarray->InitTraversal();
    int k = 0;
    for (int i = 0; i < nelem; ++i)
    {
        elems[i] = k;

        vtkIdType npts = 0;
        vtkIdType *pts = NULL;
        vcellarray->GetNextCell(npts, pts);
        if (typelist[i] == TYPE_POLYHEDRON && npts > 0)
        {
            int j=0;
            int nfaces = pts[j];
            ++j;
            for (int f=0; f<nfaces; ++f)
            {
                int nvert = pts[j];
                connlist[k] = pts[j+nvert];
                ++k;
                ++j;
                for (int n=0; n<nvert; ++n)
                {
                    connlist[k] = pts[j];
                    ++k;
                    ++j;
                }
            }
        }
        else
        {
            for (int j = 0; j < npts; ++j)
            {
                connlist[k] = pts[j];
                ++k;
            }
        }
    }

    return cugrid;
}

static vtkUnstructuredGrid *coviseUGrid2Vtk(const coDoUnstructuredGrid *ugrid)
{
    vtkUnstructuredGrid *vugrid = vtkUnstructuredGrid::New();
    int ncoord, nelem, nconn;
    ugrid->getGridSize(&nelem, &nconn, &ncoord);
    float *x, *y, *z;
    int *connlist, *celllist;
    ugrid->getAddresses(&celllist, &connlist, &x, &y, &z);
    vtkPoints *points = vtkPoints::New();
    for (int i = 0; i < ncoord; ++i)
    {
        float p[3] = { x[i], y[i], z[i] };
        points->InsertPoint(i, p);
    }
    vugrid->SetPoints(points);

    int *typelist;
    ugrid->getTypeList(&typelist);

    for (int i = 0; i < nelem; ++i)
    {
        //int wish = 0;
        int type = 0;
        switch (typelist[i])
        {
        case TYPE_POINT:
            type = VTK_VERTEX;
            //wish = 1;
            break;
        case TYPE_BAR:
            type = VTK_LINE;
            //wish=2;
            break;
        case TYPE_TRIANGLE:
            type = VTK_TRIANGLE;
            //wish=3;
            break;
        case TYPE_QUAD:
            type = VTK_QUAD;
            //wish=4;
            break;
        case TYPE_TETRAHEDER:
            type = VTK_TETRA;
            //wish=4;
            break;
        case TYPE_HEXAEDER:
            type = VTK_HEXAHEDRON;
            //wish=8;
            break;
        case TYPE_PRISM:
            type = VTK_WEDGE;
            //wish=6;
            break;
        case TYPE_PYRAMID:
            type = VTK_PYRAMID;
            //wish=5;
            break;
        default:
            type = 0;
            //wish=0;
            fprintf(stderr, "coVtk::coviseUGrid2Vtk: unhandled cell type: %d\n", typelist[i]);
            break;
        }

        int cellstart = celllist[i];
        vtkIdType celllen = (i + 1 == nelem ? nconn : celllist[i + 1]) - cellstart;
        std::vector<vtkIdType> ids(celllen);
        for (int i = 0; i < celllen; ++i)
            ids[i] = connlist[cellstart + i];
        vugrid->InsertNextCell(type, celllen, &ids[0]);
    }

    return vugrid;
}

static coDoGrid *vtkImage2Covise(const coObjInfo &info, vtkImageData *vimage)
{
    int n[3];
    vimage->GetDimensions(n);
    double origin[3] = { 0., 0., 0. };
    vimage->GetOrigin(origin);
    double spacing[3] = { 1., 1., 1. };
    vimage->GetSpacing(spacing);
    coDoUniformGrid *ug = new coDoUniformGrid(info, n[0], n[1], n[2],
                                              origin[0], spacing[0] * (n[0] - 1) + origin[0],
                                              origin[1], spacing[1] * (n[1] - 1) + origin[1],
                                              origin[2], spacing[2] * (n[2] - 1) + origin[2]);
    CHECK_AND_RETURN(ug);
    return ug;
}

static coDoGrid *vtkUniGrid2Covise(const coObjInfo &info, vtkUniformGrid *vugrid)
{
    int n[3];
    vugrid->GetDimensions(n);
    double origin[3];
    vugrid->GetOrigin(origin);
    double spacing[3];
    vugrid->GetSpacing(spacing);
    coDoUniformGrid *ug = new coDoUniformGrid(info, n[0], n[1], n[2],
                                              origin[0], spacing[0] * (n[0] - 1) + origin[0],
                                              origin[1], spacing[1] * (n[1] - 1) + origin[1],
                                              origin[2], spacing[2] * (n[2] - 1) + origin[2]);
    CHECK_AND_RETURN(ug);
    return ug;
}

static vtkUniformGrid *coviseUniGrid2Vtk(const coDoUniformGrid *ugrid)
{
    vtkUniformGrid *vugrid = vtkUniformGrid::New();

    int n[3];
    ugrid->getGridSize(&n[0], &n[1], &n[2]);
    vugrid->SetDimensions(n);

    float min[3], max[3];
    ugrid->getMinMax(&min[0], &max[0], &min[1], &max[1], &min[2], &max[2]);
    double spacing[3];
    for (int i = 0; i < 3; ++i)
    {
        if (max[i] < min[i])
            std::swap(max[i], min[i]);
        spacing[i] = (max[i] - min[i]) / n[i];
    }
    vugrid->SetOrigin(min[0], min[1], min[2]);
    vugrid->SetSpacing(spacing);

    return vugrid;
}

coDoPixelImage *coVtk::vtkImage2Covise(const coObjInfo &info, vtkImageData *vtk)
{
    int n[3];
    vtk->GetDimensions(n);
    if (n[2] != 1)
    {
        fprintf(stderr, "coVtk::vtkImage2Covise: only using first of %d slices\n", n[2]);
    }
    int nc = vtk->GetNumberOfScalarComponents();
    coDoPixelImage *pix = new coDoPixelImage(info, n[0], n[1], nc, nc);
    CHECK_AND_RETURN(pix);
    int k = 0;
    for (int y = 0; y < n[1]; ++y)
        for (int x = 0; x < n[0]; ++x)
            for (int c = 0; c < nc; ++c)
            {
                (*pix)[k++] = (char)(vtk->GetScalarComponentAsFloat(x, y, 0, c) * 255.99f);
            }

    return pix;
}

static vtkImageData *coviseImage2Vtk(const coDoPixelImage *img)
{
    vtkImageData *vimg = vtkImageData::New();
    int w = img->getWidth();
    int h = img->getHeight();
    vimg->SetDimensions(w, h, 1);

    int nc = img->getPixelsize();
#if VTK_MAJOR_VERSION < 6
    vimg->SetNumberOfScalarComponents(nc);
#endif
    const unsigned char *p = reinterpret_cast<const unsigned char *>(&(*img)[0]);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            for (int c = 0; c < nc; ++c)
            {
                vimg->SetScalarComponentFromFloat(x, y, 0, c, p[(y * w + x) * nc + c] / 255.f);
            }
        }
    }

    return vimg;
}

static vtkImageData *coviseTexture2Vtk(const coDoTexture *tex)
{
    return coviseImage2Vtk(tex->getBuffer());
}

static coDoGrid *vtkRGrid2Covise(const coObjInfo &info, vtkRectilinearGrid *vrgrid)
{
    int n[3];
    vrgrid->GetDimensions(n);

    coDoRectilinearGrid *rgrid = new coDoRectilinearGrid(info, n[0], n[1], n[2]);
    CHECK_AND_RETURN(rgrid);
    float *c[3];
    rgrid->getAddresses(&c[0], &c[1], &c[2]);

    vtkDataArray *vx = vrgrid->GetXCoordinates();
    for (int i = 0; i < n[0]; ++i)
        c[0][i] = vx->GetTuple1(i);

    vtkDataArray *vy = vrgrid->GetYCoordinates();
    for (int i = 0; i < n[1]; ++i)
        c[1][i] = vy->GetTuple1(i);

    vtkDataArray *vz = vrgrid->GetZCoordinates();
    for (int i = 0; i < n[2]; ++i)
        c[2][i] = vz->GetTuple1(i);

    return rgrid;
}

static vtkRectilinearGrid *coviseRGrid2Vtk(const coDoRectilinearGrid *rgrid)
{
    vtkRectilinearGrid *vrgrid = vtkRectilinearGrid::New();

    int n[3];
    rgrid->getGridSize(&n[0], &n[1], &n[2]);
    vrgrid->SetDimensions(n);
    float *coord[3];
    rgrid->getAddresses(&coord[0], &coord[1], &coord[2]);

    vtkFloatArray *vx = vtkFloatArray::New();
    vx->SetArray(coord[0], n[0], 1);
    vrgrid->SetXCoordinates(vx);
    vtkFloatArray *vy = vtkFloatArray::New();
    vy->SetArray(coord[1], n[1], 1);
    vrgrid->SetXCoordinates(vy);
    vtkFloatArray *vz = vtkFloatArray::New();
    vz->SetArray(coord[2], n[2], 1);
    vrgrid->SetXCoordinates(vz);

    return vrgrid;
}

static coDoGrid *vtkSGrid2Covise(const coObjInfo &info, vtkStructuredGrid *vsgrid)
{
    int dim[3];
    vsgrid->GetDimensions(dim);

    coDoStructuredGrid *csgrid = new coDoStructuredGrid(info, dim[0], dim[1], dim[2]);
    CHECK_AND_RETURN(csgrid);
    float *xc, *yc, *zc;
    csgrid->getAddresses(&xc, &yc, &zc);

    int l = 0;
    for (int i = 0; i < dim[0]; ++i)
    {
        for (int j = 0; j < dim[1]; ++j)
        {
            for (int k = 0; k < dim[2]; ++k)
            {
                int idx = k * (dim[0] * dim[1]) + j * dim[0] + i;
                xc[l] = vsgrid->GetPoint(idx)[0];
                yc[l] = vsgrid->GetPoint(idx)[1];
                zc[l] = vsgrid->GetPoint(idx)[2];
                ++l;
            }
        }
    }

    return csgrid;
}

static vtkStructuredGrid *coviseSGrid2Vtk(const coDoStructuredGrid *sgrid)
{
    vtkStructuredGrid *vsgrid = vtkStructuredGrid::New();

    int dim[3];
    sgrid->getGridSize(&dim[0], &dim[1], &dim[2]);
    vsgrid->SetDimensions(dim);

    //int ncoord = dim[0]*dim[1]*dim[2];
    float *x, *y, *z;
    sgrid->getAddresses(&x, &y, &z);
    vtkPoints *points = vtkPoints::New();
    int l = 0;
    for (int i = 0; i < dim[0]; ++i)
    {
        for (int j = 0; j < dim[1]; ++j)
        {
            for (int k = 0; k < dim[2]; ++k)
            {
                int idx = k * (dim[0] * dim[1]) + j * dim[0] + i;
                float p[3] = { x[l], y[l], z[l] };
                points->InsertPoint(idx, p);
                ++l;
            }
        }
    }
    vsgrid->SetPoints(points);

    return vsgrid;
}

static coDoGrid *vtkPoly2Covise(const coObjInfo &info, vtkPolyData *vpolydata)
{
    int ncoord = vpolydata->GetNumberOfPoints();
    int npolys = vpolydata->GetNumberOfPolys();
    int nstrips = vpolydata->GetNumberOfStrips();
    int nlines = vpolydata->GetNumberOfLines();
    int nverts = vpolydata->GetNumberOfVerts();

    coDoGrid *geo = NULL;

    float *xc = NULL, *yc = NULL, *zc = NULL;
    if (npolys == 0 && nstrips > 0)
    {
        vtkCellArray *strips = vpolydata->GetStrips();
        int ncorner = strips->GetNumberOfConnectivityEntries() - nstrips;
        coDoTriangleStrips *cstrips = new coDoTriangleStrips(info, ncoord, ncorner, nstrips);
        CHECK_AND_RETURN(cstrips);
        geo = cstrips;

        int *cornerlist, *striplist;
        cstrips->getAddresses(&xc, &yc, &zc, &cornerlist, &striplist);

        int k = 0;
        strips->InitTraversal();
        vtkIdType npts = 0, *pts = NULL;
        for (int i = 0; i < nstrips; ++i)
        {
            if (i == 0)
                striplist[0] = 0;
            else
                striplist[i] = striplist[i - 1] + npts;

            strips->GetNextCell(npts, pts);
            for (int j = 0; j < npts; ++j)
            {
                cornerlist[k] = pts[j];
                ++k;
            }
        }
    }
    else if (npolys > 0)
    {
        int nstriptris = 0;
        vtkCellArray *strips = vpolydata->GetStrips();
        strips->InitTraversal();
        for (int i = 0; i < nstrips; ++i)
        {
            vtkIdType npts = 0, *pts = NULL;
            strips->GetNextCell(npts, pts);
            nstriptris += npts - 2;
        }

        vtkCellArray *polys = vpolydata->GetPolys();
        int ncorner = polys->GetNumberOfConnectivityEntries() - npolys + 3 * nstriptris;
        coDoPolygons *cpoly = new coDoPolygons(info, nstriptris+npolys, ncorner, ncoord);
        geo = cpoly;

        int *cornerlist, *polylist;
        cpoly->getAddresses(&xc, &yc, &zc, &cornerlist, &polylist);

        int k = 0;
        polys->InitTraversal();
        for (int i = 0; i < npolys; ++i)
        {
            polylist[i] = k;

            vtkIdType npts = 0, *pts = NULL;
            polys->GetNextCell(npts, pts);
            for (int j = 0; j < npts; ++j)
            {
                cornerlist[k] = pts[j];
                ++k;
            }
        }

        strips->InitTraversal();
        for (int i = 0; i < nstrips; ++i)
        {
            polylist[npolys + i] = k;
            vtkIdType npts = 0, *pts = NULL;
            strips->GetNextCell(npts, pts);
            for (int j = 0; j < npts - 2; ++j)
            {
                if (j % 2)
                {
                    cornerlist[k++] = pts[j];
                    cornerlist[k++] = pts[j + 1];
                }
                else
                {
                    cornerlist[k++] = pts[j + 1];
                    cornerlist[k++] = pts[j];
                }
                cornerlist[k++] = pts[j + 2];
            }
        }
    }
    else if (nlines > 0)
    {
        vtkCellArray *lines = vpolydata->GetLines();
        int ncorner = lines->GetNumberOfConnectivityEntries() - nlines;
        coDoLines *clines = new coDoLines(info, ncoord, ncorner, nlines);
        CHECK_AND_RETURN(clines);
        geo = clines;

        int *cornerlist, *linelist;
        clines->getAddresses(&xc, &yc, &zc, &cornerlist, &linelist);

        int k = 0;
        lines->InitTraversal();
        for (int i = 0; i < nlines; ++i)
        {
            linelist[i] = k;

            vtkIdType npts = 0, *pts = NULL;
            lines->GetNextCell(npts, pts);
            for (int j = 0; j < npts; ++j)
            {
                cornerlist[k] = pts[j];
                ++k;
            }
        }
    }
    else if (nverts > 0)
    {
        if (nverts != ncoord)
            return NULL;
        coDoPoints *cpoints = new coDoPoints(info, ncoord);
        CHECK_AND_RETURN(cpoints);
        geo = cpoints;
        cpoints->getAddresses(&xc, &yc, &zc);
    }

    if (xc && yc && zc)
    {
        for (int i = 0; i < ncoord; ++i)
        {
            xc[i] = vpolydata->GetPoint(i)[0];
            yc[i] = vpolydata->GetPoint(i)[1];
            zc[i] = vpolydata->GetPoint(i)[2];
        }
    }

    return geo;
}

static vtkPolyData *coviseTris2Vtk(const coDoTriangleStrips *ctris)
{
    vtkPolyData *vpoly = vtkPolyData::New();

    int ncoord = ctris->getNumPoints();
    int nstrips = ctris->getNumStrips();
    float *x, *y, *z;
    int *vertexlist, *striplist;
    ctris->getAddresses(&x, &y, &z, &vertexlist, &striplist);

    vtkPoints *points = vtkPoints::New();
    for (int i = 0; i < ncoord; ++i)
    {
        float p[3] = { x[i], y[i], z[i] };
        points->InsertPoint(i, p);
    }
    vpoly->SetPoints(points);

    vtkCellArray *strips = vtkCellArray::New();
    for (int i = 0; i < nstrips; ++i)
    {
        int stripstart = striplist[i];
        vtkIdType striplen = (i + 1 == nstrips ? ctris->getNumVertices() : striplist[i + 1]) - stripstart;
        std::vector<vtkIdType> ids(striplen);
        for (int i = 0; i < striplen; ++i)
            ids[i] = vertexlist[stripstart + i];
        strips->InsertNextCell(striplen, &ids[0]);
    }
    vpoly->SetStrips(strips);

    return vpoly;
}

static vtkPolyData *covisePoly2Vtk(const coDoPolygons *cpoly)
{
    vtkPolyData *vpoly = vtkPolyData::New();
    vtkPoints *points = vtkPoints::New();

    int ncoord = cpoly->getNumPoints();
    float *x, *y, *z;
    int *cornerlist, *polylist;
    cpoly->getAddresses(&x, &y, &z, &cornerlist, &polylist);

    for (int i = 0; i < ncoord; ++i)
    {
        float p[3] = { x[i], y[i], z[i] };
        points->InsertPoint(i, p);
    }
    vpoly->SetPoints(points);

    vtkCellArray *polys = vtkCellArray::New();
    int npoly = cpoly->getNumPolygons();
    for (int i = 0; i < npoly; ++i)
    {
        int polystart = polylist[i];
        vtkIdType polylen = (i + 1 == npoly ? cpoly->getNumVertices() : polylist[i + 1]) - polystart;
        std::vector<vtkIdType> ids(polylen);
        for (int i = 0; i < polylen; ++i)
            ids[i] = cornerlist[polystart + i];
        polys->InsertNextCell(polylen, &ids[0]);
    }
    vpoly->SetPolys(polys);

    return vpoly;
}

static vtkPolyData *coviseLines2Vtk(const coDoLines *clines)
{
    vtkPolyData *vpoly = vtkPolyData::New();
    vtkPoints *points = vtkPoints::New();

    int ncoord = clines->getNumPoints();
    float *x, *y, *z;
    int *cornerlist, *linelist;
    clines->getAddresses(&x, &y, &z, &cornerlist, &linelist);

    for (int i = 0; i < ncoord; ++i)
    {
        float p[3] = { x[i], y[i], z[i] };
        points->InsertPoint(i, p);
    }
    vpoly->SetPoints(points);

    vtkCellArray *lines = vtkCellArray::New();
    int nlines = clines->getNumLines();
    for (int i = 0; i < nlines; ++i)
    {
        int linestart = linelist[i];
        vtkIdType linelen = (i + 1 == nlines ? clines->getNumVertices() : linelist[i + 1]) - linestart;
        std::vector<vtkIdType> ids(linelen);
        for (int i = 0; i < linelen; ++i)
            ids[i] = cornerlist[linestart + i];
        lines->InsertNextCell(linelen, &ids[0]);
    }
    vpoly->SetLines(lines);

    return vpoly;
}

static vtkPolyData *covisePoints2Vtk(const coDoPoints *cpoints)
{
    vtkPolyData *vpoly = vtkPolyData::New();
    vtkPoints *points = vtkPoints::New();

    int ncoord = cpoints->getNumPoints();
    float *x, *y, *z;
    cpoints->getAddresses(&x, &y, &z);

    for (int i = 0; i < ncoord; ++i)
    {
        float p[3] = { x[i], y[i], z[i] };
        points->InsertPoint(i, p);
    }
    vpoly->SetPoints(points);

    vtkCellArray *verts = vtkCellArray::New();
    for (vtkIdType i = 0; i < ncoord; ++i)
    {
        verts->InsertNextCell(1, &i);
    }
    vpoly->SetVerts(verts);

    return vpoly;
}
#endif

coDoGrid *coVtk::vtkGrid2Covise(const coObjInfo &info, vtkDataSet *vtk)
{
#ifdef HAVE_VTK
    if (vtkUnstructuredGrid *vugrid = dynamic_cast<vtkUnstructuredGrid *>(vtk))
        return ::vtkUGrid2Covise(info, vugrid);

    if (vtkStructuredGrid *vsgrid = dynamic_cast<vtkStructuredGrid *>(vtk))
        return ::vtkSGrid2Covise(info, vsgrid);

    if (vtkPolyData *vpolydata = dynamic_cast<vtkPolyData *>(vtk))
        return ::vtkPoly2Covise(info, vpolydata);

    if (vtkUniformGrid *vugrid = dynamic_cast<vtkUniformGrid *>(vtk))
        return ::vtkUniGrid2Covise(info, vugrid);

    if (vtkImageData *vimage = dynamic_cast<vtkImageData *>(vtk))
        return ::vtkImage2Covise(info, vimage);

    if (vtkRectilinearGrid *vrgrid = dynamic_cast<vtkRectilinearGrid *>(vtk))
        return ::vtkRGrid2Covise(info, vrgrid);
#else
    (void)info;
    (void)vtk;
#endif

    return NULL;
}

vtkDataSet *coVtk::coviseGrid2Vtk(const coDoGrid *grid)
{
#ifdef HAVE_VTK
    if (const coDoUniformGrid *ugrid = dynamic_cast<const coDoUniformGrid *>(grid))
        return ::coviseUniGrid2Vtk(ugrid);

    if (const coDoRectilinearGrid *rgrid = dynamic_cast<const coDoRectilinearGrid *>(grid))
        return ::coviseRGrid2Vtk(rgrid);

    if (const coDoStructuredGrid *sgrid = dynamic_cast<const coDoStructuredGrid *>(grid))
        return ::coviseSGrid2Vtk(sgrid);

    if (const coDoUnstructuredGrid *ugrid = dynamic_cast<const coDoUnstructuredGrid *>(grid))
        return ::coviseUGrid2Vtk(ugrid);

    if (const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(grid))
        return ::covisePoly2Vtk(poly);

    if (const coDoTriangleStrips *tris = dynamic_cast<const coDoTriangleStrips *>(grid))
        return ::coviseTris2Vtk(tris);

    if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(grid))
        return ::covisePoints2Vtk(points);

    if (const coDoLines *lines = dynamic_cast<const coDoLines *>(grid))
        return ::coviseLines2Vtk(lines);
#else
    (void)grid;
#endif

    return NULL;
}

vtkDataArray *coVtk::coviseData2Vtk(const coDoGrid *grid, const coDoAbstractData *data, Flags flags)
{
#ifdef HAVE_VTK
    const int n = data->getNumPoints();
    if (n != grid->getNumPoints())
    {
        fprintf(stderr, "coVtk::coviseData2Vtk: grid (%d) and data size (%d) do not match\n",
                grid->getNumPoints(), n);
        if (n < grid->getNumPoints())
            return NULL;
    }

    vtkDataArray *vd = NULL;
    if (flags & RequireDouble)
        vd = vtkDoubleArray::New();
    int dim[3] = { data->getNumPoints(), 1, 1 };
    if (const coDoAbstractStructuredGrid *sgrid = dynamic_cast<const coDoAbstractStructuredGrid *>(grid))
        sgrid->getGridSize(&dim[0], &dim[1], &dim[2]);

    if (const coDoFloat *fdata = dynamic_cast<const coDoFloat *>(data))
    {
        float *d = fdata->getAddress();
        if (!vd)
            vd = vtkFloatArray::New();
        vd->SetNumberOfComponents(1);
        vd->SetNumberOfTuples(n);
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    vd->SetTuple1(l, d[coIndex(i, j, k, dim)]);
                    ++l;
                }
    }
    else if (const coDoVec2 *vdata = dynamic_cast<const coDoVec2 *>(data))
    {
        if (!vd)
            vd = vtkFloatArray::New();
        vd->SetNumberOfComponents(2);
        vd->SetNumberOfTuples(n);
        float *x, *y;
        vdata->getAddresses(&x, &y);
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    const int idx = coIndex(i, j, k, dim);
                    vd->SetTuple2(l, x[idx], y[idx]);
                    ++l;
                }
    }
    else if (const coDoVec3 *vdata = dynamic_cast<const coDoVec3 *>(data))
    {
        if (!vd)
            vd = vtkFloatArray::New();
        vd->SetNumberOfComponents(3);
        vd->SetNumberOfTuples(n);

        float *x, *y, *z;
        vdata->getAddresses(&x, &y, &z);
        if (flags & Normalize)
        {
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        const float len = sqrtf(x[idx] * x[idx] + y[idx] * y[idx] + z[idx] * z[idx]);
                        if (len > 0.f)
                            vd->SetTuple3(l, x[idx] / len, y[idx] / len, z[idx] / len);
                        else
                            vd->SetTuple3(l, x[idx], y[idx], z[idx]);
                        ++l;
                    }
        }
        else
        {
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        vd->SetTuple3(l, x[idx], y[idx], z[idx]);
                        ++l;
                    }
        }
    }
    else if (const coDoInt *idata = dynamic_cast<const coDoInt *>(data))
    {
        if (!vd)
            vd = vtkIntArray::New();
        vd->SetNumberOfComponents(1);
        vd->SetNumberOfTuples(n);

        const int *d = idata->getAddress();
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    const int idx = coIndex(i, j, k, dim);
                    vd->SetTuple1(l, d[idx]);
                    ++l;
                }
    }
    else if (const coDoRGBA *rdata = dynamic_cast<const coDoRGBA *>(data))
    {
        if (!vd)
            vd = vtkUnsignedCharArray::New();
        vd->SetNumberOfComponents(4);
        vd->SetNumberOfTuples(n);
        const int *d = rdata->getAddress();

        if (flags & RequireDouble)
        {
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        char c[4];
                        memcpy(c, &d[idx], sizeof(c));
                        vd->SetTuple4(l, c[0] / 255., c[1] / 255., c[2] / 255., c[3] / 255.);
                        ++l;
                    }
        }
        else
        {
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        char c[4];
                        memcpy(c, &d[idx], sizeof(c));
                        vd->SetTuple4(l, c[0], c[1], c[2], c[3]);
                        ++l;
                    }
        }
    }
    else
    {
        vd->Delete();
        return NULL;
    }

    return vd;
#else
    (void)grid;
    (void)data;
    (void)flags;
#endif

    return NULL;
}

template <typename ValueType, class vtkType>
static coDoAbstractData *vtkArray2Covise(const coObjInfo &info, vtkType *vd, const coDoAbstractStructuredGrid *sgrid = NULL)
{
    const int n = vd->GetNumberOfTuples();
    int dim[3] = { n, 1, 1 };
    if (sgrid)
        sgrid->getGridSize(&dim[0], &dim[1], &dim[2]);
    if (n > 0 && n == (dim[0]-1)*(dim[1]-1)*(dim[2]-1))
    {
        // cell-mapped data
        for (int i=0; i<3; ++i)
            --dim[i];
    }

    switch (vd->GetNumberOfComponents())
    {
    case 1:
    {
        coDoFloat *cf = new coDoFloat(info, n);
        CHECK_AND_RETURN(cf);
        float *x = cf->getAddress();
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    x[coIndex(i, j, k, dim)] = vd->GetValue(l);
                    ++l;
                }
        return cf;
    }
    break;
    case 2:
    {
        coDoVec2 *cv = new coDoVec2(info, n);
        CHECK_AND_RETURN(cv);
        float *x, *y;
        cv->getAddresses(&x, &y);
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    ValueType v[2];
                    vd->GetTupleValue(l, v);
                    const int idx = coIndex(i, j, k, dim);
                    x[idx] = v[0];
                    y[idx] = v[1];
                    ++l;
                }
        return cv;
    }
    break;
    case 3:
    {
        coDoVec3 *cv = new coDoVec3(info, n);
        CHECK_AND_RETURN(cv);
        float *x, *y, *z;
        cv->getAddresses(&x, &y, &z);
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    ValueType v[3];
                    vd->GetTupleValue(l, v);
                    const int idx = coIndex(i, j, k, dim);
                    x[idx] = v[0];
                    y[idx] = v[1];
                    z[idx] = v[2];
                    ++l;
                }
        return cv;
    }
    break;
    case 4:
    {
        coDoRGBA *cv = new coDoRGBA(info, n);
        CHECK_AND_RETURN(cv);
        int *d = cv->getAddress();
        int l = 0;
        for (int k = 0; k < dim[2]; ++k)
            for (int j = 0; j < dim[1]; ++j)
                for (int i = 0; i < dim[0]; ++i)
                {
                    ValueType v[4];
                    vd->GetTupleValue(l, v);
                    const int idx = coIndex(i, j, k, dim);
                    char c[4];
                    for (int j = 0; j < 4; ++j)
                        c[j] = (char)(v[j] * 255.99f);
                    memcpy(&d[idx], c, sizeof(c));
                    ++l;
                }
        return cv;
    }
    break;
    }

    return NULL;
}

coDoAbstractData *coVtk::vtkData2Covise(const coObjInfo &info, vtkDataArray *varr, const coDoAbstractStructuredGrid *sgrid)
{
#ifdef HAVE_VTK
    const int n = varr->GetNumberOfTuples();
    int dim[3] = { n, 1, 1 };
    if (sgrid)
        sgrid->getGridSize(&dim[0], &dim[1], &dim[2]);
    if (n > 0 && n == (dim[0]-1)*(dim[1]-1)*(dim[2]-1))
    {
        // cell-mapped data
        for (int i=0; i<3; ++i)
            --dim[i];
    }
    if (dim[0] * dim[1] * dim[2] != n)
    {
        std::cerr << "coVtk::vtkData2Covise: non-matching grid size: [" << dim[0] << "*" << dim[1] << "*" << dim[2] << "] != " << n << std::endl;
        return NULL;
    }

    if (vtkFloatArray *vd = dynamic_cast<vtkFloatArray *>(varr))
    {
        return vtkArray2Covise<float, vtkFloatArray>(info, vd, sgrid);
    }
    else if (vtkDoubleArray *vd = dynamic_cast<vtkDoubleArray *>(varr))
    {
        return vtkArray2Covise<double, vtkDoubleArray>(info, vd, sgrid);
    }
    else if (vtkShortArray *vd = dynamic_cast<vtkShortArray *>(varr))
    {
        return vtkArray2Covise<short, vtkShortArray>(info, vd, sgrid);
    }
    else if (vtkUnsignedShortArray *vd = dynamic_cast<vtkUnsignedShortArray *>(varr))
    {
        return vtkArray2Covise<unsigned short, vtkUnsignedShortArray>(info, vd, sgrid);
    }
    else if (vtkUnsignedIntArray *vd = dynamic_cast<vtkUnsignedIntArray *>(varr))
    {
        return vtkArray2Covise<unsigned int, vtkUnsignedIntArray>(info, vd, sgrid);
    }
    else if (vtkIdTypeArray *vd = dynamic_cast<vtkIdTypeArray *>(varr))
    {
        switch (vd->GetNumberOfComponents())
        {
        case 1:
        {
            coDoInt *cf = new coDoInt(info, n);
            CHECK_AND_RETURN(cf);
            int *x = cf->getAddress();
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        x[idx] = vd->GetValue(l);
                        ++l;
                    }
            return cf;
        }
        break;
        default:
            return NULL;
            break;
        }
    }
    else if (vtkIntArray *vd = dynamic_cast<vtkIntArray *>(varr))
    {
        switch (vd->GetNumberOfComponents())
        {
        case 1:
        {
            coDoInt *cf = new coDoInt(info, n);
            CHECK_AND_RETURN(cf);
            int *x = cf->getAddress();
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        x[idx] = vd->GetValue(l);
                        ++l;
                    }
            return cf;
        }
        break;
        default:
            return NULL;
            break;
        }
    }
    else if (vtkUnsignedCharArray *vd = dynamic_cast<vtkUnsignedCharArray *>(varr))
    {
        switch (vd->GetNumberOfComponents())
        {
        case 1:
        {
            coDoByte *cb = new coDoByte(info, n);
            CHECK_AND_RETURN(cb);
            unsigned char *x = cb->getAddress();
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        const int idx = coIndex(i, j, k, dim);
                        x[idx] = vd->GetValue(l);
                        ++l;
                    }
            return cb;
        }
        break;
        case 4:
        {
            coDoRGBA *cv = new coDoRGBA(info, n);
            CHECK_AND_RETURN(cv);
            int *d = cv->getAddress();
            int l = 0;
            for (int k = 0; k < dim[2]; ++k)
                for (int j = 0; j < dim[1]; ++j)
                    for (int i = 0; i < dim[0]; ++i)
                    {
                        unsigned char c[4];
                        const int idx = coIndex(i, j, k, dim);
                        vd->GetTupleValue(l, c);
                        memcpy(&d[idx], c, sizeof(c));
                        ++l;
                    }
            return cv;
        }
        break;
        default:
            return NULL;
            break;
        }
    }
#else
    (void)info;
    (void)varr;
    (void)sgrid;
#endif

    return NULL;
}

coDoAbstractData *coVtk::vtkData2Covise(const coObjInfo &info, vtkDataSetAttributes *vattr, int attribute, const char *name, const coDoAbstractStructuredGrid *sgrid)
{
#ifdef HAVE_VTK
    vtkDataArray *varr = NULL;
    switch (attribute)
    {
    case Scalars:
        varr = name ? vattr->GetScalars(name) : vattr->GetScalars();
        break;
    case Vectors:
        varr = name ? vattr->GetVectors(name) : vattr->GetVectors();
        break;
    case Normals:
        varr = name ? vattr->GetNormals(name) : vattr->GetNormals();
        break;
    case TexCoords:
        varr = name ? vattr->GetTCoords(name) : vattr->GetTCoords();
        break;
    case Tensors:
        varr = name ? vattr->GetTensors(name) : vattr->GetTensors();
        break;
    case Any:
        if (name)
        {
            varr = vattr->GetScalars(name);
            if (!varr)
                varr = vattr->GetVectors(name);
            if (!varr)
                varr = vattr->GetNormals(name);
            if (!varr)
                varr = vattr->GetTCoords(name);
            if (!varr)
                varr = vattr->GetTensors(name);
        }
        break;
    default:
        break;
    }

    if (!varr)
    {
        if (name)
        {
            std::cerr << "coVtk::vtkData2Covise: did not find vtkDataArray " << name << std::endl;
        }
        return NULL;
    }

    return vtkData2Covise(info, varr, sgrid);
#else
    (void)info;
    (void)vattr;
    (void)attribute;
    (void)name;
    (void)sgrid;
    return NULL;
#endif
}

coDoAbstractData *coVtk::vtkData2Covise(const coObjInfo &info, vtkDataSet *vtk, int attribute, const char *name, const coDoAbstractStructuredGrid *sgrid)
{
#ifdef HAVE_VTK
    vtkDataSetAttributes *vattr = vtk->GetPointData();
    coDoAbstractData *data = vtkData2Covise(info, vattr, attribute, name, sgrid);
    if (data)
        return data;
    vattr = vtk->GetCellData();
    return vtkData2Covise(info, vattr, attribute, name, sgrid);
#else
    (void)info;
    (void)vtk;
    (void)attribute;
    (void)name;
    (void)sgrid;
    return NULL;
#endif
}

coDoGeometry *coVtk::vtk2Covise(const coObjInfo &info, vtkDataSet *vtk)
{
#ifdef HAVE_VTK
    char name[10000];
    strcpy(name, info.getName());
    strcat(name, "_grid");
    coDistributedObject *obj = vtkGrid2Covise(coObjInfo(name), vtk);
    if (!obj)
    {
        fprintf(stderr, "coVtk::vtk2Covise: failed to convert vtk grid of type %s\n", vtk->GetClassName());
        return NULL;
    }

    coDoGeometry *geo = new coDoGeometry(info, obj);
    strcpy(name, info.getName());
    strcat(name, "_data");
    if (coDistributedObject *data = vtkData2Covise(coObjInfo(name), vtk, Vectors, NULL, dynamic_cast<coDoAbstractStructuredGrid *>(obj)))
        geo->setColors(NONE, data);
    else if (coDistributedObject *data = vtkData2Covise(coObjInfo(name), vtk, Scalars, NULL, dynamic_cast<coDoAbstractStructuredGrid *>(obj)))
        geo->setColors(NONE, data);

    strcpy(name, info.getName());
    strcat(name, "_normals");
    if (coDistributedObject *norm = vtkData2Covise(coObjInfo(name), vtk, Normals, NULL, dynamic_cast<coDoAbstractStructuredGrid *>(obj)))
        geo->setNormals(NONE, norm);

    return geo;
#else
    (void)info;
    (void)vtk;
    return NULL;
#endif
}

vtkDataSet *coVtk::coviseGeometry2Vtk(const coDoGeometry *cgeo)
{
#ifdef HAVE_VTK
    coDoGeometry *geo = const_cast<coDoGeometry *>(cgeo);
    const coDoGrid *grid = dynamic_cast<const coDoGrid *>(geo->getGeometry());
    if (!grid)
        return NULL;
    vtkDataSet *vgrid = coviseGrid2Vtk(grid);
    vtkDataSetAttributes *vattr = vgrid->GetPointData();
    if (const coDoFloat *sdata = dynamic_cast<const coDoFloat *>(geo->getColors()))
    {
        vattr->SetScalars(coviseData2Vtk(grid, sdata));
    }
    else if (const coDoInt *idata = dynamic_cast<const coDoInt *>(geo->getColors()))
    {
        vattr->SetScalars(coviseData2Vtk(grid, idata));
    }
    else if (const coDoVec2 *vdata = dynamic_cast<const coDoVec2 *>(geo->getColors()))
    {
        vattr->SetVectors(coviseData2Vtk(grid, vdata));
    }
    else if (const coDoVec3 *vdata = dynamic_cast<const coDoVec3 *>(geo->getColors()))
    {
        vattr->SetVectors(coviseData2Vtk(grid, vdata));
    }
    else if (const coDoRGBA *vdata = dynamic_cast<const coDoRGBA *>(geo->getColors()))
    {
        vattr->SetVectors(coviseData2Vtk(grid, vdata));
    }

    if (const coDoVec3 *normals = dynamic_cast<const coDoVec3 *>(geo->getNormals()))
    {
        vattr->SetNormals(coviseData2Vtk(grid, normals, Normalize));
    }

    return vgrid;
#else
    (void)cgeo;
    return NULL;
#endif
}

vtkDataSet *coVtk::covise2Vtk(const coDistributedObject *obj)
{
#ifdef HAVE_VTK
    if (const coDoGeometry *geo = dynamic_cast<const coDoGeometry *>(obj))
        return coviseGeometry2Vtk(geo);
    else if (const coDoGrid *grid = dynamic_cast<const coDoGrid *>(obj))
        return coviseGrid2Vtk(grid);
    else if (const coDoPixelImage *img = dynamic_cast<const coDoPixelImage *>(obj))
        return ::coviseImage2Vtk(img);
    else if (const coDoTexture *tex = dynamic_cast<const coDoTexture *>(obj))
        return ::coviseTexture2Vtk(tex);
    else
        return NULL;
#else
    (void)obj;
    return NULL;
#endif
}

vtkDataObject *coVtk::covise2Vtk(const coDistributedObject *geo,
                                 const std::vector<const coDistributedObject *> &fields,
                                 const coDistributedObject *normals)
{
    if (const coDoGeometry *geom = dynamic_cast<const coDoGeometry *>(geo))
    {
        // reject if other data is present
        if (normals != NULL || !fields.empty())
        {
            std::cerr << "coVtk::covise2Vtk: converting geometry, but also normals or other field data present" << std::endl;
            return NULL;
        }

        std::vector<const coDistributedObject *> f;
        if (geom->getColors())
            f.push_back(geom->getColors());
        return covise2Vtk(geom->getGeometry(), f, geom->getNormals());
    }
    else if (const coDoSet *set = dynamic_cast<const coDoSet *>(geo))
    {
        // traverse set hierarchy
        bool timestep = set->getAttribute("TIMESTEP") != NULL;
        const int nelem = set->getNumElements();
        const coDoSet *nset = dynamic_cast<const coDoSet *>(normals);
        if (normals)
        {
            if (!nset)
            {
                std::cerr << "coVtk::covise2Vtk: converting set of grids, but no normal set" << std::endl;
                return NULL;
            }
            if (nset->getNumElements() != nelem)
            {
                std::cerr << "coVtk::covise2Vtk: converting set of grids, but normal set of differing size" << std::endl;
                return NULL;
            }
            timestep |= nset->getAttribute("TIMESTEP") != NULL;
        }

        std::vector<const coDoSet *> fsets;
        for (std::vector<const coDistributedObject *>::const_iterator it = fields.begin();
             it != fields.end();
             ++it)
        {
            fsets.push_back(dynamic_cast<const coDoSet *>(*it));
            if (!fsets.back() && *it)
            {
                std::cerr << "coVtk::covise2Vtk: converting set of grids, but no field set" << std::endl;
                return NULL;
            }
            if (fsets.back()->getNumElements() != nelem)
            {
                std::cerr << "coVtk::covise2Vtk: converting set of grids, but field set of differing size" << std::endl;
                return NULL;
            }
            timestep |= fsets.back()->getAttribute("TIMESTEP") != NULL;
        }

        // convert time step do vtkTemporalDataSet, other sets to vtkMultiPieceDataSet
        vtkCompositeDataSet *comp = NULL;
        vtkMultiPieceDataSet *mp = NULL;
#ifdef HAVE_VTK_TEMP
        vtkTemporalDataSet *temp = NULL;
        if (timestep)
        {
            temp = vtkTemporalDataSet::New();
            temp->SetNumberOfTimeSteps(nelem);
            comp = temp;
        }
        else
#endif
        {
            mp = vtkMultiPieceDataSet::New();
            mp->SetNumberOfPieces(nelem);
            comp = mp;
        }

        for (int i = 0; i < nelem; ++i)
        {
            std::vector<const coDistributedObject *> f;
            for (std::vector<const coDoSet *>::const_iterator it = fsets.begin();
                 it != fsets.end();
                 ++it)
            {
                f.push_back((*it)->getElement(i));
            }
            vtkDataObject *vtk = covise2Vtk(set->getElement(i), f, nset ? nset->getElement(i) : NULL);
            if (!vtk)
            {
                comp->Delete();
                std::cerr << "coVtk::covise2Vtk: conversion of set hierarchy failed" << std::endl;
                return NULL;
            }
#ifdef HAVE_VTK_TEMP
            if (timestep)
                temp->SetTimeStep(i, vtk);
            else
#endif
                mp->SetPiece(i, (vtkDataSet *)vtk);
        }

        return comp;
    }
    else
    {
        const coDoGrid *grid = dynamic_cast<const coDoGrid *>(geo);
        if (!grid)
        {
            std::cerr << "coVtk::covise2Vtk: grid no grid data type" << std::endl;
            return NULL;
        }
        vtkDataSet *vtk = covise2Vtk(grid);
        if (!vtk)
        {
            std::cerr << "coVtk::covise2Vtk: conversion of grid failed" << std::endl;
            return NULL;
        }
        vtkDataSetAttributes *vattr = vtk->GetPointData();
        coVtk::Flags flags = coVtk::None;
        if (normals)
        {
            const coDoAbstractData *norm = dynamic_cast<const coDoVec3 *>(normals);
            if (!norm)
            {
                vtk->Delete();
                std::cerr << "coVtk::covise2Vtk: conversion of normals failed" << std::endl;
                return NULL;
            }
            vtkDataArray *vnorm = coviseData2Vtk(grid, norm, coVtk::Flags(flags | coVtk::Normalize));
            vattr->SetNormals(vnorm);
        }
        for (int i = 0; i < fields.size(); ++i)
        {
            if (const coDoAbstractData *data = dynamic_cast<const coDoAbstractData *>(fields[i]))
            {
                vtkDataArray *vdata = coVtk::coviseData2Vtk(grid, data, flags);
                if (!vdata)
                {
                    vtk->Delete();
                    std::cerr << "coVtk::covise2Vtk: conversion of field data failed" << std::endl;
                    return NULL;
                }
                std::stringstream s;
                s << "field" << i;
                vdata->SetName(s.str().c_str());
                vattr->AddArray(vdata);
                vattr->SetActiveScalars(s.str().c_str());
            }
        }
        return vtk;
    }

    return NULL;
}

bool coVtk::isPortRequired(vtkInformation *info)
{
#ifdef HAVE_VTK
    return !info->Get(vtkAlgorithm::INPUT_IS_OPTIONAL());
#else
    (void)info;
    return true;
#endif
}

std::string coVtk::inPortTypeList(vtkInformation *info)
{
    static std::map<std::string, std::string> typeMap;
    if (typeMap.empty())
    {
        typeMap["vtkImageData"] = "UniformGrid|PixelImage|Texture";
        typeMap["vtkStructuredPoints"] = "UniformGrid";
        typeMap["vtkUniformGrid"] = "UniformGrid";
        typeMap["vtkRectilinearGrid"] = "RectilinearGrid";
        typeMap["vtkUnstructuredGrid"] = "UnstructuredGrid";
        typeMap["vtkStructuredGrid"] = "StructuredGrid";
        typeMap["vtkPolyData"] = "Points|Lines|Polygons|TriangleStrips";
        typeMap["vtkPointSet"] = typeMap["vtkPolyData"] + "|StructuredGrid|UnstructuredGrid";
        typeMap["vtkDataSet"] = typeMap["vtkPointSet"] + "|UniformGrid|RectilinearGrid";
    }
#ifdef HAVE_VTK
    const char *type = info->Get(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE());
    if (type && typeMap.find(type) != typeMap.end())
        return typeMap[type] + "|Geometry";
    else
        return "coDistributedObject";
#else
    (void)info;
    return "coDistributedObject";
#endif
}

std::string coVtk::outPortTypeList(vtkInformation *info)
{
    static std::map<std::string, std::string> typeMap;
    if (typeMap.empty())
    {
        typeMap["vtkImageData"] = "UniformGrid|PixelImage|Texture";
        typeMap["vtkStructuredPoints"] = "UniformGrid";
        typeMap["vtkUniformGrid"] = "UniformGrid";
        typeMap["vtkRectilinearGrid"] = "RectilinearGrid";
        typeMap["vtkUnstructuredGrid"] = "UnstructuredGrid";
        typeMap["vtkStructuredGrid"] = "StructuredGrid";
        typeMap["vtkPolyData"] = "Points|Lines|Polygons|TriangleStrips";
        typeMap["vtkPointSet"] = typeMap["vtkPolyData"] + "|StructuredGrid|UnstructuredGrid";
        typeMap["vtkDataSet"] = typeMap["vtkPointSet"] + "|UniformGrid|RectilinearGrid";
    }
#ifdef HAVE_VTK
    const char *type = info->Get(vtkDataObject::DATA_TYPE_NAME());
    if (type && typeMap.find(type) != typeMap.end())
        return typeMap[type];
    else
        return "coDistributedObject";
#else
    (void)info;
    return "coDistributedObject";
#endif
}
