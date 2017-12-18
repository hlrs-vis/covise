/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoTriangleStrips.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include "Transform.h"
#include <vector>

#ifdef __hpux
double drem(double x, double y)
{
    return remainder(x, y);
}

#elif defined(__APPLE__)
double remainder(double x, double y)
{
    double tmp = x / y;
    return tmp - ((int)tmp);
}
#endif

bool
Transform::isUnstructured(const coDistributedObject *obj)
{
    if (obj->isType("POLYGN")
        || obj->isType("LINES")
        || obj->isType("POINTS")
        || obj->isType("TRIANG")
        || obj->isType("UNSGRD")
        || obj->isType("USTSDT")
        || obj->isType("USTVDT"))
    {
        return true;
    }
    return false;
}

coDoPolygons *
AssemblePolygons(const char *name, const coDistributedObject *const *list)
{
    std::vector<int> pl;
    std::vector<int> vl;
    std::vector<float> xc, yc, zc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("POLYGN")))
        {
            return NULL;
        }
        coDoPolygons *pol = (coDoPolygons *)(*list);
        int no_pl, no_vl, no_points;
        no_pl = pol->getNumPolygons();
        no_vl = pol->getNumVertices();
        no_points = pol->getNumPoints();
        int *plist, *vlist;
        float *xclist, *yclist, *zclist;
        pol->getAddresses(&xclist, &yclist, &zclist, &vlist, &plist);
        int basis_pl, basis_vl;
        basis_pl = (int)vl.size();
        basis_vl = (int)xc.size();
        int p, v, point;
        for (p = 0; p < no_pl; ++p)
        {
            pl.push_back(plist[p] + basis_pl);
        }
        for (v = 0; v < no_vl; ++v)
        {
            vl.push_back(vlist[v] + basis_vl);
        }
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
            yc.push_back(yclist[point]);
            zc.push_back(zclist[point]);
        }
    }
    return new coDoPolygons(name, (int)xc.size(), &xc[0], &yc[0], &zc[0],
                            (int)vl.size(), &vl[0], (int)pl.size(), &pl[0]);
}

coDoLines *
AssembleLines(const char *name, const coDistributedObject *const *list)
{
    std::vector<int> pl;
    std::vector<int> vl;
    std::vector<float> xc, yc, zc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("LINES")))
        {
            return NULL;
        }
        coDoLines *pol = (coDoLines *)(*list);
        int no_pl, no_vl, no_points;
        no_pl = pol->getNumLines();
        no_vl = pol->getNumVertices();
        no_points = pol->getNumPoints();
        int *plist, *vlist;
        float *xclist, *yclist, *zclist;
        pol->getAddresses(&xclist, &yclist, &zclist, &vlist, &plist);
        int basis_pl, basis_vl;
        basis_pl = (int)vl.size();
        basis_vl = (int)xc.size();
        int p, v, point;
        for (p = 0; p < no_pl; ++p)
        {
            pl.push_back(plist[p] + basis_pl);
        }
        for (v = 0; v < no_vl; ++v)
        {
            vl.push_back(vlist[v] + basis_vl);
        }
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
            yc.push_back(yclist[point]);
            zc.push_back(zclist[point]);
        }
    }
    return new coDoLines(name, (int)xc.size(), &xc[0], &yc[0], &zc[0],
		(int)vl.size(), &vl[0], (int)pl.size(), &pl[0]);
}

coDoPoints *
AssemblePoints(const char *name, const coDistributedObject *const *list)
{
    std::vector<float> xc, yc, zc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("POINTS")))
        {
            return NULL;
        }
        coDoPoints *pol = (coDoPoints *)(*list);
        int no_points;
        no_points = pol->getNumPoints();
        float *xclist, *yclist, *zclist;
        pol->getAddresses(&xclist, &yclist, &zclist);
        int point;
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
            yc.push_back(yclist[point]);
            zc.push_back(zclist[point]);
        }
    }
    return new coDoPoints(name, (int)xc.size(), &xc[0], &yc[0], &zc[0]);
}

coDoUnstructuredGrid *
AssembleUnsGrd(const char *name, const coDistributedObject *const *list)
{
    std::vector<int> pl;
    std::vector<int> vl;
    std::vector<int> tl;
    std::vector<float> xc, yc, zc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("UNSGRD")))
        {
            return NULL;
        }
        coDoUnstructuredGrid *pol = (coDoUnstructuredGrid *)(*list);
        int no_pl, no_vl, no_points;
        pol->getGridSize(&no_pl, &no_vl, &no_points);
        int *plist, *vlist, *tlist;
        float *xclist, *yclist, *zclist;
        pol->getAddresses(&plist, &vlist, &xclist, &yclist, &zclist);
        pol->getTypeList(&tlist);
        int basis_pl, basis_vl;
        basis_pl = (int)vl.size();
        basis_vl = (int)xc.size();
        int p, v, point;
        for (p = 0; p < no_pl; ++p)
        {
            pl.push_back(plist[p] + basis_pl);
            tl.push_back(tlist[p]);
        }
        for (v = 0; v < no_vl; ++v)
        {
            vl.push_back(vlist[v] + basis_vl);
        }
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
            yc.push_back(yclist[point]);
            zc.push_back(zclist[point]);
        }
    }
    return new coDoUnstructuredGrid(name, (int)pl.size(), (int)vl.size(), (int)xc.size(),
                                    &pl[0], &vl[0],
                                    &xc[0], &yc[0], &zc[0],
                                    &tl[0]);
}

coDoTriangleStrips *
AssembleTriang(const char *name, const coDistributedObject *const *list)
{
    std::vector<int> pl;
    std::vector<int> vl;
    std::vector<float> xc, yc, zc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("TRIANG")))
        {
            return NULL;
        }
        coDoTriangleStrips *pol = (coDoTriangleStrips *)(*list);
        int no_pl, no_vl, no_points;
        no_pl = pol->getNumStrips();
        no_vl = pol->getNumVertices();
        no_points = pol->getNumPoints();
        int *plist, *vlist;
        float *xclist, *yclist, *zclist;
        pol->getAddresses(&xclist, &yclist, &zclist, &vlist, &plist);
        int basis_pl, basis_vl;
        basis_pl = (int)vl.size();
        basis_vl = (int)xc.size();
        int p, v, point;
        for (p = 0; p < no_pl; ++p)
        {
            pl.push_back(plist[p] + basis_pl);
        }
        for (v = 0; v < no_vl; ++v)
        {
            vl.push_back(vlist[v] + basis_vl);
        }
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
            yc.push_back(yclist[point]);
            zc.push_back(zclist[point]);
        }
    }
    return new coDoTriangleStrips(name, (int)xc.size(), &xc[0], &yc[0], &zc[0],
                                  (int)vl.size(), &vl[0], (int)pl.size(), &pl[0]);
}

coDoFloat *
AssembleUnsSdt(const char *name, const coDistributedObject *const *list)
{
    std::vector<float> xc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("USTSDT")))
        {
            return NULL;
        }
        coDoFloat *pol = (coDoFloat *)(*list);
        int no_points;
        no_points = pol->getNumPoints();
        float *xclist;
        pol->getAddress(&xclist);
        int point;
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
        }
    }
    return new coDoFloat(name, (int)xc.size(), &xc[0]);
}

coDoVec3 *
AssembleUnsVdt(const char *name, const coDistributedObject *const *list)
{
    std::vector<float> xc, yc, zc;
    for (; *list; ++list)
    {
        if (!((*list)->isType("USTVDT")))
        {
            return NULL;
        }
        coDoVec3 *pol = (coDoVec3 *)(*list);
        int no_points;
        no_points = pol->getNumPoints();
        float *xclist, *yclist, *zclist;
        pol->getAddresses(&xclist, &yclist, &zclist);
        int point;
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(xclist[point]);
            yc.push_back(yclist[point]);
            zc.push_back(zclist[point]);
        }
    }
    return new coDoVec3(name, (int)xc.size(), &xc[0], &yc[0], &zc[0]);
}

coDistributedObject *
Transform::AssembleObjects(const coDistributedObject *in, const char *name, const coDistributedObject *const *list)
{
    coDistributedObject *out;
    if (!list[0] || !isUnstructured(list[0]))
    {
        out = new coDoSet(name, list);
    }
    if (list[0]->isType("POLYGN"))
    {
        out = AssemblePolygons(name, list);
    }
    else if (list[0]->isType("LINES"))
    {
        out = AssembleLines(name, list);
    }
    else if (list[0]->isType("POINTS"))
    {
        out = AssemblePoints(name, list);
    }
    else if (list[0]->isType("TRIANG"))
    {
        out = AssembleTriang(name, list);
    }
    else if (list[0]->isType("UNSGRD"))
    {
        out = AssembleUnsGrd(name, list);
    }
    else if (list[0]->isType("USTSDT"))
    {
        out = AssembleUnsSdt(name, list);
    }
    else if (list[0]->isType("USTVDT"))
    {
        out = AssembleUnsVdt(name, list);
    }
    else
    {
        return new coDoSet(name, list);
    }
    Matrix matrix; // identity transformation, what else to do!?
    copyRotateAttributes(out, in, matrix);
    return out;
}

// reusing object
const coDistributedObject *
Transform::retObject(const coDistributedObject *obj)
{
    // remove Attribute TRANSFORM from output polygon
    if (obj->getAttribute("TRANSFORM") != NULL)
    {
        if (obj->isType("POLYGN")) // copy
        {
            int *pl, *vl;
            float *x, *y, *z;

            const coDoPolygons *in = (const coDoPolygons *)obj;
            in->getAddresses(&x, &y, &z, &vl, &pl);

            char *new_name = new char[strlen(obj->getName()) + 3];
            sprintf(new_name, "%s_1", obj->getName());
            coDoPolygons *out = new coDoPolygons(new_name, in->getNumPoints(), x, y, z,
                                                 in->getNumVertices(), vl, in->getNumPolygons(), pl);
            Matrix dummy;
            copyRotateAttributes(out, in, dummy);

            delete[] new_name;
            return out;
        }
    }
    obj->incRefCount();
    return obj;
}

// outputs displacements when tiling
// the algorithm relies on the fact that the output displacements
// may be obtained as the difference between the tiled coordinates
// of the 'displaced' (input) configuration and the tiled coordinates
// of the 'lagrangian' configuration
static coDistributedObject *
TilingFormula(const char *name, // output data object name
              const Geometry &geometry, // input and output geometry coordinates
              float *u[3], // displacements
              const Matrix *lagrangeTrans) // tiling transformation for 'largangian configuration
{
    int no_points = geometry.getSize();
    std::vector<float> xc(no_points);
    std::vector<float> yc(no_points);
    std::vector<float> zc(no_points);
    // dump eulerian input geometry
    geometry.dumpGeometry(&xc[0], &yc[0], &zc[0], false);

    int point;
    for (point = 0; point < no_points; ++point)
    {
        xc[point] -= u[0][point];
        yc[point] -= u[1][point];
        zc[point] -= u[2][point];
    }
    // now  xc, yc, zc have the input lagrangian configuration
    // lagrangeTrans performs the tiling of it
    lagrangeTrans->transformCoordinates(no_points,
                                        &xc[0], &yc[0], &zc[0]);
    float *uout[3];
    coDistributedObject *outObj = NULL;
    coDoVec3 *outUdata = new coDoVec3(name, no_points);
    outUdata->getAddresses(&uout[0], &uout[1], &uout[2]);
    outObj = outUdata;
    // dump tiled eulerian conf.
    geometry.dumpGeometry(uout[0], uout[1], uout[2], true);

    // tiled eulerian conf. - tiling of lagrange conf.
    for (point = 0; point < no_points; ++point)
    {
        uout[0][point] -= xc[point];
        uout[1][point] -= yc[point];
        uout[2][point] -= zc[point];
    }
    return outObj;
}

// output geometry object
const coDistributedObject *
Transform::OutputGeometry(const char *name, // output object name
                          Geometry &geometry, // keep here coordinates of input and output for future use
                          const coDistributedObject *obj, // input object
                          const Matrix &theMatrix, // active transformation
                          int setFlag) // if we are bunching results in a set
{
    coDistributedObject *outObj = NULL;
    static const float tolerance = 1.0e-5f;

    if (setFlag && theMatrix.type() == Matrix::TRIVIAL)
    {
        //obj->incRefCount();
        return retObject(obj);
    }
    // the rest should be as clear as the soup in an orphanage
    if (obj->isType("POLYGN"))
    {
        coDoPolygons *polygons = (coDoPolygons *)(obj);

        // get dimensions
        int nPoints = polygons->getNumPoints();
        int nCorners = polygons->getNumVertices();
        int nPolygons = polygons->getNumPolygons();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[3];
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoPolygons(name, nPoints, coords[0], coords[1], coords[2],
                                  nCorners, cl, nPolygons, pl);
        if (!outObj->objectOk())
            return NULL;
        // now use the matrix to apply the transformation
        float *outCoords[3];
        int *outCl, *outPl;
        ((coDoPolygons *)(outObj))->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2], &outCl, &outPl);

        theMatrix.transformCoordinates(nPoints,
                                       outCoords[0], outCoords[1], outCoords[2]);

        theMatrix.transformLists(nCorners, nPolygons, outCl, outPl);

        geometry.setInfo(coords[0], coords[1], coords[2], nPoints,
                         outCoords[0], outCoords[1], outCoords[2]);
    }
    else if (obj->isType("LINES"))
    {
        //      cerr << "GetSetElem::createNewSimpleObj(..) POLYGN" << endl;
        coDoLines *lines = (coDoLines *)(obj);

        // get dimensions
        int nPoints = lines->getNumPoints();
        int nCorners = lines->getNumVertices();
        int nLines = lines->getNumLines();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[3];
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        lines->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoLines(name, nPoints, coords[0], coords[1], coords[2],
                               nCorners, cl, nLines, pl);
        if (!outObj->objectOk())
            return NULL;
        // now use the matrix to apply the transformation
        float *outCoords[3];
        int *outCl, *outPl;
        ((coDoLines *)(outObj))->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2], &outCl, &outPl);

        theMatrix.transformCoordinates(nPoints,
                                       outCoords[0], outCoords[1], outCoords[2]);

        geometry.setInfo(coords[0], coords[1], coords[2], nPoints,
                         outCoords[0], outCoords[1], outCoords[2]);

        // Line lists are not changed by mirroring!!!!
    }
    else if (obj->isType("POINTS"))
    {
        coDoPoints *lines = (coDoPoints *)(obj);

        // get dimensions
        int nPoints = lines->getNumPoints();

        // create new arrays
        float *coords[3];
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        lines->getAddresses(&coords[0], &coords[1], &coords[2]);

        // create new DO
        outObj = new coDoPoints(name, nPoints, coords[0], coords[1], coords[2]);
        if (!outObj->objectOk())
            return NULL;
        // now use the matrix to apply the transformation
        float *outCoords[3];
        ((coDoPoints *)(outObj))->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2]);

        theMatrix.transformCoordinates(nPoints,
                                       outCoords[0], outCoords[1], outCoords[2]);

        geometry.setInfo(coords[0], coords[1], coords[2], nPoints,
                         outCoords[0], outCoords[1], outCoords[2]);
    }
    else if (obj->isType("TRIANG"))
    {
        coDoTriangleStrips *polygons = (coDoTriangleStrips *)(obj);

        // get dimensions
        int nPoints = polygons->getNumPoints();
        int nCorners = polygons->getNumVertices();
        int nPolygons = polygons->getNumStrips();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[3];
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoTriangleStrips(name, nPoints,
                                        coords[0], coords[1], coords[2],
                                        nCorners, cl, nPolygons, pl);
        // now use the matrix to apply the transformation
        float *outCoords[3];
        int *outCl, *outPl;
        ((coDoTriangleStrips *)(outObj))->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2], &outCl, &outPl);

        theMatrix.transformCoordinates(nPoints,
                                       outCoords[0], outCoords[1], outCoords[2]);

        // trangleStrips lists need a special treatment!!!
        // theMatrix.transformLists(nCorners,nPolygons,outCl,outPl);
        if (theMatrix.getJacobian() == Matrix::NEG_JACOBIAN)
        {
            // this will describe the mirrored geometry "almost" exactly:
            // swap the 2 first vertices, and the next 2 ones etc, etc
            int elem;
            for (elem = 0; elem < nPolygons - 1; ++elem)
            {
                int first = outPl[elem];
                int last = outPl[elem + 1];
                int vert;
                for (vert = first; vert < last - 1; vert += 2)
                {
                    int tmp = outCl[vert];
                    outCl[vert] = outCl[vert + 1];
                    outCl[vert + 1] = tmp;
                }
            }
            int first = outPl[nPolygons - 1];
            int last = nCorners;
            int vert;
            for (vert = first; vert < last - 1; vert += 2)
            {
                int tmp = outCl[vert];
                outCl[vert] = outCl[vert + 1];
                outCl[vert + 1] = tmp;
            }
        }
        geometry.setInfo(coords[0], coords[1], coords[2], nPoints,
                         outCoords[0], outCoords[1], outCoords[2]);
    }
    else if (obj->isType("UNSGRD"))
    {
        coDoUnstructuredGrid *unsgrid = (coDoUnstructuredGrid *)(obj);

        // get dimensions
        int nPoints;
        int nCorners;
        int nElems;
        unsgrid->getGridSize(&nElems, &nCorners, &nPoints);

        // create new arrays
        int *cl, *pl, *tl;
        pl = NULL;
        cl = NULL;
        float *coords[3];
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        unsgrid->getAddresses(&pl, &cl, &coords[0], &coords[1], &coords[2]);
        unsgrid->getTypeList(&tl);

        // create new DO
        outObj = new coDoUnstructuredGrid(name, nElems, nCorners, nPoints,
                                          pl, cl, coords[0], coords[1], coords[2], tl);
        if (!outObj->objectOk())
            return NULL;
        // now use the matrix to apply the transformation
        float *outCoords[3];
        int *outCl, *outPl;
        ((coDoUnstructuredGrid *)(outObj))->getAddresses(&outPl, &outCl, &outCoords[0], &outCoords[1], &outCoords[2]);

        theMatrix.transformCoordinates(nPoints,
                                       outCoords[0], outCoords[1], outCoords[2]);

        theMatrix.transformLists(nCorners, nElems, outCl, outPl, tl);
        geometry.setInfo(coords[0], coords[1], coords[2], nPoints,
                         outCoords[0], outCoords[1], outCoords[2]);
    }
    else if (obj->isType("STRGRD"))
    {
        coDoStructuredGrid *strgrid = (coDoStructuredGrid *)(obj);
        int sx, sy, sz;
        strgrid->getGridSize(&sx, &sy, &sz);
        float *xc, *yc, *zc;
        strgrid->getAddresses(&xc, &yc, &zc);
        outObj = new coDoStructuredGrid(name, sx, sy, sz, xc, yc, zc);
        float *outCoords[3];
        ((coDoStructuredGrid *)(outObj))->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2]);
        theMatrix.transformCoordinates(sx * sy * sz,
                                       outCoords[0], outCoords[1], outCoords[2]);
        geometry.setInfo(xc, yc, zc, sx, sy, sz,
                         outCoords[0], outCoords[1], outCoords[2]);
    }
    else if (obj->isType("UNIGRD") || obj->isType("RCTGRD"))
    {
        // check if transformation is possible!!!
        // problematic transformations are (A) either rotations
        //    but we tolerate rotations which satisfy these 2
        //    conditions:
        //    1. angle is a multiple of 90 and
        //    2. the axis vector coincides with one of the 3 coordinate axis
        // or (B) mirroring with planes that do not coincide
        // with any of the 3 coordinate planes
        coHideParam *axis = NULL;
        if (h_type_->getIValue() == TYPE_MIRROR)
        {
            axis = h_mirror_normal_;
        }
        else if (h_type_->getIValue() == TYPE_MULTI_ROTATE)
        {
            if (fabs(remainder(h_multirot_scalar_->getFValue(), 90)) > tolerance)
            {
                sendWarning("This angle is not supported Multirotation for UNIGRD or RCTGRD");
                return NULL;
            }
            axis = h_multirot_normal_;
        }
        else if (obj->isType("RCTGRD"))
        {
            if (h_type_->getIValue() == TYPE_ROTATE)
            {
                if (fabs(remainder(h_rotate_scalar_->getFValue(), 90)) > tolerance)
                {
                    sendWarning("This angle is not supported for RCTGRD");
                    return NULL;
                }
                axis = h_rotate_normal_;
            }
            if (axis != NULL)
            {
                float x = axis->getFValue(0);
                float y = axis->getFValue(1);
                float z = axis->getFValue(2);
                if (!(
                        (x == 0.0 && y == 0.0)
                        || (x == 0.0 && z == 0.0)
                        || (y == 0.0 && z == 0.0)))
                {
                    sendWarning("For this transformation type only normal directions coincident with one coordinate axis are permitted");
                    return NULL;
                }
            }
        }
        else
        {
            if (h_type_->getIValue() == TYPE_ROTATE)
            {
                // Transforming UniGRD does not work, runs into assert, thus alwazs transAsAttrib   if(fabs(remainder(h_rotate_scalar_->getFValue(),90)) > tolerance)
                {
                    sendWarning("This angle is not supported for UNIGRD, instead I will create a Transformation attribute respected by the Sample module");
                    transformationAsAttribute = true;
                }
                axis = h_rotate_normal_;
            }
            if (axis != NULL)
            {
                float x = axis->getFValue(0);
                float y = axis->getFValue(1);
                float z = axis->getFValue(2);
                if (!(
                        (x == 0.0 && y == 0.0)
                        || (x == 0.0 && z == 0.0)
                        || (y == 0.0 && z == 0.0)))
                {
                    sendWarning("For this transformation type I can do the work only for normal directions coincident with one coordinate axis, instead I will create a Transformation attribute respected by the Sample module");
                    transformationAsAttribute = true;
                }
            }
        }
    }
    if (obj->isType("UNIGRD"))
    {
        coDoUniformGrid *unigrid = (coDoUniformGrid *)(obj);
        int sizes[3];
        int &sx = sizes[0];
        int &sy = sizes[1];
        int &sz = sizes[2];
        float xp[2];
        float yp[2];
        float zp[2];
        unigrid->getGridSize(&sx, &sy, &sz);
        unigrid->getMinMax(xp, xp + 1, yp, yp + 1, zp, zp + 1);

        float xpo[2];
        memcpy(xpo, xp, 2 * sizeof(float));
        float ypo[2];
        memcpy(ypo, yp, 2 * sizeof(float));
        float zpo[2];
        memcpy(zpo, zp, 2 * sizeof(float));

        int sxp, syp, szp;
        if (transformationAsAttribute)
        {
            sxp = sx;
            syp = sy;
            szp = sz;
        }
        else
        {
            theMatrix.transformCoordinates(2, xpo, ypo, zpo);
            theMatrix.reOrder(sx, sy, sz, &sxp, &syp, &szp);
        }

        outObj = new coDoUniformGrid(name, sxp, syp, szp,
                                     xpo[0], xpo[1], ypo[0], ypo[1], zpo[0], zpo[1]);
        geometry.setInfo(xp[0], xp[1], yp[0], yp[1], zp[0], zp[1], sx, sy, sz,
                         xpo[0], xpo[1], ypo[0], ypo[1], zpo[0], zpo[1]);

        if (transformationAsAttribute)
        {
            char transMat[64 * 16];
            char *p = transMat;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int sz = sprintf(p, "%f ", theMatrix.get(j, i));
                    p += sz;
                }
            }

            outObj->addAttribute("Transformation", transMat);
        }
    }
    else if (obj->isType("RCTGRD"))
    {
        coDoRectilinearGrid *rctgrid = (coDoRectilinearGrid *)(obj);
        int sx, sy, sz;
        float *xc, *yc, *zc;
        rctgrid->getGridSize(&sx, &sy, &sz);
        rctgrid->getAddresses(&xc, &yc, &zc);
        std::vector<float> xx(sx);
        std::vector<float> yy(sy);
        std::vector<float> zz(sz);

        std::vector<float> yx(sx);
        std::vector<float> zx(sx);
        int cx;
        for (cx = 0; cx < sx; ++cx)
        {
            xx.push_back(xc[cx]);
            yx.push_back(yc[0]);
            zx.push_back(zc[0]);
        }
        std::vector<float> xy(sy);
        std::vector<float> zy(sy);
        int cy;
        for (cy = 0; cy < sy; ++cy)
        {
            xy.push_back(xc[0]);
            yy.push_back(yc[cy]);
            zy.push_back(zc[0]);
        }
        std::vector<float> xz(sz);
        std::vector<float> yz(sz);
        int cz;
        for (cz = 0; cz < sz; ++cz)
        {
            xz.push_back(xc[0]);
            yz.push_back(yc[0]);
            zz.push_back(zc[cz]);
        }

        // transform X
        float *XMapped[3];
        XMapped[0] = &xx[0];
        XMapped[1] = &yx[0];
        XMapped[2] = &zx[0];
        theMatrix.transformCoordinates(sx, XMapped[0], XMapped[1], XMapped[2]);
        // transform Y
        float *YMapped[3];
        YMapped[0] = &xy[0];
        YMapped[1] = &yy[0];
        YMapped[2] = &zy[0];
        theMatrix.transformCoordinates(sy, YMapped[0], YMapped[1], YMapped[2]);
        // transform Z
        float *ZMapped[3];
        ZMapped[0] = &xz[0];
        ZMapped[1] = &yz[0];
        ZMapped[2] = &zz[0];
        theMatrix.transformCoordinates(sz, ZMapped[0], ZMapped[1], ZMapped[2]);
        // determine new directions
        float Origin[3] = { 0.0, 0.0, 0.0 };
        theMatrix.transformCoordinates(1, Origin, Origin + 1, Origin + 2);
        float Xaxis[3] = { 1.0, 0.0, 0.0 };
        theMatrix.transformCoordinates(1, Xaxis, Xaxis + 1, Xaxis + 2);
        float Yaxis[3] = { 0.0, 1.0, 0.0 };
        theMatrix.transformCoordinates(1, Yaxis, Yaxis + 1, Yaxis + 2);
        float Zaxis[3] = { 0.0, 0.0, 1.0 };
        theMatrix.transformCoordinates(1, Zaxis, Zaxis + 1, Zaxis + 2);

        Xaxis[0] -= Origin[0];
        Xaxis[1] -= Origin[1];
        Xaxis[2] -= Origin[2];
        Yaxis[0] -= Origin[0];
        Yaxis[1] -= Origin[1];
        Yaxis[2] -= Origin[2];
        Zaxis[0] -= Origin[0];
        Zaxis[1] -= Origin[1];
        Zaxis[2] -= Origin[2];
        if (Matrix::Normalise(Xaxis))
        {
            sendWarning("could not normalise Xaxis: This is a bug");
            return NULL;
        }
        if (Matrix::Normalise(Yaxis))
        {
            sendWarning("could not normalise Yaxis: This is a bug");
            return NULL;
        }
        if (Matrix::Normalise(Zaxis))
        {
            sendWarning("could not normalise Zaxis: This is a bug");
            return NULL;
        }

        float *nAxis[3];
        float **oCoord[3];
        oCoord[0] = XMapped;
        oCoord[1] = YMapped;
        oCoord[2] = ZMapped;
        nAxis[0] = Xaxis;
        nAxis[1] = Yaxis;
        nAxis[2] = Zaxis;
        float *nCoord[3] = { NULL, NULL, NULL };
        int oSizes[3];
        oSizes[0] = sx;
        oSizes[1] = sy;
        oSizes[2] = sz;
        int nSizes[3] = { 0, 0, 0 };
        int axis;
        for (axis = 0; axis < 3; ++axis)
        {
            int new_axis = Matrix::WhichAxis(nAxis[axis]);
            nCoord[new_axis] = oCoord[axis][new_axis];
            nSizes[new_axis] = oSizes[axis];
        }
        // now use nCoord !!!
        outObj = new coDoRectilinearGrid(name, nSizes[0], nSizes[1], nSizes[2],
                                         nCoord[0], nCoord[1], nCoord[2]);
        float *outInfo[3];
        ((coDoRectilinearGrid *)(outObj))->getAddresses(&outInfo[0], &outInfo[1], &outInfo[2]);
        geometry.setInfo(xc, sx, yc, sy, zc, sz, outInfo[0], outInfo[1], outInfo[2]);
    }
    copyRotateAttributes(outObj, obj, theMatrix);
    return outObj;
}

// picking up a sign
static void
pickSign(float *scal, int no_points)
{
    int point;
    for (point = 0; point < no_points; ++point)
    {
        scal[point] *= -1.0;
    }
}

// output data objects
const coDistributedObject *
Transform::OutputData(const char *name, // output data object name
                      const coDistributedObject *dataIn, // input data
                      const Geometry &geometry, // used when tiling displacements
                      int dataType, // real vector or scalar, or pseudo.. or displacements
                      const Matrix &matrix, // the eulerian transformation
                      const Matrix *lagrangeTrans, // the lagrangian rtansformation
                      int setFlag) // if we are bunching results in a set or not
{
    if (dataType == 3 && lagrangeTrans && !dataIn->isType("USTVDT"))
    {
        sendWarning("A magnitude declared to be displacements is not a vector field");
        return NULL;
    }

    if (setFlag && matrix.type() == Matrix::TRIVIAL)
    {
        return retObject(dataIn);
    }

    if (dataIn->isType("USTSDT") && geometry.getType() == Geometry::UNS)
    {
        if (dataType == 3)
        {
            sendWarning("The 'Displacements' option is only valid for vector fields");
            return NULL;
        }
        if (setFlag
            && (dataType != 2 || matrix.getJacobian() == Matrix::POS_JACOBIAN))
        {
            return retObject(dataIn);
        }

        coDoFloat *sdata = (coDoFloat *)(dataIn);
        float *u;
        sdata->getAddress(&u);
        int no_points = sdata->getNumPoints();

        coDoFloat *outsdata = new coDoFloat(name, no_points, u);

        if (dataType == 2 && matrix.getJacobian() == Matrix::NEG_JACOBIAN)
        {
            float *uout;
            outsdata->getAddress(&uout);
            pickSign(uout, no_points);
        }
        copyRotateAttributes(outsdata, dataIn, matrix);
        return outsdata;
    }
    else if (dataIn->isType("USTSDT") && geometry.getType() != Geometry::UNS)
    {
        if (dataType == 3)
        {
            sendWarning("The 'Displacements' option is only valid for vector fields");
            return NULL;
        }
        if (setFlag && transformationAsAttribute)
        {
            return retObject(dataIn);
        }
        if (setFlag
            && (dataType != 2 || matrix.getJacobian() == Matrix::POS_JACOBIAN))
        {
            return retObject(dataIn);
        }

        // retObject requires set condition, and (str geometry or no reordering),
        // and no sign picking is necessary
        if (setFlag // set
            // str geometry
            && (geometry.getType() == Geometry::STR
                || matrix.type() != Matrix::ROTATION // no reordering
                )
            && (dataType != 2 // no sign picking
                || matrix.getJacobian() == Matrix::POS_JACOBIAN))
        {
            return retObject(dataIn);
        }

        coDoFloat *sdata = (coDoFloat *)(dataIn);
        float *u;
        sdata->getAddress(&u);
        int nelem = sdata->getNumPoints();

        coDoFloat *outsdata = NULL;
        if (transformationAsAttribute)
        {
            outsdata = new coDoFloat(name, nelem, u);
        }
        else if (geometry.getType() != Geometry::STR
                 && matrix.type() == Matrix::ROTATION) //(rct or unigrd)and rotation
        {
#if 0
         int nnx,nny,nnz;
         matrix.reOrder(nx,ny,nz,&nnx,&nny,&nnz);
         outsdata = new coDoFloat(name,nelem);
         float *uout;
         outsdata->getAddress(&uout);
         matrix.reOrder(u,uout,nx,ny,nz);
#else
            assert("hier braucht man das gitter" == NULL);
#endif
        }
        else // str or not a rotation
        {
            outsdata = new coDoFloat(name, nelem, u);
            float *uout;
            outsdata->getAddress(&uout);
            if (matrix.getJacobian() == Matrix::NEG_JACOBIAN
                && dataType == 2)
            {
                pickSign(uout, nelem);
            }
        }
        copyRotateAttributes(outsdata, dataIn, matrix);
        return outsdata;
    }
    else if (dataIn->isType("USTVDT") && geometry.getType() == Geometry::UNS)
    {
        coDoVec3 *vdata = (coDoVec3 *)(dataIn);
        float *u[3];
        int no_points = vdata->getNumPoints();
        vdata->getAddresses(&u[0], &u[1], &u[2]);
        coDoVec3 *outdata = NULL;
        if (dataType != 3 || lagrangeTrans == NULL)
        {
            outdata = new coDoVec3(name, no_points, u[0], u[1], u[2]);
            float *uout[3];
            outdata->getAddresses(&uout[0], &uout[1], &uout[2]);
            matrix.transformVector(no_points, uout);
            if (dataType == 2 && matrix.getJacobian() == Matrix::NEG_JACOBIAN)
            {
                pickSign(uout[0], no_points);
                pickSign(uout[1], no_points);
                pickSign(uout[2], no_points);
            }
        }
        else // tiling and displacements!!!!
        {
            if (geometry.getSize() != no_points)
            {
                sendWarning("When tiling, displacements are expected per point");
                return NULL;
            }
            outdata = (coDoVec3 *)(TilingFormula(name, geometry, u, lagrangeTrans));
        }
        copyRotateAttributes(outdata, dataIn, matrix);
        return outdata;
    }
    else if (dataIn->isType("USTVDT") && geometry.getType() != Geometry::UNS)
    {
        coDoVec3 *vdata = (coDoVec3 *)(dataIn);
        int nelem = vdata->getNumPoints();
        float *u[3];
        vdata->getAddresses(&u[0], &u[1], &u[2]);
        coDoVec3 *outvdata = NULL;
        if (dataType == 3 && lagrangeTrans != NULL)
        {
            int ngx, ngy, ngz;
            geometry.getSize(&ngx, &ngy, &ngz);
            if (ngx * ngy * ngz != nelem)
            {
                sendWarning("Displacement and geometry sizes do not coincide");
            }
            outvdata = (coDoVec3 *)(TilingFormula(name, geometry, u, lagrangeTrans));
        }
        else if (geometry.getType() != Geometry::STR
                 && matrix.type() == Matrix::ROTATION) //(rct or unigrd)and rotation
        {
#if 0
         int nnx,nny,nnz;
         matrix.reOrder(nx,ny,nz,&nnx,&nny,&nnz);
         outvdata = new coDoVec3(name,nnx,nny,nnz);
         if(!transformationAsAttribute)
         {
            float *uout[3];
            outvdata->getAddresses(&uout[0],&uout[1],&uout[2]);
            matrix.reOrderAndTransform(u,uout,nx,ny,nz);
         }
#else
            assert("hier braucht man das gitter" == NULL);
#endif
        }
        else // str or not a rotation
        {
            outvdata = new coDoVec3(name, nelem, u[0], u[1], u[2]);
            float *uout[3];
            outvdata->getAddresses(&uout[0], &uout[1], &uout[2]);
            matrix.transformVector(nelem, uout);
            if (matrix.getJacobian() == Matrix::NEG_JACOBIAN
                && dataType == 2)
            {
                pickSign(uout[0], nelem);
                pickSign(uout[1], nelem);
                pickSign(uout[2], nelem);
            }
        }
        copyRotateAttributes(outvdata, dataIn, matrix);
        return outvdata;
    }
    sendWarning("Only scalar and vector fields are supported");
    return NULL;
}

// which data port has been tagged with the option displacements??
int
Transform::lookForDisplacements()
{
    int port;
    int ret = -1; // no port has the tag
    for (port = 0; port < NUM_DATA_IN_PORTS; ++port)
    {
        if (ret == -1 && h_dataType_[port]->getIValue() == 2)
        {
            ret = port;
            continue;
        }
        if (ret >= 0 && h_dataType_[port]->getIValue() == 2)
        {
            ret = -2; // if two ports have this tag
            break;
        }
    }
    return ret;
}

// copy attributes with especial handling of ROTATE_POINT, etc, etc
void
Transform::copyRotateAttributes(coDistributedObject *tgt,
                                const coDistributedObject *src,
                                const Matrix &matrix) const
{
    int n;
    const char **name, **setting;

    if (src && tgt)
    {
        n = src->getAllAttributes(&name, &setting);
        if (n > 0)
        {
            int attr;
            float buf[3];
            char newSetting[1024];
            for (attr = 0; attr < n; ++attr)
            {
                if (strcmp(name[attr], "ROTATE_POINT") == 0)
                {
                    if (sscanf(setting[attr], "%f %f %f", &(buf[0]), &(buf[1]), &(buf[2])) != 3)
                    {
                        fprintf(stderr, "Transform::copyRotateAttributes: sscanf1 failed\n");
                    }
                    matrix.transformCoordinates(1, buf, buf + 1, buf + 2);
                    sprintf(newSetting, "%f %f %f", buf[0], buf[1], buf[2]);
                    tgt->addAttribute(name[attr], newSetting);
                }
                if (strcmp(name[attr], "ROTATE_VECTOR") == 0)
                {
                    float *vector[3];
                    vector[0] = buf;
                    vector[1] = buf + 1;
                    vector[2] = buf + 2;
                    if (sscanf(setting[attr], "%f %f %f", &(buf[0]), &(buf[1]), &(buf[2])) != 3)
                    {
                        fprintf(stderr, "Transform::copyRotateAttributes: sscanf2 failed\n");
                    }
                    matrix.transformVector(1, vector);
                    sprintf(newSetting, "%f %f %f", buf[0], buf[1], buf[2]);
                    tgt->addAttribute(name[attr], newSetting);
                }
                if (strcmp(name[attr], "ROTATE_ANGLE") == 0
                    && matrix.getJacobian() == Matrix::NEG_JACOBIAN)
                {
                    float angle;
                    if (sscanf(setting[attr], "%f", &angle) != 3)
                    {
                        fprintf(stderr, "Transform::copyRotateAttributes: sscanf3 failed\n");
                    }
                    angle *= -1.0;
                    sprintf(newSetting, "%f", angle);
                    tgt->addAttribute(name[attr], newSetting);
                }
                else
                {
                    // never copy TRANSFORM attribute
                    if (strcmp(name[attr], "TRANSFORM") != 0)
                    {
                        tgt->addAttribute(name[attr], setting[attr]);
                    }
                }
            }
        }
    }
}
