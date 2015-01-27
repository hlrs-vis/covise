/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                    (C) 2000 VirCinity  **
 ** Description: axe-murder geometry                                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Lars Frenzel                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
 **        18.10.2000 V1.0 new API, several data ports, triangle strips    **
 **                                               ( converted to polygons )**
 **			Sven Kufer 					  **
\**************************************************************************/

#include <iostream.h>
#include "ChangeAttribute.h"

ChangeAttribute::ChangeAttribute()
    : coModule("dfdsf")
{
    int i;
    char portname[32];
    //param
    p_rmvAttr = addStringParam("remove", "remove attr");
    p_rmvAttr->setValue("nothing");

    p_chAttr = addStringParam("change_attribute", "change attr");
    p_chAttr->setValue("from_nothing");

    p_chAttrVal = addStringParam("change_to_value", "change attr");
    p_chAttrVal->setValue("to_nothing");

    // ports
    p_geo_in = addInputPort("geo_in", "coDoSet|coDoIntArr|coDoUniformGrid|coDoRectilinearGrid|coDoStructuredGrid|coDoUnstructuredGrid|coDoPoints|coDoLines|coDoPolygons|coDoTriangleStrips|Float|Vec3|Float|Vec3", "geometry");
    p_geo_out = addOutputPort("geo_out", "coDoSet|coDoIntArr|coDoUniformGrid|coDoRectilinearGrid|coDoStructuredGrid|coDoUnstructuredGrid|coDoPoints|coDoLines|coDoPolygons|coDoTriangleStrips|Float|Vec3|Float|Vec3", "geometry");
}

int main(int argc, char *argv[])
{
    ChangeAttribute *application = new ChangeAttribute;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something

int ChangeAttribute::compute()
{
    // in objectsp_geo_in

    if (p_geo_in->getCurrentObject()->isType("SETELE"))
    {
        coDoSet *s_in = (coDoSet *)p_geo_in->getCurrentObject();

        int i, num = 0;
        coDistributedObject *const *elems = s_in->getAllElements(&num);

        coDistributedObject **new_elems = new coDistributedObject *[num + 1];
        new_elems[num] = 0;

        for (i = 0; i < num; i++)
        {
            elems[i]->incRefCount();
            new_elems[i] = elems[i];
        }

        coDoSet *s_out = new coDoSet(p_geo_out->getObjName(), new_elems);
        copyAttributes(s_out, s_in);

        p_geo_out->setCurrentObject(s_out);
        delete[] new_elems;
    }
    else if (p_geo_in->getCurrentObject()->isType("GEOMET"))
    {
        coDoGeometry *g_in = (coDoGeometry *)p_geo_in->getCurrentObject();
        coDistributedObject *g = g_in->getGeometry();
        g->incRefCount();

        coDoGeometry *g_out = new coDoGeometry(p_geo_out->getObjName(), g);
        copyAttributes(g_out, g_in);

        coDistributedObject *colors = g_in->get_colors();
        if (colors)
        {
            colors->incRefCount();
            g_out->setColor(0, colors);
        }

        coDistributedObject *normals = g_in->get_normals();
        if (normals)
        {
            normals->incRefCount();
            g_out->setNormal(0, normals);
        }

        coDistributedObject *texture = g_in->getTexture();
        if (texture)
        {
            texture->incRefCount();
            g_out->setTexture(0, texture);
        }
        p_geo_out->setCurrentObject(g_out);
    }
    else
    {

        coDistributedObject *out = createNewObj(p_geo_out->getObjName(), p_geo_in->getCurrentObject());
        copyAttributes(out, p_geo_in->getCurrentObject());

        p_geo_out->setCurrentObject(out);
    }
}

coDistributedObject *
ChangeAttribute::createNewObj(const char *name, coDistributedObject *obj) const
{
    const int DIM = 3;

    // return NULL for empty input
    if (!(obj))
        return NULL;
    if (!(name))
        return NULL;

    coDistributedObject *outObj = NULL;

    // SETELE
    if (obj->isType("SETELE"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) SETELE" << endl;
        coDoSet *shmSubSet;
        if (!(shmSubSet = (coDoSet *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::compute( ) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int numSubSets = shmSubSet->getNumElements();
        coDistributedObject **subSetElements = new coDistributedObject *[numSubSets + 1];
        int i;
        for (i = 0; i < numSubSets; ++i)
        {
            coDistributedObject *subSetEle = shmSubSet->getElement(i);
            subSetEle->incRefCount();
            subSetElements[i] = subSetEle;
        }
        subSetElements[numSubSets] = NULL;
        outObj = new coDoSet(name, subSetElements);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // Int array
    else if (obj->isType("INTARR"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) UNIGRD" << endl;
        coDoIntArr *iArr;
        if (!(iArr = (coDoIntArr *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int numDim = iArr->getNumDimensions();
        const int *sizeArr = iArr->getDimensionPtr();
        const int *values = iArr->getAddress();

        outObj = new coDoIntArr(name, numDim, sizeArr, values);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNIFORM GRID
    else if (obj->isType("UNIGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) UNIGRD" << endl;
        coDoUniformGrid *uGrd;
        if (!(uGrd = (coDoUniformGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int nX, nY, nZ;
        uGrd->getGridSize(&nX, &nY, &nZ);

        float xMin, xMax, yMin, yMax, zMin, zMax;
        uGrd->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);

        outObj = new coDoUniformGrid(name, nX, nY, nZ, xMin, xMax, yMin, yMax, zMin, zMax);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }

    // RECTILINEAR GRID
    else if (obj->isType("RCTGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) RCTGRD" << endl;
        coDoRectilinearGrid *rGrd;
        if (!(rGrd = (coDoRectilinearGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int nX, nY, nZ;
        rGrd->getGridSize(&nX, &nY, &nZ);

        float *X, *Y, *Z;
        rGrd->getAddresses(&X, &Y, &Z);

        outObj = new coDoRectilinearGrid(name, nX, nY, nZ, X, Y, Z);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // STRUCTURED GRID
    else if (obj->isType("STRGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) STRGRD" << endl;
        coDoStructuredGrid *sGrd;
        if (!(sGrd = (coDoStructuredGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int nX, nY, nZ;
        sGrd->getGridSize(&nX, &nY, &nZ);
        float *X, *Y, *Z;
        sGrd->getAddresses(&X, &Y, &Z);

        outObj = new coDoStructuredGrid(name, nX, nY, nZ, X, Y, Z);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNSGRD
    else if (obj->isType("UNSGRD"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) UNSGRD" << endl;

        coDoUnstructuredGrid *unsGrd;
        if (!(unsGrd = (coDoUnstructuredGrid *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nElem, nConn, nCoords;
        unsGrd->getGridSize(&nElem, &nConn, &nCoords);

        // create new arrays
        int *el, *cl, *tl;
        el = NULL;
        cl = NULL;
        tl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        unsGrd->getAddresses(&el, &cl, &coords[0], &coords[1], &coords[2]);
        if (unsGrd->has_type_list())
        {
            unsGrd->getTypeList(&tl);
        }

        // create new DO
        outObj = new coDoUnstructuredGrid(name, nElem, nConn, nCoords, el, cl,
                                          coords[0], coords[1], coords[2], tl);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // POINTS
    else if (obj->isType("POINTS"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) POINTS" << endl;
        coDoPoints *points;
        if (!(points = (coDoPoints *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = points->getNumPoints();

        // new pointers
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        points->getAddresses(&coords[0], &coords[1], &coords[2]);

        // create new DO
        outObj = new coDoPoints(name, nPoints, coords[0], coords[1], coords[2]);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // LINES
    else if (obj->isType("LINES"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) LINES" << endl;
        coDoLines *lines;
        if (!(lines = (coDoLines *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = lines->getNumPoints();
        int nCorners = lines->getNumVertices();
        int nLines = lines->getNumLines();

        // create new arrays
        int *cl, *ll;
        ll = NULL;
        cl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        lines->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &ll);

        // create new DO
        outObj = new coDoLines(name, nPoints, coords[0], coords[1], coords[2],
                               nCorners, cl, nLines, ll);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // POLYGONS
    else if (obj->isType("POLYGN"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) POLYGN" << endl;
        coDoPolygons *polygons;
        if (!(polygons = (coDoPolygons *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = polygons->getNumPoints();
        int nCorners = polygons->getNumVertices();
        int nPolygons = polygons->getNumPolygons();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoPolygons(name, nPoints, coords[0], coords[1], coords[2],
                                  nCorners, cl, nPolygons, pl);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // TRIANGLE STRIPS
    else if (obj->isType("TRIANG"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) TRIANG" << endl;
        coDoTriangleStrips *triangleStrips;
        if (!(triangleStrips = (coDoTriangleStrips *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        // get dimensions
        int nPoints = triangleStrips->getNumPoints();
        int nCorners = triangleStrips->getNumVertices();
        int nStrips = triangleStrips->getNumStrips();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[DIM];
        int i;
        for (i = 0; i < DIM; ++i)
            coords[i] = NULL;

        triangleStrips->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        outObj = new coDoTriangleStrips(name, nPoints, coords[0], coords[1], coords[2],
                                        nCorners, cl, nStrips, pl);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // VOLUMES
    else if (obj->isType("VOLUME"))
    {
        cerr << "GetSetElem::createNewSimpleObj(..) VOLUME not supported" << endl;
    }
    // STRUCTURED SCALAR DATA
    else if (obj->isType("STRSDT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) STRSDT" << endl;
        coDoFloat *sData;
        if (!(sData = (coDoFloat *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int nX, nY, nZ;
        float *dat;

        sData->getGridSize(&nX, &nY, &nZ);
        sData->getAddress(&dat);

        outObj = new coDoFloat(name, nX, nY, nZ, dat);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // STRUCTURED VECTOR DATA
    else if (obj->isType("STRVDT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) STRVDT" << endl;
        coDoVec3 *vData;
        if (!(vData = (coDoVec3 *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int nX, nY, nZ;
        float *dat[DIM];

        vData->getGridSize(&nX, &nY, &nZ);
        vData->getAddresses(&dat[0], &dat[1], &dat[2]);

        outObj = new coDoVec3(name, nX, nY, nZ, dat[0], dat[1], dat[2]);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNSTRUCTURED SCALAR DATA
    else if (obj->isType("USTSDT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) USTSDT" << endl;
        coDoFloat *sData;
        if (!(sData = (coDoFloat *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }
        int n = sData->getNumPoints();
        float *dat;
        sData->getAddress(&dat);

        outObj = new coDoFloat(name, n, dat);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }
    // UNSTRUCTURED VECTOR DATA
    else if (obj->isType("USTVDT"))
    {
        //	cerr << "GetSetElem::createNewSimpleObj(..) USTVDT" << endl;
        coDoVec3 *vData;
        if (!(vData = (coDoVec3 *)(obj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::createNewSimpleObj(..) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
            return NULL;
        }

        int n = vData->getNumPoints();
        float *dat[DIM];

        vData->getAddresses(&dat[0], &dat[1], &dat[2]);

        outObj = new coDoVec3(name, n, dat[0], dat[1], dat[2]);
        if (!outObj->objectOk())
            return NULL;
        return outObj;
    }

    return outObj;
}

void
ChangeAttribute::copyAttributes(coDistributedObject *tgt,
                                coDistributedObject *src)
{
    int n;
    const char **name, **setting;

    const char *rmv_attr = p_rmvAttr->getValue();
    const char *ch_attr = p_chAttr->getValue();
    const char *ch_attr_v = p_chAttrVal->getValue();
    bool changed = false;

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
                if (strcmp(name[attr], ch_attr) == 0)
                {
                    tgt->addAttribute(name[attr], ch_attr_v);
                    changed = true;
                }
                else
                {
                    // never copy TRANSFORM attribute
                    if (strcmp(name[attr], rmv_attr) != 0)
                    {
                        tgt->addAttribute(name[attr], setting[attr]);
                    }
                }
            }
            if (!changed)
            {
                tgt->addAttribute(ch_attr, ch_attr_v);
            }
        }
        else if (strcmp(ch_attr, "from_nothing"))
        {
            tgt->addAttribute(ch_attr, ch_attr_v);
        }
    }
}
