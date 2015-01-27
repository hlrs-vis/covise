/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MergeAndNormals.h"
#include "ReadASCIIDyna.h"
#include <do/coDoText.h>
#include <do/coDoSet.h>

#include <sstream>
#include <vector>

MergeAndNormals::MergeAndNormals(int argc, char *argv[])
    : coSimpleModule(argc, argv, "partial node merging and normal generation")
{
    p_inGeom_ = addInputPort("InGeometry", "coDoPolygons|coDoLines", "input geometry");
    p_inNormals_ = addInputPort("InNormals", "coDoVec3", "input normals");
    p_inNormals_->setRequired(0);
    p_text_ = addInputPort("Text", "coDoText", "where to merge");
    p_outGeom_ = addOutputPort("OutGeometry", "coDoPolygons|coDoLines", "output geometry");
    p_Normals_ = addOutputPort("OutNormals", "coDoVec3|DO_Unstructured_V3D_Normals", "output normals");
}

MergeAndNormals::~MergeAndNormals()
{
}

int
MergeAndNormals::compute(const char *port)
{
    (void)port; // silence warning

    if (!preOK_)
    {
        sendWarning("preHandleObjects failed");
        return FAIL;
    }
    const coDistributedObject *in_obj = p_inGeom_->getCurrentObject();
    if (!in_obj->isType("LINES") && !in_obj->isType("POLYGN"))
    {
        sendWarning("Invalid input type");
        return FAIL;
    }
    if (in_obj->isType("LINES"))
    {
        if (!p_inNormals_->isConnected())
        {
            coDoLines *lines = (coDoLines *)(in_obj);

            // get dimensions
            int nPoints = lines->getNumPoints();
            int nCorners = lines->getNumVertices();
            int nLines = lines->getNumLines();

            // create new arrays
            int *cl, *ll;
            ll = NULL;
            cl = NULL;
            float *coords[3];
            int i;
            for (i = 0; i < 3; ++i)
                coords[i] = NULL;

            lines->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &ll);

            // create new DO
            coDoLines *outObj = new coDoLines(p_outGeom_->getObjName(), nPoints, coords[0], coords[1], coords[2],
                                              nCorners, cl, nLines, ll);
            p_outGeom_->setCurrentObject(outObj);
        }
        else
        {
            coDoLines *outObj = new coDoLines(p_outGeom_->getObjName(), 0, 0, 0);
            p_outGeom_->setCurrentObject(outObj);
        }
        p_Normals_->setCurrentObject(new coDoVec3(p_Normals_->getObjName(), 0));
        return SUCCESS;
    }
    coDoPolygons *preNormals = (coDoPolygons *)in_obj;

    // if we are using input normals... then everything
    // is much easier.
    if (p_inNormals_->isConnected())
    {
        const coDistributedObject *in_normals = p_inNormals_->getCurrentObject();
        if (!in_normals || !in_normals->isType("USTVDT"))
        {
            sendWarning("Incorrect type of input normals or NULL pointer");
            return FAIL;
        }
        coDoVec3 *inNormals = (coDoVec3 *)in_normals;
        if (ProjectNormals(preNormals, inNormals) == 0)
        {
            return SUCCESS;
        }
        else
        {
            return FAIL;
        }
    }

    if (merge_)
    {
        // get arrays...
        int no_poly = preNormals->getNumPolygons();
        int no_vert = preNormals->getNumVertices();
        int no_point = preNormals->getNumPoints();
        float *x_c;
        float *y_c;
        float *z_c;
        int *v_l;
        int *l_l;
        preNormals->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

        // make vectors out of them
        vector<float> xc, yc, zc;
        vector<int> vl, ll;
        int point;
        for (point = 0; point < no_point; ++point)
        {
            xc.push_back(x_c[point]);
            yc.push_back(y_c[point]);
            zc.push_back(z_c[point]);
        }
        int vert;
        for (vert = 0; vert < no_vert; ++vert)
        {
            vl.push_back(v_l[vert]);
        }
        int poly;
        for (poly = 0; poly < no_poly; ++poly)
        {
            ll.push_back(l_l[poly]);
        }
        // call MergeNodes
        if (mergeNodes_)
        {
            ReadASCIIDyna::MergeNodes(xc, yc, zc, vl, ll, 1e-6, grundZellenBreite_, grundZellenHoehe_);
        }
        // redefine forNormals...
        {
            float *x, *y, *z;
            int *v, *l;
            preNormals = new coDoPolygons(p_outGeom_->getObjName(),
                                          xc.size(), vl.size(), ll.size());
            preNormals->getAddresses(&x, &y, &z, &v, &l);
            copy(xc.begin(), xc.end(), x);
            copy(yc.begin(), yc.end(), y);
            copy(zc.begin(), zc.end(), z);
            copy(vl.begin(), vl.end(), v);
            copy(ll.begin(), ll.end(), l);
        }
    }
    else
    {
        // make a copy for output of the geometry
        // get dimensions
        int nPoints = preNormals->getNumPoints();
        int nCorners = preNormals->getNumVertices();
        int nPolygons = preNormals->getNumPolygons();

        // create new arrays
        int *cl, *pl;
        pl = NULL;
        cl = NULL;
        float *coords[3];
        int i;
        for (i = 0; i < 3; ++i)
            coords[i] = NULL;

        preNormals->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

        // create new DO
        preNormals = new coDoPolygons(p_outGeom_->getObjName(),
                                      nPoints, coords[0], coords[1], coords[2],
                                      nCorners, cl, nPolygons, pl);
    }
    p_outGeom_->setCurrentObject(preNormals);
    // normals for nodes satisfying z==0.0
    // get arrays...
    //int no_poly = preNormals->getNumPolygons();
    //int no_vert = preNormals->getNumVertices();
    int no_point = preNormals->getNumPoints();
    float *x_c;
    float *y_c;
    float *z_c;
    int *v_l;
    int *l_l;
    preNormals->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    vector<float> nx, ny, nz;
    int point;
    for (point = 0; point < no_point; ++point)
    {
        nx.push_back(0.0);
        ny.push_back(0.0);
        if (z_c[point] == 0.0)
        {
            nz.push_back(1.0);
        }
        else
        {
            nz.push_back(0.0);
        }
    }
    {
        coDoVec3 *vobj = new coDoVec3(
            p_Normals_->getObjName(), nx.size());
        float *x, *y, *z;
        vobj->getAddresses(&x, &y, &z);
        copy(nx.begin(), nx.end(), x);
        copy(ny.begin(), ny.end(), y);
        copy(nz.begin(), nz.end(), z);
        p_Normals_->setCurrentObject(vobj);
    }
    return SUCCESS;
}

void
MergeAndNormals::preHandleObjects(coInputPort **in_ports)
{
    preOK_ = true;
    merge_ = false;
    const coDistributedObject *intext = in_ports[2]->getCurrentObject();
    if (!intext || !intext->isType("DOTEXT"))
    {
        return;
    }
    coDoText *theText = (coDoText *)intext;
    int size = theText->getTextLength();
    if (size == 0)
    {
        return;
    }
    char *text;
    theText->getAddress(&text);
    istringstream strText;
    strText.str(string(text));
    int maxLen = strlen(text) + 1;
    std::vector<char> name(maxLen);
    readGrundZellenHoehe_ = false;
    readGrundZellenBreite_ = false;
    grundZellenHoehe_ = 0.0;
    grundZellenBreite_ = 0.0;
    mergeNodes_ = 0;
    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "grundZellenHoehe") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &grundZellenHoehe_, maxLen) != 0)
            {
                sendWarning("Could not read grundZellenHoehe");
                return;
            }
            readGrundZellenHoehe_ = true;
        }
        else if (strcmp(&name[0], "grundZellenBreite") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &grundZellenBreite_, maxLen) != 0)
            {
                sendWarning("Could not read grundZellenBreite");
                return;
            }
            readGrundZellenBreite_ = true;
        }
        /*
            else if(strcmp(&name[0],"mergeNodes")==0){
               if(ReadASCIIDyna::readBoolean(strText,&mergeNodes_,maxLen)!=0){
                  sendWarning("Could not read mergeNodes");
                  return;
               }
               readMergeNodes_ = true;
            }
      */
    }
    if (!readGrundZellenHoehe_
        || !readGrundZellenBreite_
           // || !readMergeNodes_
        )
    {
        preOK_ = false;
        return;
    }
    // merge_ = true;
}

void
Project(float &vx, float &vy, float &vz, float nx, float ny, float nz)
{
    float proj = vx * nx + vy * ny + vz * nz;
    vx -= proj * nx;
    vy -= proj * ny;
    vz -= proj * nz;
}

#include <string>

int
MergeAndNormals::ProjectNormals(coDoPolygons *poly, coDoVec3 *norm)
{
    // unpack coordinates, loop over them, and correct
    // normals for coordintes satisfying the gemetry condition
    //int no_poly = poly->getNumPolygons();
    //int no_vert = poly->getNumVertices();
    int no_point = poly->getNumPoints();
    float *x_c;
    float *y_c;
    float *z_c;
    int *v_l;
    int *l_l;
    poly->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    if (no_point != norm->getNumPoints())
    {
        sendWarning("Number of points of the input normals does not match the number of points in polygons");
        return -1;
    }
    coDistributedObject *polyList[2];
    polyList[0] = poly;
    poly->incRefCount();
    polyList[1] = NULL;
    coDoSet *polySet = new coDoSet(p_outGeom_->getObjName(), polyList);
    p_outGeom_->setCurrentObject(polySet);

    float *nx;
    float *ny;
    float *nz;
    norm->getAddresses(&nx, &ny, &nz);
    vector<float> nX, nY, nZ;

    int point;
    for (point = 0; point < no_point; ++point)
    {
        nX.push_back(nx[point]);
        nY.push_back(ny[point]);
        nZ.push_back(nz[point]);

        if (z_c[point] == 0.0)
        {
            nX[point] = 0.0;
            nY[point] = 0.0;
            nZ[point] = 1.0;
            continue;
        }

        float resX = 0.0, resY = 0.0;
        if (grundZellenBreite_ != 0.0)
        {
#if defined(__APPLE__) || defined(__MINGW32__)
            resX = remainder(x_c[point], grundZellenBreite_);
#else
            resX = drem(x_c[point], grundZellenBreite_);
#endif
        }
        if (grundZellenHoehe_ != 0.0)
        {
#if defined(__APPLE__) || defined(__MINGW32__)
            resY = remainder(y_c[point], grundZellenHoehe_);
#else
            resY = drem(y_c[point], grundZellenHoehe_);
#endif
        }
        if (fabs(resX) <= 1e-6)
        {
            // project onto YZ plane
            Project(nX[point], nY[point], nZ[point], 1.0, 0.0, 0.0);
        }
        if (fabs(resY) <= 1e-6)
        {
            // project onto ZX plane
            Project(nX[point], nY[point], nZ[point], 0.0, 1.0, 0.0);
        }
    }
    coDistributedObject *normList[2];
    normList[1] = NULL;
    string name = p_Normals_->getObjName();
    name += "_0";
    {
        normList[0] = new coDoVec3(name.c_str(), nX.size());
        float *x, *y, *z;
        ((coDoVec3 *)normList[0])->getAddresses(&x, &y, &z);
        copy(nX.begin(), nX.end(), x);
        copy(nY.begin(), nY.end(), y);
        copy(nZ.begin(), nZ.end(), z);
    }
    p_Normals_->setCurrentObject(new coDoSet(p_Normals_->getObjName(), normList));
    delete normList[0];
    return 0;
}

MODULE_MAIN(SCA, MergeAndNormals)
