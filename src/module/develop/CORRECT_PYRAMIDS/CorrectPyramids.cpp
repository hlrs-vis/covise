/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "CorrectPyramids.h"
#include "covise_gridmethods.h"

CorrectPyramids::CorrectPyramids()
    : coSimpleModule("Correct pyramids")
{
    _p_in_grid = addInputPort("meshIn", "coDoUnstructuredGrid", "input mesh");
    _p_out_grid = addOutputPort("meshOut", "coDoUnstructuredGrid", "output mesh");
    _p_volume = addFloatParam("relative_volume", "relative volume");
    _p_volume->setValue(0.05);
}

int main(int argc, char *argv[])
{
    CorrectPyramids *application = new CorrectPyramids;

    application->start(argc, argv);

    return 0;
}

CorrectPyramids::~CorrectPyramids()
{
}

int
CorrectPyramids::compute()
{
    //coDoUnstructuredGrid *grid = (
    coDistributedObject *in_grid = _p_in_grid->getCurrentObject();
    if (!in_grid->isType("UNSGRD"))
    {
        sendError("Only unstructured grids are acceptable");
        return STOP_PIPELINE;
    }

    coDoUnstructuredGrid *InGrid = (coDoUnstructuredGrid *)in_grid;
    int *el, *vl;
    float *xc, *yc, *zc;
    InGrid->getAddresses(&el, &vl, &xc, &yc, &zc);
    int no_el, no_v, no_p;
    InGrid->getGridSize(&no_el, &no_v, &no_p);
    int *tl;
    InGrid->getTypeList(&tl);

    vector<int> el_out;
    vector<int> vl_out;
    vector<int> tl_out;
    /*
      vector<float> xc_out;
      vector<float> yc_out;
      vector<float> zc_out;
      insert_iterator<vector<float> > it_xc_out(xc_out,xc_out.end());
      insert_iterator<vector<float> > it_yc_out(yc_out,yc_out.end());
      insert_iterator<vector<float> > it_zc_out(zc_out,zc_out.end());
      copy(xc,xc+no_p,it_xc_out);
      copy(yc,yc+no_p,it_yc_out);
      copy(zc,zc+no_p,it_zc_out);
   */
    int element;
    for (element = 0; element < no_el; ++element)
    {
        int vertex = -1;
        if (tl[element] != TYPE_PYRAMID
            || (vertex = PyramidProblem(element, el, vl, xc, yc, zc)) == -1)
        {
            tl_out.push_back(tl[element]);
            el_out.push_back(vl_out.size());
            int no_vert = NumberOfVertices(tl[element]);
            int vert;
            for (vert = 0; vert < no_vert; ++vert)
            {
                vl_out.push_back(vl[el[element] + vert]);
            }
        }
        else
        {
            tl_out.push_back(TYPE_TETRAHEDER);
            el_out.push_back(vl_out.size());
            int no_vert = 4;
            int vert;
            for (vert = 0; vert < no_vert; ++vert)
            {
                if (vert == vertex)
                {
                    continue;
                }
                vl_out.push_back(vl[el[element] + vert]);
            }
        }
    }
    // I do not care about unused points... FixUsg if necessary

    coDoUnstructuredGrid *OutGrid = new coDoUnstructuredGrid(_p_out_grid->getObjName(),
                                                             el_out.size(), vl_out.size(), no_p, 1);
    int *el_out_a;
    int *vl_out_a;
    int *tl_out_a;
    float *xc_out_a;
    float *yc_out_a;
    float *zc_out_a;
    OutGrid->getAddresses(&el_out_a, &vl_out_a, &xc_out_a, &yc_out_a, &zc_out_a);
    OutGrid->getTypeList(&tl_out_a);
    copy(el_out.begin(), el_out.end(), el_out_a);
    copy(vl_out.begin(), vl_out.end(), vl_out_a);
    copy(tl_out.begin(), tl_out.end(), tl_out_a);
    copy(xc, xc + no_p, xc_out_a);
    copy(yc, yc + no_p, yc_out_a);
    copy(zc, zc + no_p, zc_out_a);
    _p_out_grid->setCurrentObject(OutGrid);
    return CONTINUE_PIPELINE;
}

int
CorrectPyramids::PyramidProblem(int element, const int *el, const int *vl,
                                const float *xc, const float *yc, const float *zc)
{
    float p[5][3];
    int base = el[element];
    int i;
    for (i = 0; i < 5; ++i)
    {
        int index = vl[base + i];
        p[i][0] = xc[index];
        p[i][1] = yc[index];
        p[i][2] = zc[index];
    }
    float voltot = fabs(grid_methods::tetra_vol(p[0], p[1], p[2], p[4])) + fabs(grid_methods::tetra_vol(p[0], p[2], p[3], p[4]));
    float volrel = _p_volume->getValue();
    for (i = 0; i < 4; ++i)
    {
        int p0 = (i - 1) % 4;
        int p1 = i;
        int p2 = (i + 1) % 4;
        if (fabs(grid_methods::tetra_vol(p[p0], p[p1], p[p2], p[4])) < volrel * voltot)
        {
            return i;
        }
    }
    return -1;
}

int
CorrectPyramids::NumberOfVertices(int type)
{
    switch (type)
    {
    case TYPE_HEXAEDER:
        return 8;
    case TYPE_PRISM:
        return 6;
    case TYPE_PYRAMID:
        return 5;
    case TYPE_TETRAHEDER:
    case TYPE_QUAD:
        return 4;
    case TYPE_TRIANGLE:
        return 3;
    case TYPE_BAR:
        return 2;
    default:
        return 0;
    }
    return 0;
}
