/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

int processLine(UniField *unif,
                int line,
                float *verts,
                UniGeom *ugeom)
{
    if ((unif->getVectorComp(0, line, 0) < 0) || (unif->getVectorComp(1, line, 0) < 0))
    {
        // skip empty line (negative first component (time) indicates line end)
        return 0;
    }

    // sample vertices
    int nvertices = 0;
    for (int vertex = 0; vertex < unif->getDim(0); vertex++)
    {

        if (unif->getVectorComp(vertex, line, 0) < 0)
        {
            // abort
            break;
        }

        vec3 pos;
        unif->getCoord(vertex, line, pos);

        verts[vertex * 3 + 0] = pos[0];
        verts[vertex * 3 + 1] = pos[1];
        verts[vertex * 3 + 2] = pos[2];
        nvertices++;
    }

    // add polyline to geom
    ugeom->addPolyline(verts, NULL, nvertices);
    return nvertices;
}

bool field_to_lines_impl(UniSys *us,
                         UniField *unif,
                         int nodes_x, int nodes_y, int nodes_z,
                         int stride,
                         UniGeom *ugeom,
                         std::vector<int> *usedNodes,
                         std::vector<int> *usedNodesVertCnt)
{ // usedNodes: may be NULL
    // usedNodesVertCnt: may be NULL

    if ((stride > 1) && (nodes_x * nodes_y * nodes_z > unif->getDim(1)))
    {
        us->error("nodes_x(=%d) * nodes_y(=%d) * nodes_z(=%d) > number of lines in field (=%d)",
                  nodes_x, nodes_y, nodes_z, unif->getDim(1));
        return false;
    }

    // TODO: create empty geom instead?
    if (unif->getDim(0) < 2)
    {
        // no lines
        return false; // false ok?
    }

    if (stride > nodes_x)
        stride = nodes_x;
    if (stride > nodes_y)
        stride = nodes_y;
    //if (stride > nodes_z) stride = nodes_z;

    float *verts = new float[unif->getDim(0) * 3];
    if (!verts)
    {
        us->error("out of memory");
        return false;
    }

    ugeom->createObj(UniGeom::GT_LINE);

    if (stride <= 1)
    {
        for (int line = 0; line < unif->getDim(1); line++)
        {

            int vertCnt = processLine(unif, line, verts, ugeom);

            if (vertCnt < 1)
                continue;

            if (usedNodes)
                usedNodes->push_back(line);
            if (usedNodesVertCnt)
                usedNodesVertCnt->push_back(vertCnt);
        }
    }
    else
    {
        for (int z = 0; z < nodes_z; z += stride)
        {
            for (int y = 0; y < nodes_y; y += stride)
            {
                for (int x = 0; x < nodes_x; x += stride)
                {

                    int line = x + y * nodes_x + z * nodes_x * nodes_y;

                    int vertCnt = processLine(unif, line, verts, ugeom);

                    if (vertCnt < 1)
                        continue;

                    if (usedNodes)
                        usedNodes->push_back(line);
                    if (usedNodesVertCnt)
                        usedNodesVertCnt->push_back(vertCnt);
                }
            }
        }
    }

    delete[] verts;

    ugeom->assignObj("lines");

    return true;
}
