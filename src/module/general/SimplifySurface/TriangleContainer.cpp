/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TriangleContainer.h"
#include "Vertex.h"

TriangleContainer *
TriangleContainer::NewTriangleContainer(TriangleContainer::TYPE type)
{
    switch (type)
    {
    case VECTOR:
        return (new TriangleVector());
    }
    return NULL;
}

TriangleContainer::~TriangleContainer()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++

TriangleVector::TriangleVector()
{
}

void
TriangleVector::reserve(size_t laenge)
{
    _triangleList.reserve(laenge);
}

const Triangle *
TriangleVector::insert(const Vertex *v0,
                       const Vertex *v1,
                       const Vertex *v2)
{
    _triangleList.push_back(Triangle(v0, v1, v2));
    return &(*(_triangleList.rbegin()));
}

TriangleVector::~TriangleVector()
{
}

void
TriangleVector::MaxLabel(int &mark_max) const
{
    mark_max = -1;
    unsigned int tri;
    for (tri = 0; tri < _triangleList.size(); ++tri)
    {
        if (_triangleList[tri].visible())
        {
            if (mark_max < _triangleList[tri][0]->label())
            {
                mark_max = _triangleList[tri][0]->label();
            }
            if (mark_max < _triangleList[tri][1]->label())
            {
                mark_max = _triangleList[tri][1]->label();
            }
            if (mark_max < _triangleList[tri][2]->label())
            {
                mark_max = _triangleList[tri][2]->label();
            }
        }
    }
}

void
TriangleVector::SetMarks(int *mark, int length) const
{
    unsigned int tri;
    for (tri = 0; tri < _triangleList.size(); ++tri)
    {
        if (_triangleList[tri].visible())
        {
            mark[_triangleList[tri][0]->label()] = 0;
            mark[_triangleList[tri][1]->label()] = 0;
            mark[_triangleList[tri][2]->label()] = 0;
        }
    }

    int vert, count = 0;
    for (vert = 0; vert < length; ++vert)
    {
        if (mark[vert] == 0)
        {
            mark[vert] = count;
            ++count;
        }
    }
}

void
TriangleVector::SetConnectivities(vector<int> &leftTriangles, const int *mark)
{
    unsigned int tri;
    for (tri = 0; tri < _triangleList.size(); ++tri)
    {
        if (_triangleList[tri].visible())
        {
            int conn = mark[_triangleList[tri][0]->label()];
            leftTriangles.push_back(conn);
            conn = mark[_triangleList[tri][1]->label()];
            leftTriangles.push_back(conn);
            conn = mark[_triangleList[tri][2]->label()];
            leftTriangles.push_back(conn);
        }
    }
}

void TriangleVector::print(std::string pre)
{
    std::cout << pre << "TriangleContainer" << std::endl;
    vector<Triangle>::iterator it;
    for (it = _triangleList.begin(); it != _triangleList.end(); ++it)
    {
        Triangle *t = const_cast<Triangle *>(&(*it));
        t->print(pre + " ");
    }
}
