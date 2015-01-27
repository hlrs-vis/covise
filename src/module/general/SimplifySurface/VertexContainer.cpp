/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VertexContainer.h"
#include "Vertex.h"
#include "Point.h"

VertexContainer *
VertexContainer::NewVertexContainer(TYPE type)
{
    switch (type)
    {
    case VECTOR:
        return (new VertexVector());
    }
    return NULL;
}

VertexContainer::~VertexContainer()
{
}

VertexContainer::VertexContainer()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++

VertexVector::VertexVector()
    : VertexContainer()
{
}

VertexVector::~VertexVector()
{
}

void
VertexVector::reserve(size_t laenge)
{
    _vertexList.reserve(laenge);
}

void
VertexVector::insert(int label, float x, float y, float z,
                     int no_data_per_vertex, const float *data,
                     const float *normals)
{
    _vertexList.push_back(Vertex(label, x, y, z, no_data_per_vertex, data, normals));
}

Vertex *
    VertexVector::
    operator[](int label)
{
    return &(_vertexList[label]);
}

const Vertex *
    VertexVector::
    operator[](int label) const
{
    return &(_vertexList[label]);
}

void
VertexVector::MakeBoundary()
{
    vector<Vertex>::iterator vert_it;
    for (vert_it = _vertexList.begin(); vert_it != _vertexList.end(); ++vert_it)
    {
        vert_it->MakeBoundary();
    }
}

void
VertexVector::MakeQFull()
{
    vector<Vertex>::iterator vert_it;
    for (vert_it = _vertexList.begin(); vert_it != _vertexList.end(); ++vert_it)
    {
        vert_it->MakeQFull();
    }
}

void
VertexVector::erase(Vertex *v)
{
    v->erase();
}

void
VertexVector::SetCoordinates(vector<float> &leftVertexX,
                             vector<float> &leftVertexY,
                             vector<float> &leftVertexZ,
                             vector<float> &leftData,
                             vector<float> &leftNormals,
                             const int *mark, int length) const
{
    int i;
    for (i = 0; i < length; ++i)
    {
        if (mark[i] >= 0)
        {
            leftVertexX.push_back(_vertexList[i].point()->data()[0]);
            leftVertexY.push_back(_vertexList[i].point()->data()[1]);
            leftVertexZ.push_back(_vertexList[i].point()->data()[2]);
            int no_datapervertex = _vertexList[i].point()->data_dim();
            int dim;
            for (dim = 0; dim < no_datapervertex; ++dim)
            {
                leftData.push_back(_vertexList[i].point()->data()[3 + dim]);
            }
            if (_vertexList[i].normal())
            {
                leftNormals.push_back(_vertexList[i].normal()[0]);
                leftNormals.push_back(_vertexList[i].normal()[1]);
                leftNormals.push_back(_vertexList[i].normal()[2]);
            }
        }
    }
}

void VertexVector::print(std::string pre)
{
    std::cout << pre << "VertexContainer" << std::endl;
    vector<Vertex>::iterator it;
    for (it = _vertexList.begin(); it != _vertexList.end(); ++it)
    {
        Vertex *t = const_cast<Vertex *>(&(*it));
        t->print(pre + " ");
    }
}
