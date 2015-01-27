/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VERTEX_CONTAINER_H_
#define _VERTEX_CONTAINER_H_

#include "util/coviseCompat.h"

class Vertex;

class VertexContainer
{
public:
    enum TYPE
    {
        VECTOR
    };
    static VertexContainer *NewVertexContainer(TYPE type);
    VertexContainer();
    virtual ~VertexContainer();
    virtual void insert(int label, float x, float y, float z, int no_data_per_vertex,
                        const float *data, const float *normals) = 0;
    virtual Vertex *operator[](int label) = 0;
    virtual const Vertex *operator[](int label) const = 0;
    virtual void MakeBoundary() = 0;
    virtual void MakeQFull() = 0;
    virtual void erase(Vertex *) = 0;

    virtual void SetCoordinates(vector<float> &leftVertexX,
                                vector<float> &leftVertexY,
                                vector<float> &leftVertexZ,
                                vector<float> &leftData,
                                vector<float> &leftNormals,
                                const int *mark, int length) const = 0;
    virtual void reserve(size_t) = 0;
    virtual void print(std::string pre) = 0;

protected:
private:
};

class VertexVector : public VertexContainer
{
public:
    VertexVector();
    virtual ~VertexVector();
    virtual void insert(int label, float x, float y, float z, int no_data_per_vertex,
                        const float *data, const float *normals);
    virtual Vertex *operator[](int label);
    virtual const Vertex *operator[](int label) const;
    virtual void MakeBoundary();
    virtual void MakeQFull();
    virtual void erase(Vertex *);
    virtual void SetCoordinates(vector<float> &leftVertexX,
                                vector<float> &leftVertexY,
                                vector<float> &leftVertexZ,
                                vector<float> &leftData,
                                vector<float> &leftNormals,
                                const int *mark, int length) const;
    virtual void reserve(size_t);
    virtual void print(std::string pre);

protected:
private:
    vector<Vertex> _vertexList;
};
#endif
