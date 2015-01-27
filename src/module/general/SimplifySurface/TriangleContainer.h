/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRIANGLE_CONTAINER_H_
#define _TRIANGLE_CONTAINER_H_

#include "util/coviseCompat.h"
#include "Triangle.h"

class Vertex;

class TriangleContainer
{
public:
    enum TYPE
    {
        VECTOR
    };
    static TriangleContainer *NewTriangleContainer(TYPE);
    virtual const Triangle *insert(const Vertex *v0,
                                   const Vertex *v1,
                                   const Vertex *v2) = 0;
    virtual ~TriangleContainer();
    virtual void MaxLabel(int &mark_max) const = 0;
    virtual void SetMarks(int *mark, int length) const = 0;
    virtual void SetConnectivities(vector<int> &leftTriangles, const int *mark) = 0;
    virtual void reserve(size_t) = 0;
    virtual void print(std::string pre) = 0;

protected:
private:
};

class TriangleVector : public TriangleContainer
{
public:
    TriangleVector();
    const Triangle *insert(const Vertex *v0,
                           const Vertex *v1,
                           const Vertex *v2);
    virtual ~TriangleVector();
    virtual void MaxLabel(int &mark_max) const;
    virtual void SetMarks(int *mark, int length) const;
    virtual void SetConnectivities(vector<int> &leftTriangles, const int *mark);
    virtual void reserve(size_t);
    virtual void print(std::string pre);

protected:
private:
    vector<Triangle> _triangleList;
};
#endif
