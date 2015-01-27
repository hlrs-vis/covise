/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Triangle.h"
#include "Vertex.h"
#include "Point.h"

extern float normaldeviation_cos;
extern float max_cos_2;
extern float boundary_factor;
extern bool ignoreData;

Triangle::Triangle(const Vertex *v0, const Vertex *v1, const Vertex *v2)
    : _invisible(false)
    , _v0(const_cast<Vertex *>(v0))
    , _v1(const_cast<Vertex *>(v1))
    , _v2(const_cast<Vertex *>(v2))
{
    const Point *p0 = v0->point();
    const Point *p1 = v1->point();
    const Point *p2 = v2->point();

    _q = p0->NewQ(p1, p2);
}

void
Triangle::invisible()
{
    _invisible = true;
}

bool
Triangle::visible() const
{
    return (!_invisible);
}

Triangle::~Triangle()
{
    delete _q;
}

Triangle::Triangle(const Triangle &rhs)
{
    _v0 = rhs._v0;
    _v1 = rhs._v1;
    _v2 = rhs._v2;
    _q = rhs._q->Copy();
    _invisible = rhs._invisible;
}

void
Triangle::UpdateVertex(Vertex *old, Vertex *neu)
{
    if (old == _v0)
    {
        _v0 = neu;
    }
    else if (old == _v1)
    {
        _v1 = neu;
    }
    else if (old == _v2)
    {
        _v2 = neu;
    }
}

bool
TooUgly(const float *e0, const float *e1)
{
    float res = (float)ScalarProd(3, e0, e1);
    res *= res;
    float res_0 = (float)ScalarProd(3, e0, e0);
    float res_1 = (float)ScalarProd(3, e1, e1);
    if (res_0 * res_1 * max_cos_2 < res)
    {
        return true;
    }

    float e2[3];
    e2[0] = e1[0] - e0[0];
    e2[1] = e1[1] - e0[1];
    e2[2] = e1[2] - e0[2];
    res = (float)ScalarProd(3, e0, e2);
    res *= res;
    float res_2 = (float)ScalarProd(3, e2, e2);
    if (res_0 * res_2 * max_cos_2 < res)
    {
        return true;
    }

    return false;
}

bool
Triangle::CheckSide(const Vertex *v, const Point *point) const
{
    if (_invisible)
    {
        return true;
    }
    // first identify which vertex is being moved
    Vertex *side0 = NULL;
    Vertex *side1 = NULL;
    if (_v0 == v)
    {
        side0 = _v1;
        side1 = _v2;
    }
    else if (_v1 == v)
    {
        side0 = _v2;
        side1 = _v0;
    }
    else if (_v2 == v)
    {
        side0 = _v0;
        side1 = _v1;
    }
    if (side0 == NULL || side1 == NULL)
    {
        return false;
    }
    const Point *point0 = side0->point();
    const Point *point1 = side1->point();
    int datadim = point0->data_dim();
    const float *data0 = point0->data();
    const float *data1 = point1->data();
    const float *prev_data = v->point()->data();
    const float *moved_data = point->data();
    // FIXME
    static float e0[32], e1[32], normal[32];
    static float moved_e0[32], moved_e1[32], moved_normal[32];
    int coord;
    for (coord = 0; coord < 3 + datadim; ++coord)
    {
        if (!ignoreData || (coord < 3))
        {
            e0[coord] = data0[coord] - prev_data[coord];
            e1[coord] = data1[coord] - prev_data[coord];
            moved_e0[coord] = data0[coord] - moved_data[coord];
            moved_e1[coord] = data1[coord] - moved_data[coord];
        }
        else
        {
            e0[coord] = 0.0f;
            e1[coord] = 0.0f;
            moved_e0[coord] = 0.0f;
            moved_e1[coord] = 0.0f;
        }
    }
    // it is also advisable to make sure that the triangle
    // does not look too ugly
    if (TooUgly(moved_e0, moved_e1))
    {
        return false;
    }

    vect_prod(normal, e0, e1);
    vect_prod(moved_normal, moved_e0, moved_e1);
    if (!Normalise(normal))
    {
        return false;
    }
    if (!Normalise(moved_normal))
    {
        return false;
    }
    float res = (float)ScalarProd(3, normal, moved_normal);
    if (res <= normaldeviation_cos)
    {
        return false;
    }
    // redo for an arbitrary number of dimensions
    if (datadim > 0)
    {
        Normalise(e0, 3 + datadim);
        Normalise(moved_e0, 3 + datadim);
        float res = (float)ScalarProd(3 + datadim, e0, e1);
        float moved_res = (float)ScalarProd(3 + datadim, moved_e0, moved_e1);
        int i;
        for (i = 0; i < 3 + datadim; ++i)
        {
            e1[i] -= res * e0[i];
            moved_e1[i] -= moved_res * moved_e0[i];
        }
        if (!Normalise(e1, 3 + datadim) || !Normalise(moved_e1, 3 + datadim))
        {
            return false;
        }
        float a11 = (float)ScalarProd(3 + datadim, e0, moved_e0);
        float a12 = (float)ScalarProd(3 + datadim, e1, moved_e0);
        float a21 = (float)ScalarProd(3 + datadim, e0, moved_e1);
        float a22 = (float)ScalarProd(3 + datadim, e1, moved_e1);
        float A11 = a11 * a11 + a21 * a21;
        float A22 = a12 * a12 + a22 * a22;
        float A12 = a11 * a12 + a21 * a22;
        float minmu = 0.5f * (A11 + A22 - (float)sqrt((A11 - A22) * (A11 - A22) + 4.0f * A12 * A12));
        //float maxmu = 0.5*(A11+A22+sqrt((A11-A22)*(A11-A22) + 4.0*A12*A12));
        if (minmu <= normaldeviation_cos * normaldeviation_cos)
        {
            return false;
        }
    }
    return true;
}

Q *
Triangle::QBound(const Vertex *va, const Vertex *vb) const
{
    const float *data0 = NULL;
    const float *data1 = NULL;
    const float *interior = NULL;

    if ((va == _v0 && vb == _v1)
        || (va == _v1 && vb == _v0))
    {
        data0 = _v0->point()->data();
        data1 = _v1->point()->data();
        interior = _v2->point()->data();
    }
    else if ((va == _v1 && vb == _v2)
             || (va == _v2 && vb == _v1))
    {
        data0 = _v1->point()->data();
        data1 = _v2->point()->data();
        interior = _v0->point()->data();
    }
    else if ((va == _v2 && vb == _v0)
             || (va == _v0 && vb == _v2))
    {
        data0 = _v2->point()->data();
        data1 = _v0->point()->data();
        interior = _v1->point()->data();
    }
    if (data0 == NULL || data1 == NULL || interior == NULL)
    {
        return NULL;
    }
    int data_dim = va->point()->data_dim();
    // FIXME
    double copy_data0[32];
    double copy_data1[32];
    double copy_interior[32];
    /*
      double *copy_data0 = new double[data_dim+3]; // FIXME
      double *copy_data1 = new double[data_dim+3];
      double *copy_interior = new double[data_dim+3];
   */
    copy_data0[0] = data0[0];
    copy_data0[1] = data0[1];
    copy_data0[2] = data0[2];
    copy_data1[0] = data1[0];
    copy_data1[1] = data1[1];
    copy_data1[2] = data1[2];

    int coord;
    for (coord = 0; coord < data_dim; ++coord)
    {
        copy_data0[3 + coord] = 0.0;
        copy_data1[3 + coord] = 0.0;
        copy_interior[3 + coord] = 0.0;
    }

    double side0[3];
    double side1[3];
    side0[0] = data0[0] - interior[0];
    side0[1] = data0[1] - interior[1];
    side0[2] = data0[2] - interior[2];
    side1[0] = data1[0] - interior[0];
    side1[1] = data1[1] - interior[1];
    side1[2] = data1[2] - interior[2];
    vect_prod(copy_interior, side0, side1);
    double surf = sqrt(copy_interior[0] * copy_interior[0] + copy_interior[1] * copy_interior[1] + copy_interior[2] * copy_interior[2]);
    double sidelength = sqrt((copy_data0[0] - data1[0]) * (copy_data0[0] - data1[0]) + (copy_data0[1] - data1[1]) * (copy_data0[1] - data1[1]) + (copy_data0[2] - data1[2]) * (copy_data0[2] - data1[2]));
    if (surf == 0.0)
    {
        return NULL;
    }
    double lengthBound = sidelength / surf;

    copy_interior[0] *= lengthBound;
    copy_interior[1] *= lengthBound;
    copy_interior[2] *= lengthBound;
    copy_interior[0] += data0[0];
    copy_interior[1] += data0[1];
    copy_interior[2] += data0[2];

    Q *ret = Factory::NewQs(data_dim, copy_interior, copy_data0, copy_data1);
    (*ret) *= (boundary_factor * surf / (sidelength * sidelength));
    ret->ReduceData();
    /* FIXME
      delete [] copy_data0;
      delete [] copy_data1;
      delete [] copy_interior;
   */
    return ret;
}

Q *
Triangle::Sum(vector<const Triangle *>::iterator start,
              vector<const Triangle *>::iterator ende)
{
    if (start != ende)
    {
        vector<const Triangle *>::iterator it = start;
        Q *ret = (*it)->_q->Copy();
        ++it;
        while (it != ende)
        {
            *ret += *((*it)->_q);
            ++it;
        }
        return ret;
    }
    return NULL;
}

Q *
Triangle::Sum(set<const Triangle *>::iterator start,
              set<const Triangle *>::iterator ende)
{
    if (start != ende)
    {
        set<const Triangle *>::iterator it = start;
        Q *ret = (*it)->_q->Copy();
        ++it;
        while (it != ende)
        {
            *ret += *((*it)->_q);
            ++it;
        }
        return ret;
    }
    return NULL;
}

Vertex *
    Triangle::
    operator[](int i)
{
    if (i == 0)
    {
        return _v0;
    }
    else if (i == 1)
    {
        return _v1;
    }
    else if (i == 2)
    {
        return _v2;
    }
    return NULL;
}

const Vertex *
    Triangle::
    operator[](int i) const
{
    if (i == 0)
    {
        return _v0;
    }
    else if (i == 1)
    {
        return _v1;
    }
    else if (i == 2)
    {
        return _v2;
    }
    return NULL;
}

void Triangle::print(std::string pre)
{
    std::cout << pre << "Triangle" << std::endl;
    std::cout << pre << " Visible " << _invisible << std::endl;
    _v0->print(pre + " ");
    _v1->print(pre + " ");
    _v2->print(pre + " ");
    if (_q)
        _q->print(pre + " ");
    else
        std::cout << pre << " Q NULL" << std::endl;
}

ostream &
operator<<(ostream &out, const Triangle &tr)
{
    out << "Triangle: "
        << " vertices: "
        << tr._v0->label() << ",...."
        << tr._v1->label() << ",...."
        << tr._v2->label() << endl;
    return out;
}

bool
    Triangle::
    operator==(const Triangle &rhs) const
{
    return (_v0 == rhs._v0) && (_v1 == rhs._v1) && (_v2 == rhs._v2);
}

bool
    Triangle::
    operator<(const Triangle &rhs) const
{
    if (_v0 < rhs._v0)
        return true;
    if (_v0 > rhs._v0)
        return false;
    if (_v1 < rhs._v1)
        return true;
    if (_v1 > rhs._v1)
        return false;
    return _v2 < rhs._v2;
}
