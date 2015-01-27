/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Edge.h"
#include "Vertex.h"
#include "Triangle.h"
#include "Point.h"
#ifdef WIN32
#include <iterator>
#endif
#include "util/coviseCompat.h"

// #include <algorithm>

using namespace std;

extern float domaindeviation_cos;
extern bool ignoreData;

Edge::Edge(const Vertex *v0, const Vertex *v1, const Triangle *tr)
    : _tr(tr)
    , _v0(v0)
    , _v1(v1)
    , _popt(NULL)
    , _q(NULL)
    , _q_bound(NULL)
    , _cost(0.0)
    , _boundary(true)
{
    if (v0 > v1)
    {
        _v0 = v1;
        _v1 = v0;
    }
    const Point *point0 = _v0->point();
    const Point *point1 = _v1->point();
    const float *data0 = point0->data();
    const float *data1 = point1->data();
    _direction[0] = data1[0] - data0[0];
    _direction[1] = data1[1] - data0[1];
    _direction[2] = data1[2] - data0[2];
    if (!Normalise(_direction))
    {
        _direction[0] = 0.0;
        _direction[1] = 0.0;
        _direction[2] = 0.0;
    }
}

bool
Edge::boundary() const
{
    return _boundary;
}

Edge::Edge(const Edge &rhs)
{
    _v0 = rhs._v0;
    _v1 = rhs._v1;
    _tr = rhs._tr;
    if (rhs._q != NULL)
    {
        _q = rhs._q->Copy();
    }
    else
    {
        _q = NULL;
    }
    if (rhs._q_bound != NULL)
    {
        _q_bound = rhs._q_bound->Copy();
    }
    else
    {
        _q_bound = NULL;
    }
    if (rhs._popt != NULL)
    {
        _popt = rhs._popt->Copy();
    }
    else
    {
        _popt = NULL;
    }
    _cost = rhs._cost;
    _boundary = rhs._boundary;
    std::copy(rhs._direction, rhs._direction + 3, _direction);
}

Edge::~Edge()
{
    delete _popt;
    delete _q;
    delete _q_bound;
}

Edge::operator size_t() const
{
    return size_t(v0() /*+ v1()->label()*_dimension*/);
}

float
Edge::cost() const
{
    return _cost;
}

void
Edge::boundary(bool bound)
{
    _boundary = bound;
}

const Vertex *
Edge::v0() const
{
    return _v0;
}

const Vertex *
Edge::v1() const
{
    return _v1;
}

const Point *
Edge::popt() const
{
    return _popt;
}

const Q *
Edge::QBound() const
{
    return _q_bound;
}

void
Edge::new_cost(QCalc calc)
{
    switch (calc)
    {
    case ADD_TRIANGLES:
    {
        pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator>
            tr0_int = _v0->triangle_interval();
        pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator>
            tr1_int = _v1->triangle_interval();
        vector<const Triangle *> all_triangles;
        std::insert_iterator<vector<const Triangle *> >
            all_triangles_it(all_triangles, all_triangles.begin());
        set_union(tr0_int.first, tr0_int.second,
                  tr1_int.first, tr1_int.second, all_triangles_it);
        delete _q;
        delete _popt;
        _q = NULL;
        _popt = NULL;
        _q = Triangle::Sum(all_triangles.begin(), all_triangles.end());
        // get all boundary edges with _v0 or _v1
        // and accumulate boundary contribution
        set<const Edge *> boundaryEdges_0, boundaryEdges_1;
        _v0->BoundaryEdges(boundaryEdges_0);
        _v1->BoundaryEdges(boundaryEdges_1);
        vector<const Edge *> boundaryEdges;
        insert_iterator<vector<const Edge *> >
            boundaryEdgesInsert(boundaryEdges, boundaryEdges.begin());
        set_union(boundaryEdges_0.begin(), boundaryEdges_0.end(),
                  boundaryEdges_1.begin(), boundaryEdges_1.end(),
                  boundaryEdgesInsert);
        vector<const Edge *>::iterator bound_edge_it;
        for (bound_edge_it = boundaryEdges.begin();
             bound_edge_it != boundaryEdges.end(); ++bound_edge_it)
        {
            const Triangle *tr = (*bound_edge_it)->_tr;
            const Vertex *va = (*bound_edge_it)->_v0;
            const Vertex *vb = (*bound_edge_it)->_v1;
            Q *q = tr->QBound(va, vb);
            if (q)
                *_q += *q;
            if ((*bound_edge_it) == this)
            {
                _q_bound = q;
            }
            else
            {
                delete q;
            }
        }
    }
    // no break here!!
    case DO_NOT_ADD_TRIANGLES:
        assert(_q != NULL);
        delete _popt;
        _popt = _q->Solve(_cost);
        if (_popt == NULL) // the matrix is degenerate
        {
            float cost0, cost1, cost_int;
            Point *popt_int = NULL;

            cost0 = _q->Eval(_v0->point());
            cost1 = _q->Eval(_v1->point());

            if (_v0->QBound() == NULL && _v1->QBound() != NULL)
            {
                // in this case, we move _v0 to the boundary
                _popt = _v1->point()->Copy();
                _cost = cost1;
            }
            else if (_v0->QBound() != NULL && _v1->QBound() == NULL)
            {
                // in this case, we move _v1 to the boundary
                _popt = _v0->point()->Copy();
                _cost = cost0;
            }
            else
            {
                popt_int = _v0->point()->Copy();
                (*popt_int) += *(_v1->point());
                (*popt_int) *= 0.5;
                cost_int = _q->Eval(popt_int);

                if (cost0 <= cost1 && cost0 <= cost_int)
                {
                    delete popt_int;
                    _popt = _v0->point()->Copy();
                    _cost = cost0;
                }
                else if (cost1 <= cost0 && cost1 <= cost_int)
                {
                    delete popt_int;
                    _popt = _v1->point()->Copy();
                    _cost = cost1;
                }
                else
                {
                    _popt = popt_int;
                    _cost = cost_int;
                }
            }
        }
        if (ignoreData)
        {
            // in case we ignore the data, cost is calculated using the edges length in order to avoid large triangles
            _cost = fabs(_v0->point()->data()[0] - _v1->point()->data()[0]) + fabs(_v0->point()->data()[1] - _v1->point()->data()[1]) + fabs(_v0->point()->data()[2] - _v1->point()->data()[2]);
        }
    }
    if (0 && calc == ADD_TRIANGLES)
    {
        cerr << "-----------------------------" << endl;
        operator<<(cerr, *this);
        _q->Print();
        cerr << "-----------------------------" << endl;
    }
}

void
Edge::AddQ(const Q &q)
{
    if (_q == NULL)
    {
        _q = q.Copy();
    }
    else
    {
        (*_q) += q;
    }
}

bool
    Edge::
    operator==(const Edge &rhs) const
{
    return ((_v0 == rhs._v0 && _v1 == rhs._v1) || (_v0 == rhs._v1 && _v1 == rhs._v0));
}

bool
    Edge::
    operator<(const Edge &rhs) const
{
    if (_v0 < rhs._v0)
    {
        return true;
    }
    else if (_v0 == rhs._v0 && _v1 < rhs._v1)
    {
        return true;
    }
    return false;
}

void
Edge::UpdateVertex(const Vertex *vold, const Vertex *vnew)
{
    if (vold == _v0)
    {
        _v0 = vnew;
    }
    else if (vold == _v1)
    {
        _v1 = vnew;
    }
    if (_v0 > _v1)
    {
        const Vertex *tmp = _v0;
        _v0 = _v1;
        _v1 = tmp;
    }
}

ostream &
operator<<(ostream &out, const Edge &edge)
{
    out << "Edge "
        << " vertices: " << edge._v0->label() << ",......" << edge._v1->label() << ' ';
    const float *data = edge._popt->data();
    out << data[0] << ' ' << data[1] << ' ' << data[2] << ' ';
    out << edge._cost << endl;
    return out;
}

bool
Edge::CheckDirection(const Vertex *vertex, const Point *point) const
{
    const Point *point0 = _v0->point();
    const Point *point1 = _v1->point();
    assert(_v0 == vertex || _v1 == vertex);
    if (_v0 == vertex)
    {
        point0 = point;
    }
    else if (_v1 == vertex)
    {
        point1 = point;
    }
    const float *data0 = point0->data();
    const float *data1 = point1->data();
    float direction[3];
    direction[0] = data1[0] - data0[0];
    direction[1] = data1[1] - data0[1];
    direction[2] = data1[2] - data0[2];
    if (!Normalise(direction))
    {
        return false;
    }
    else if (_direction[0] == 0.0 && _direction[1] == 0.0 && _direction[2] == 0.0)
    {
        return false;
    }
    double scal = ScalarProd(3, _direction, direction);
    if (scal >= domaindeviation_cos)
    {
        return true;
    }
    return false;
}

void Edge::print(std::string pre)
{
    std::cout << pre << "Edge" << std::endl;
    ((Vertex *)(_v0))->print(pre + " ");
    ((Vertex *)(_v1))->print(pre + " ");
    _popt->print(pre + " ");
    if (_q)
        _q->print(pre + " ");
    else
        std::cout << pre << " Q NULL" << std::endl;
    if (_q_bound)
        _q_bound->print(pre + " ");
    else
        std::cout << pre << " Q NULL" << std::endl;
    std::cout << pre << " Cost " << _cost << std::endl;
    std::cout << pre << " Boundary " << _boundary << std::endl;
    std::cout << pre << " Direction " << _direction[0] << " " << _direction[1] << " " << _direction[2] << std::endl;
}
