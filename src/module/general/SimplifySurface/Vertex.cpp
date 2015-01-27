/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Vertex.h"
#include "Triangle.h"
#include "Edge.h"
#include "Point.h"

Vertex::Vertex(int label, float x, float y, float z,
               int no_data, const float *data,
               const float *normal)
    : _label(label)
    , _q_bound(NULL)
    , _q_full(NULL)
    , _normal(NULL)
{
    _point = Factory::NewPointAndData(no_data, x, y, z, data);
    if (normal)
    {
        _normal = new float[3];
        _normal[0] = normal[0];
        _normal[1] = normal[1];
        _normal[2] = normal[2];
    }
}

const float *
Vertex::normal() const
{
    return _normal;
}

Vertex::Vertex(const Vertex &rhs)
{
    _label = rhs._label;
    _point = rhs._point->Copy();
    _tr_set = rhs._tr_set;
    _edge_set = rhs._edge_set;
    if (rhs._q_bound)
    {
        _q_bound = rhs._q_bound->Copy();
    }
    else
    {
        _q_bound = NULL;
    }

    if (rhs._q_full)
    {
        _q_full = rhs._q_full->Copy();
    }
    else
    {
        _q_full = NULL;
    }

    if (rhs._normal)
    {
        _normal = new float[3];
        _normal[0] = rhs._normal[0];
        _normal[1] = rhs._normal[1];
        _normal[2] = rhs._normal[2];
    }
    else
    {
        _normal = NULL;
    }
}

void
Vertex::UpdatePoint(const Point *point, const Vertex *v1)
{
    const float *normal1 = v1->normal();
    if (_normal && normal1)
    {
        float disp[3];
        disp[0] = v1->_point->data()[0] - _point->data()[0];
        disp[1] = v1->_point->data()[1] - _point->data()[1];
        disp[2] = v1->_point->data()[2] - _point->data()[2];
        float dispnew[3];
        dispnew[0] = point->data()[0] - _point->data()[0];
        dispnew[1] = point->data()[1] - _point->data()[1];
        dispnew[2] = point->data()[2] - _point->data()[2];
        float len2 = (float)ScalarProd(3, disp, disp);
        if (Normalise(disp) && len2 > 0.0)
        {
            len2 = sqrt(len2);
            float tau = (float)ScalarProd(3, dispnew, disp) / len2;
            float tau_comp = 1.0f - tau;
            _normal[0] = tau_comp * _normal[0] + tau * normal1[0];
            _normal[1] = tau_comp * _normal[1] + tau * normal1[1];
            _normal[2] = tau_comp * _normal[2] + tau * normal1[2];
            Normalise(_normal);
        }
    }
    if (_point == point)
    {
        return;
    }
    delete _point;
    _point = point->Copy();
}

int
Vertex::label() const
{
    return _label;
}

Vertex::~Vertex()
{
    delete _point;
    delete _q_bound;
    delete _q_full;
    delete _normal;
}

const Point *
Vertex::point() const
{
    return _point;
}

const Q *
Vertex::QBound() const
{
    return _q_bound;
}

const Q *
Vertex::QFull() const
{
    return _q_full;
}

void
Vertex::MakeBoundary()
{
    set<const Edge *>::iterator it;
    for (it = _edge_set.begin(); it != _edge_set.end(); ++it)
    {
        const Q *q_edge = (*it)->QBound();
        if (q_edge)
        {
            if (_q_bound)
            {
                (*_q_bound) += (*q_edge);
            }
            else
            {
                _q_bound = q_edge->Copy();
            }
        }
    }
}

void
Vertex::MakeQFull()
{
    delete _q_full;
    _q_full = Triangle::Sum(_tr_set.begin(), _tr_set.end());
    if (_q_bound)
    {
        if (_q_full)
        {
            *_q_full += *_q_bound;
        }
        else
        {
            _q_full = _q_bound->Copy();
        }
    }
}

void
Vertex::AccumulateBoundary(const Q *q)
{
    if (q)
    {
        if (_q_bound)
        {
            (*_q_bound) += (*q);
        }
        else
        {
            _q_bound = q->Copy();
        }
    }
}

void
Vertex::AccumulateQFull(const Q *q)
{
    if (q)
    {
        if (_q_full)
        {
            (*_q_full) += (*q);
        }
        else
        {
            _q_full = q->Copy();
        }
    }
}

void
Vertex::add_tr(Triangle *tr)
{
    _tr_set.insert(tr);
}

void
Vertex::add_edge(const Edge *edge)
{
    _edge_set.insert(edge);
}

bool
    Vertex::
    operator==(const Vertex &rhs) const
{
    return (_label == rhs._label);
}

bool
    Vertex::
    operator<(const Vertex &rhs) const
{
    return (_label < rhs._label);
}

pair<set<const Edge *>::const_iterator, set<const Edge *>::const_iterator>
Vertex::edge_interval() const
{
    pair<set<const Edge *>::const_iterator, set<const Edge *>::const_iterator> ret;
    ret.first = _edge_set.begin();
    ret.second = _edge_set.end();
    return ret;
}

pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator>
Vertex::triangle_interval() const
{
    pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator> ret;
    ret.first = _tr_set.begin();
    ret.second = _tr_set.end();
    return ret;
}

void
Vertex::erase_edge(const Edge *edge)
{
    _edge_set.erase(edge);
}

void
Vertex::erase()
{
    _label = -1;
}

bool
Vertex::IsInEdgeSet(const Edge *edge) const
{
    set<const Edge *>::const_iterator it;
    for (it = _edge_set.begin(); it != _edge_set.end(); ++it)
    {
        if (*edge == *(*it))
        {
            return true;
        }
    }
    return false;
}

void
Vertex::BoundaryEdges(set<const Edge *> &boundaryEdges) const
{
    boundaryEdges.clear();
    set<const Edge *>::const_iterator it;
    for (it = _edge_set.begin(); it != _edge_set.end(); ++it)
    {
        if ((*it)->boundary())
        {
            boundaryEdges.insert(*it);
        }
    }
}

void Vertex::print(std::string pre)
{
    std::cout << pre << "Vertex" << std::endl;
    std::cout << pre << " Label " << _label << std::endl;
    _point->print(pre + " ");
    std::cout << pre << " Triangle Set SIZE " << _tr_set.size() << std::endl;
    std::cout << pre << " Edge Set " << _edge_set.size() << std::endl;
    set<const Edge *>::iterator it;
    for (it = _edge_set.begin(); it != _edge_set.end(); ++it)
    {
        std::cout << pre << "  Vertex " << (*it)->v0()->label() << " " << (*it)->v1()->label() << std::endl;
    }
    if (_q_bound)
        _q_bound->print(pre + " ");
    else
        std::cout << pre << " Q NULL" << std::endl;
    if (_q_full)
        _q_full->print(pre + " ");
    else
        std::cout << pre << " Q NULL" << std::endl;
    if (_normal)
        std::cout << pre << " Normal " << _normal[0] << " " << _normal[1] << " " << _normal[2] << std::endl;
    else
        std::cout << pre << " Normal NULL" << std::endl;
}

ostream &
operator<<(ostream &out, const Vertex &v)
{
    out << "Vertex: " << v._label << " vvvvvvvvvvvvvvvvvvvvvvvv " << endl;
    set<const Triangle *>::const_iterator tr_it;
    for (tr_it = v._tr_set.begin(); tr_it != v._tr_set.end(); ++tr_it)
    {
        out << "Triangle " << (*tr_it)->operator[](0)->label() << ' ' << (*tr_it)->operator[](1)->label() << ' ' << (*tr_it)->operator[](2)->label() << endl;
    }
    set<const Edge *>::const_iterator edge_it;
    for (edge_it = v._edge_set.begin(); edge_it != v._edge_set.end(); ++edge_it)
    {
        out << "Edge " << (*edge_it)->v0()->label() << ' ' << (*edge_it)->v1()->label() << endl;
    }
    out << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << endl;
    return out;
}

bool
Vertex::ValenceTooHigh(int num_max) const
{
    if (_edge_set.size() >= (unsigned int)num_max)
    {
        return true;
    }
    return false;
}
