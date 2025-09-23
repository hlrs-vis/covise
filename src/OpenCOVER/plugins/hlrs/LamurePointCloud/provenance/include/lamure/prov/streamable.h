// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_STREAMABLE_H
#define LAMURE_STREAMABLE_H

#include <lamure/prov/readable.h>
#include <lamure/prov/common.h>
#include <lamure/prov/point.h>

namespace lamure {
namespace prov
{
template <class Entity>
class Streamable : public Readable
{
  public:
    Streamable(ifstream &is) : is(is) { read_header(is); }
    virtual ~Streamable() {}
    virtual const Entity &access_at_implicit(uint32_t index) = 0;
    virtual const vec<Entity> access_at_implicit_range(uint32_t index_start, uint32_t index_end) = 0;

  protected:
    ifstream &is;
};
}
}
#endif // LAMURE_STREAMABLE_H
