// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_DENSE_POINT_H
#define LAMURE_DENSE_POINT_H

#include <lamure/prov/point.h>
#include <lamure/prov/common.h>

namespace lamure {
namespace prov
{
class DensePoint : public Point
{
  public:
    DensePoint() { _normal = vec3f(); }
    DensePoint(const vec3f &_center, const vec3f &_color, const vec<uint8_t> &_metadata, const vec3f &_normal) : Point(_center, _color, _metadata), _normal(_normal) {}
    ~DensePoint() {}
    const vec3f &get_normal() const { return _normal; }
    friend ifstream &operator>>(ifstream &is, DensePoint &dense_point)
    {
        dense_point.read_essentials(is);

        is.read(reinterpret_cast<char *>(&dense_point._normal[0]), 4);
        dense_point._normal[0] = swap(dense_point._normal[0], true);
        is.read(reinterpret_cast<char *>(&dense_point._normal[1]), 4);
        dense_point._normal[1] = swap(dense_point._normal[1], true);
        is.read(reinterpret_cast<char *>(&dense_point._normal[2]), 4);
        dense_point._normal[2] = swap(dense_point._normal[2], true);

        // if(DEBUG)
        //     printf("\nNormal: %f %f %f", dense_point._normal[0], dense_point._normal[1], dense_point._normal[2]);

        return is;
    }
    static const uint32_t ENTITY_LENGTH = 72;

  protected:
    vec3f _normal;
};
}
}

#endif // LAMURE_DENSE_POINT_H