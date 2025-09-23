// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_POINT_H
#define LAMURE_POINT_H

#include <lamure/prov/meta_data.h>
#include <lamure/prov/common.h>

namespace lamure {
namespace prov
{
class Point
{
  public:
    Point(const vec3f &_position, const vec3f &_color, const vec<uint8_t> &_metadata) : _position(_position), _color(_color) {}
    Point()
    {
        _position = vec3f();
        _color = vec3f();
    }
    ~Point() {}
    const vec3f &get_position() const { return _position; }
    const vec3f &get_color() const { return _color; };
    ifstream &read_essentials(ifstream &is)
    {
        is.read(reinterpret_cast<char *>(&_position[0]), 4);
        _position[0] = swap(_position[0], true);
        is.read(reinterpret_cast<char *>(&_position[1]), 4);
        _position[1] = swap(_position[1], true);
        is.read(reinterpret_cast<char *>(&_position[2]), 4);
        _position[2] = swap(_position[2], true);

        // if(DEBUG)
        //     printf("\nPosition: %f %f %f", _position[0], _position[1], _position[2]);

        is.read(reinterpret_cast<char *>(&_color[0]), 4);
        _color[0] = swap(_color[0], true);
        is.read(reinterpret_cast<char *>(&_color[1]), 4);
        _color[1] = swap(_color[1], true);
        is.read(reinterpret_cast<char *>(&_color[2]), 4);
        _color[2] = swap(_color[2], true);

        // if(DEBUG)
        //     printf("\nColor: %f %f %f", _color[0], _color[1], _color[2]);

        return is;
    }

  protected:
    vec3f _position;
    vec3f _color;
};
}
}

#endif // LAMURE_POINT_H