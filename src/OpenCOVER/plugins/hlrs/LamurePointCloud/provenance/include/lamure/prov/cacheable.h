// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_CACHEABLE_DATA_H
#define LAMURE_CACHEABLE_DATA_H

#include <lamure/prov/readable.h>
#include <lamure/prov/common.h>
#include <lamure/prov/point.h>
#include <lamure/prov/cacheable.h>

namespace lamure {
namespace prov
{
template <class TPoint, class TMetaData>
class Cacheable : public Readable
{
  public:
    Cacheable(ifstream &is_prov, ifstream &is_meta)
    {
        this->is_prov = &is_prov;
        this->is_meta = &is_meta;
        static_assert(std::is_base_of<Point, TPoint>::value, "The used point type is not a derivative of Point");
        static_assert(std::is_base_of<MetaData, TMetaData>::value, "The used meta data type is not a derivative of MetaData");
        this->_points = vec<TPoint>();
        this->_points_metadata = vec<TMetaData>();
    }
    ~Cacheable(){};

    void cache_points()
    {
        uint32_t points_length;
        (*is_prov).read(reinterpret_cast<char *>(&points_length), 4);
        points_length = swap(points_length, true);

        // if(DEBUG)
        //            printf("\nPoints length: %i", points_length);

        uint32_t meta_data_length;
        (*is_prov).read(reinterpret_cast<char *>(&meta_data_length), 4);
        meta_data_length = swap(meta_data_length, true);

        // if(DEBUG)
        //             printf("\nPoints meta data length: %i ", meta_data_length);

        for(uint32_t i = 0; i < points_length; i++)
        {
            TPoint point = TPoint();
            (*is_prov) >> point;
            _points.push_back(point);

            TMetaData meta_container = TMetaData();
            meta_container.read_metadata((*is_meta), meta_data_length);
            _points_metadata.push_back(meta_container);
        }
    }

    const vec<TPoint> &get_points() const { return _points; }
    const vec<TMetaData> &get_points_metadata() const { return _points_metadata; }
    virtual void cache()
    {
        read_header((*is_prov));
        read_header((*is_meta));

        cache_points();
    }

  protected:
    ifstream *is_prov, *is_meta;
    vec<TPoint> _points;
    vec<TMetaData> _points_metadata;
};
}
}

#endif // LAMURE_CACHEABLE_DATA_H