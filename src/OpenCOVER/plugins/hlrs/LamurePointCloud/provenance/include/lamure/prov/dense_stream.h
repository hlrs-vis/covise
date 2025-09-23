// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_DENSESTREAM_H
#define LAMURE_DENSESTREAM_H

#include <lamure/prov/dense_point.h>
#include <lamure/prov/streamable.h>

namespace lamure {
namespace prov
{
class DenseStream : public Streamable<DensePoint>
{
  public:
    DenseStream(ifstream &is) : Streamable(is) {}
    const DensePoint &access_at_implicit(uint32_t index) override
    {
        uint32_t sought_pos = Readable::HEADER_LENGTH + DensePoint::ENTITY_LENGTH * index;

        if(is.tellg() < sought_pos)
        {
            is.ignore(sought_pos - is.tellg());
        }
        else
        {
            is.clear();
            is.seekg(sought_pos);
        }

        DensePoint densePoint;
        is >> densePoint;

        return densePoint;
    }

    const vec<DensePoint> access_at_implicit_range(uint32_t index_start, uint32_t index_end) override
    {
        uint32_t sought_pos = Readable::HEADER_LENGTH + DensePoint::ENTITY_LENGTH * index_start;

        if(is.tellg() < sought_pos)
        {
            is.ignore(sought_pos - is.tellg());
        }
        else
        {
            is.clear();
            is.seekg(sought_pos);
        }

        vec<DensePoint> range;

        for(uint32_t i = 0; i < index_end - index_start; i++)
        {
            DensePoint densePoint;
            is >> densePoint;
            range.push_back(densePoint);
        }

        return range;
    }
};
}
}

#endif // LAMURE_DENSESTREAM_H
