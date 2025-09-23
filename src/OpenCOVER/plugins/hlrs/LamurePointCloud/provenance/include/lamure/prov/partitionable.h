// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PARTITIONABLE_H
#define LAMURE_PARTITIONABLE_H

#include <lamure/prov/cacheable.h>
#include <lamure/prov/partition.h>

namespace lamure {
namespace prov
{
template <class TPartition>
class Partitionable
{
  public:
    enum Sort
    {
        STD_SORT = 0,
        BOOST_SPREADSORT = 1,
        PDQ_SORT = 2
    };

    Partitionable()
    {
        this->_partitions = vec<TPartition>();
        this->_sort = STD_SORT;
    }

    virtual void partition() = 0;
    // vec<TPartition> const &get_partitions() { return _partitions; }
    vec<TPartition> &get_partitions() { return _partitions; }

  protected:
    vec<TPartition> _partitions;
    Sort _sort;
    uint8_t _max_depth = 10;
    uint8_t _min_per_node = 1;
};
}
}

#endif // LAMURE_PARTITIONABLE_H