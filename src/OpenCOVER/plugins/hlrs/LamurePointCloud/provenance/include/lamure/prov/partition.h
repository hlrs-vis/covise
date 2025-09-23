// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PARTITION_H
#define LAMURE_PARTITION_H

#include <lamure/prov/meta_data.h>
#include <lamure/prov/point.h>

namespace lamure {
namespace prov
{
template <class TPair, class TMetaData>
class Partition
{
  public:
    Partition()
    {
        this->_pair_ptrs = vec<s_ptr<TPair>>();
        this->_aggregate_metadata = TMetaData();
    }
    ~Partition() {}

    void set_pair_ptrs(vec<s_ptr<TPair>> &pair_ptrs) { this->_pair_ptrs = pair_ptrs; }
    TMetaData &get_aggregate_metadata() { return this->_aggregate_metadata; };

  protected:
    virtual void aggregate_metadata() = 0;

    vec<s_ptr<TPair>> _pair_ptrs;
    TMetaData _aggregate_metadata;
};
}
}
#endif // LAMURE_PARTITION_H
