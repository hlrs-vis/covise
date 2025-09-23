// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef QUAD_TREE_H
#define QUAD_TREE_H

#include <lamure/vt/common.h>
#include <lamure/vt/ext/morton.h>

using namespace std;

namespace vt
{
class QuadTree
{
  public:

    static const id_type get_child_id(id_type node_id, id_type child_index);

    static const id_type get_parent_id(id_type node_id);

    static const id_type get_first_node_id_of_depth(uint32_t depth);

    static const size_t get_length_of_depth(uint32_t depth);

    static const uint16_t get_depth_of_node(id_type node_id);

    static const uint16_t calculate_depth(size_t dim, size_t tile_size);

    static const size_t get_tiles_per_row(uint32_t _depth);

    static void get_pos_by_id(id_type node_id, uint_fast32_t  &x, uint_fast32_t  &y);
};
}

#endif // QUAD_TREE_H