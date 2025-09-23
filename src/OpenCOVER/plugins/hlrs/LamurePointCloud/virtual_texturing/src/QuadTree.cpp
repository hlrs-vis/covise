// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/QuadTree.h>

using namespace std;

namespace vt
{
const id_type QuadTree::get_child_id(const id_type node_id, const id_type child_index) { return node_id * 4 + 1 + child_index; }

const id_type QuadTree::get_parent_id(const id_type node_id)
{
    if(node_id == 0)
        return 0;

    if(node_id % 4 == 0)
    {
        return node_id / 4 - 1;
    }
    else
    {
        return (node_id + 4 - (node_id % 4)) / 4 - 1;
    }
}

const id_type QuadTree::get_first_node_id_of_depth(uint32_t depth) { return (id_type)0x5555555555555555 ^ ((id_type)0x5555555555555555 << (depth << 1)); }

const size_t QuadTree::get_length_of_depth(uint32_t depth) { return (const size_t)round(pow((double)4, (double)depth)); }

const uint16_t QuadTree::get_depth_of_node(const id_type node_id) { return (uint16_t)(log((node_id + 1) * (4 - 1)) / log(4)); }

const uint16_t QuadTree::calculate_depth(size_t dim, size_t tile_size)
{
    size_t dim_tiled = dim / tile_size;
    return (uint16_t)(log(dim_tiled * dim_tiled) / log(4));
}

const size_t QuadTree::get_tiles_per_row(uint32_t depth) { return (size_t)pow(2, depth); }

void QuadTree::get_pos_by_id(id_type node_id, uint_fast32_t &x, uint_fast32_t&y)
{
    uint16_t depth = QuadTree::get_depth_of_node(node_id);
    id_type first_id = QuadTree::get_first_node_id_of_depth(depth);

    const uint_fast64_t id_in_depth = node_id - first_id;

    morton2D_64_decode(id_in_depth, x, y);
}
}