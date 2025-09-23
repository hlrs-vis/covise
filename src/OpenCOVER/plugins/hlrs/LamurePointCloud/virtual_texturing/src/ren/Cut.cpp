// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/vt/ren/CutDatabase.h>

namespace vt
{
CutState::CutState(uint16_t depth) : _cut(), _mem_slots_updated(), _mem_slots_locked(), _mem_slots_cleared()
{
    uint16_t level = 0;

    while(level < depth)
    {
        uint32_t length_of_depth = (uint32_t)QuadTree::get_length_of_depth(level) * 4;

        _index_buffer_sizes.emplace_back(length_of_depth);

        uint8_t *index_buffer = new uint8_t[length_of_depth];
        std::fill(index_buffer, index_buffer + length_of_depth, 0);

        _index_buffers.emplace_back(index_buffer);

        level++;
    }
}
void CutState::accept(CutState &cut_state)
{
    _cut.clear();
    _cut.insert(cut_state._cut.begin(), cut_state._cut.end());

    _mem_slots_locked.clear();
    _mem_slots_locked.insert(cut_state._mem_slots_locked.begin(), cut_state._mem_slots_locked.end());

    _mem_slots_updated.clear();
    _mem_slots_updated.insert(cut_state._mem_slots_updated.begin(), cut_state._mem_slots_updated.end());

    _mem_slots_cleared.clear();
    _mem_slots_cleared.insert(cut_state._mem_slots_cleared.begin(), cut_state._mem_slots_cleared.end());

    for(size_t i = 0; i < _index_buffers.size(); ++i)
    {
        std::copy(cut_state._index_buffers[i], cut_state._index_buffers[i] + cut_state._index_buffer_sizes[i], _index_buffers[i]);
    }
}
CutState::~CutState()
{
    for(auto &index_buffer : _index_buffers)
    {
        delete index_buffer;
    }
}
cut_type &CutState::get_cut() { return _cut; }
uint8_t *CutState::get_index(uint16_t level) { return _index_buffers.at(level); }
mem_slots_index_type &CutState::get_mem_slots_cleared() { return _mem_slots_cleared; }
mem_slots_index_type &CutState::get_mem_slots_updated() { return _mem_slots_updated; }
mem_slots_index_type &CutState::get_mem_slots_locked() { return _mem_slots_locked; }
Cut::Cut(pre::AtlasFile *atlas, CutState *front, CutState *back) : DoubleBuffer<CutState>(front, back)
{
    _atlas = atlas;
    _drawn = false;
}
void Cut::deliver() { _front->accept((*_back)); }
Cut &Cut::init_cut(pre::AtlasFile *atlas)
{
    CutState *front_state = new CutState((uint16_t)atlas->getDepth());
    CutState *back_state = new CutState((uint16_t)atlas->getDepth());

    Cut *cut = new Cut(atlas, front_state, back_state);
    return *cut;
}
pre::AtlasFile *Cut::get_atlas() const { return _atlas; }
bool Cut::is_drawn() const { return _drawn; }
void Cut::set_drawn(bool drawn) { _drawn = drawn; }
uint32_t Cut::get_dataset_id(uint64_t cut_id) { return (uint32_t)(cut_id >> 32); }
uint16_t Cut::get_view_id(uint64_t cut_id) { return (uint16_t)(cut_id >> 16); }
uint16_t Cut::get_context_id(uint64_t cut_id) { return (uint16_t)cut_id; }
}