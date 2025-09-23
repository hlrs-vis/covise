// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_CUT_H
#define LAMURE_CUT_H

#include <lamure/vt/common.h>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/ren/DoubleBuffer.h>
namespace vt
{
class CutDatabase;
class VIRTUAL_TEXTURING_DLL CutState
{
  public:
    CutState(uint16_t depth);
    ~CutState();

    uint8_t *get_index(uint16_t level);
    cut_type &get_cut();
    mem_slots_index_type &get_mem_slots_updated();
    mem_slots_index_type &get_mem_slots_cleared();
    mem_slots_index_type &get_mem_slots_locked();
    void accept(CutState &cut_state);

  private:
    std::vector<uint32_t> _index_buffer_sizes;
    std::vector<uint8_t *> _index_buffers;
    cut_type _cut;
    mem_slots_index_type _mem_slots_updated;
    mem_slots_index_type _mem_slots_cleared;
    mem_slots_index_type _mem_slots_locked;
};

class VIRTUAL_TEXTURING_DLL Cut : public DoubleBuffer<CutState>
{
  public:
    static Cut& init_cut(pre::AtlasFile * atlas);
    ~Cut() override{};

    pre::AtlasFile *get_atlas() const;

    bool is_drawn() const;
    void set_drawn(bool _drawn);

    static uint32_t get_dataset_id(uint64_t cut_id);
    static uint16_t get_view_id(uint64_t cut_id);
    static uint16_t get_context_id(uint64_t cut_id);

  protected:
    void deliver() override;

  private:
    Cut(pre::AtlasFile *atlas, CutState *front, CutState *back);

    pre::AtlasFile *_atlas;
    bool _drawn;
};
}

#endif // LAMURE_CUT_H
