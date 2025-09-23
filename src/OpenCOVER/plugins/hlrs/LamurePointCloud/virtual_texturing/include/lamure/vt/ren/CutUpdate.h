// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_CUTUPDATE_H
#define LAMURE_CUTUPDATE_H

#include <lamure/vt/QuadTree.h>
#include <lamure/vt/common.h>
#include <lamure/vt/VTConfig.h>
#include <lamure/vt/ren/Cut.h>

namespace vt
{
typedef std::set<id_type> id_set_type;

class VIRTUAL_TEXTURING_DLL CutUpdate
{
  public:
    static CutUpdate &get_instance()
    {
        static CutUpdate instance;
        return instance;
    }
    CutUpdate(CutUpdate const &) = delete;
    void operator=(CutUpdate const &) = delete;

    ~CutUpdate();

    void start();
    void stop();

    void feedback(int32_t *buf_lod, uint32_t *buf_count);
    const float &get_dispatch_time() const;

    void toggle_freeze_dispatch();

private:
    CutUpdate();

    std::thread _worker;
    std::mutex _dispatch_lock;
    std::condition_variable _cv;
    std::atomic<bool> _new_feedback;

    VTConfig *_config;
    CutDatabase *_cut_db;

    float _dispatch_time;

    int32_t *_feedback_lod_buffer;
    uint32_t *_feedback_count_buffer;

    std::atomic<bool> _should_stop;
    std::atomic<bool> _freeze_dispatch;

    void run();
    void dispatch();

    bool collapse_to_id(Cut *cut, id_type tile_id);
    bool split_id(Cut *cut, id_type tile_id);
    bool keep_id(Cut *cut, id_type tile_id);

    bool add_to_indexed_memory(Cut *cut, id_type tile_id, uint8_t *tile_ptr);
    mem_slot_type *write_mem_slot_for_id(Cut *cut, id_type tile_id);

    bool check_all_siblings_in_cut(id_type tile_id, const cut_type &cut);
    void remove_from_indexed_memory(Cut *cut, id_type tile_id);
};
}

#endif // LAMURE_CUTUPDATE_H
