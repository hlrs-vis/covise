// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CACHE_H_
#define REN_CACHE_H_

#include <map>
#include <queue>
#include <lamure/utils.h>

#include <memory>
#include <mutex>
#include <scm/core.h>
#include <scm/gl_core.h>

#include <lamure/ren/model_database.h>
#include <lamure/ren/policy.h>

#include <lamure/types.h>
#include <lamure/ren/platform.h>
#include <lamure/ren/cache_index.h>

namespace lamure {
namespace ren {

class cache
{
public:
                        cache(const cache&) = delete;
                        cache& operator=(const cache&) = delete;
    virtual             ~cache();

    const bool          is_node_resident(const model_t model_id, const node_t node_id);

    const slot_t        num_free_slots();
    const slot_t        slot_id(const model_t model_id, const node_t node_id);

    const slot_t        num_slots() const { return num_slots_; };
    const slot_t        slot_size() const { return slot_size_; };

    void                lock();
    void                unlock();

    void                aquire_node(const context_t context_id, const view_t view_id, const model_t model_id, const node_t node_id);
    void                release_node(const context_t context_id, const view_t view_id, const model_t model_id, const node_t node_id);
    const bool          release_node_invalidate(const context_t context_id, const view_t view_id, const model_t model_id, const node_t node_id);

protected:
                        cache(const slot_t num_slots);

    cache_index*        index_;
    std::mutex          mutex_;

private:
    /* data */

    slot_t              num_slots_;
    node_t              num_nodes_;
    size_t              slot_size_;


};


} } // namespace lamure


#endif // REN_CACHE_H_
