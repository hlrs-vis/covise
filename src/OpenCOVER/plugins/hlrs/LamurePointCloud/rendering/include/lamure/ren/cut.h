// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CUT_H_
#define REN_CUT_H_

#include <vector>
#include <deque>

#include <lamure/types.h>
#include <lamure/ren/platform.h>
#include <lamure/semaphore.h>
#include <lamure/ren/cache_queue.h>
#include <lamure/utils.h>

namespace lamure {
namespace ren {

class cut
{
public:
                        cut();
                        cut(const context_t context_id, const view_t view_id, const model_t model_id);
    virtual             ~cut();

    struct node_slot_aggregate
    {
        node_slot_aggregate(
            const node_t node_id,
            const slot_t slot_id)
            : node_id_(node_id),
            slot_id_(slot_id) {};

        node_t          node_id_;
        slot_t          slot_id_;
    };

    const context_t     context_id() const { return context_id_; };
    const view_t        view_id() const { return view_id_; };
    const model_t       model_id() const { return model_id_; };

    std::vector<cut::node_slot_aggregate>& complete_set() { return complete_set_; };
    void                set_complete_set(std::vector<cut::node_slot_aggregate>& complete_set) { complete_set_ = complete_set; };


protected:

private:

    context_t           context_id_;
    view_t              view_id_;
    model_t             model_id_;

    std::vector<cut::node_slot_aggregate> complete_set_;
};


} } // namespace lamure


#endif // REN_CUT_H_
