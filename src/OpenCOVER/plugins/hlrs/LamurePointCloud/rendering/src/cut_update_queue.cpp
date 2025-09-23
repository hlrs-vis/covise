// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/cut_update_queue.h>


namespace lamure
{

namespace ren
{

cut_update_queue::
cut_update_queue() {

}

cut_update_queue::
~cut_update_queue() {

}

void cut_update_queue::
push_job(const job& job) {
    std::lock_guard<std::mutex> lock(mutex_);
    job_queue_.push(job);
}

const cut_update_queue::job cut_update_queue::
pop_front_job() {
    std::lock_guard<std::mutex> lock(mutex_);

    job job;

    if (!job_queue_.empty()) {
        job = job_queue_.front();
        job_queue_.pop();
    }

    return job;
}

const size_t cut_update_queue::
num_jobs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return job_queue_.size();
}

} // namespace ren

} // namespace lamure

