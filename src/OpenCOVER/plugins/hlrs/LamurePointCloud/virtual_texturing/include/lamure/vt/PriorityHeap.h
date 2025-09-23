// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_PRIORITYHEAP_H
#define VT_PRIORITYHEAP_H

#include <atomic>
#include <mutex>
#include <lamure/vt/FibonacciHeap.h>

namespace vt {

    template<typename priority_type>
    class PriorityHeapContent;

    template<typename priority_type>
    class PriorityHeap : public FibonacciHeap<priority_type, PriorityHeapContent<priority_type>*> {
    protected:
        typedef priority_type key_type;
        typedef PriorityHeapContent<priority_type>* value_type;

        void _remove() {
            auto root = (PriorityHeap<priority_type> *) this->_getRoot();
            this->_cut();

            root->_union(this->_min);

            this->_parent = nullptr;
            this->_min = nullptr;
            this->_rank = 0;
            this->_size = 0;
        }

    public:
        explicit PriorityHeap(priority_type key, PriorityHeapContent<priority_type> *value) : FibonacciHeap<priority_type, PriorityHeapContent<priority_type>*>(std::numeric_limits<key_type>::max() - key, value) {

        }

        ~PriorityHeap(){
            if(this->_value != nullptr) {
                this->_value->setHeap(nullptr);
            }
        }

        virtual void push(key_type key, value_type value);

        void remove(){
            std::lock_guard<std::mutex> lock(this->_lock);
            _remove();
        }

        void reinsert(key_type key){
            std::lock_guard<std::mutex> lock(this->_lock);

            auto root = (PriorityHeap<priority_type>*)this->_getRoot();
            this->_cut();

            root->_union(this->_min);

            this->_min = nullptr;
            this->_rank = 0;
            this->_size = 0;
            this->_key = key;

            root->_union(this);

            ++this->_actions;

            if(this->_consolidateFreq > 0 && (this->_actions % this->_consolidateFreq) == 0){
                this->_consolidate();
            }
        }

        void setValue(value_type value){
            this->_value = value;
        }
    };

    template<typename priority_type>
    class PriorityHeapContent {
    protected:
        std::mutex _lock;

        priority_type _priority;
        std::atomic<PriorityHeap<priority_type>*> _heap;

    public:
        PriorityHeapContent(){
            _heap.store(nullptr);
        }

        ~PriorityHeapContent(){
            if(_heap.load() != nullptr){
                _heap.load()->setValue(nullptr);
            }
        }

        virtual void remove(){
            std::unique_lock<std::mutex> lock(_lock);

            auto heap = _heap.load();

            if(heap != nullptr){
                _heap = nullptr;

                lock.unlock();

                heap->remove();
                delete heap;
            }
        }

        void setPriority(priority_type priority){
            std::lock_guard<std::mutex> lock(_lock);

            priority = std::numeric_limits<priority_type>::max() - priority;

            if(_priority != priority){
                _priority = priority;

                if(_heap != nullptr){
                    _heap.load()->reinsert(priority);
                }
            }
        }

        priority_type getPriority(){
            std::lock_guard<std::mutex> lock(_lock);

            return std::numeric_limits<priority_type>::max() - _priority;
        }

        void setHeap(PriorityHeap<priority_type> *heap){
            std::lock_guard<std::mutex> lock(this->_lock);

            _heap.store(heap);
        }
    };


    template<typename priority_type>
    void PriorityHeap<priority_type>::push(priority_type key, PriorityHeapContent<priority_type> *value){
        auto heap = new PriorityHeap<priority_type>(key, value);

        std::unique_lock<std::mutex> lock(this->_lock);

        bool newEntry = (this->_size == 0);

        this->_union(heap);
        ++this->_actions;

        if(this->_consolidateFreq > 0 && (this->_actions % this->_consolidateFreq) == 0){
            this->_consolidate();
        }

        lock.unlock();

        if(newEntry){
            this->_newEntry.notify_all();
        }

        value->setHeap(heap);
    }

}

#endif //TILE_PROVIDER_PRIORITYHEAP_H
