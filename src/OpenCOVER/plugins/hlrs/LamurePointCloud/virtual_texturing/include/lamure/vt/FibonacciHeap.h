// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_FIBONACCIHEAP_H
#define VT_FIBONACCIHEAP_H

#include <cstddef>
#include <map>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace vt {

    template<typename key_type, typename value_type>
    class FibonacciHeap {
    public:
        enum MARK {
            ROOT = 1,
            NONE,
            ONE_CHILD_REMOVED
        };

    protected:
        alignas(64) std::mutex _lock;
        std::condition_variable _newEntry;

        FibonacciHeap<key_type, value_type> *_parent;
        FibonacciHeap<key_type, value_type> *_min;
        FibonacciHeap<key_type, value_type> *_prev;
        FibonacciHeap<key_type, value_type> *_next;

        size_t _rank;
        size_t _size;

        key_type _key;
        value_type _value;

        MARK _mark;
        size_t _actions;
        size_t _consolidateFreq;

        static void _insertBefore(FibonacciHeap<key_type, value_type> *first,
                                  FibonacciHeap<key_type, value_type> *last,
                                  FibonacciHeap<key_type, value_type> *ref){
            if(ref == nullptr){
                return;
            }
            first->_prev = ref->_prev;
            last->_next = ref;
            ref->_prev->_next = first;
            ref->_prev = last;

            size_t size = 0;
            size_t rank = 0;

            for(auto heap = first; ; heap = heap->_next){
                heap->_parent = ref->_parent;
                size += heap->_size;
                ++rank;

                if(heap == last){
                    break;
                }
            }

            ref->_parent->_rank += rank ;
            ref->_parent->_addToSize(size + rank);

            if(ref == ref->_parent->_min && first->_key <= ref->_key){
                ref->_parent->_min = first;
            }
        }

        void _union(FibonacciHeap<key_type, value_type> *min) {
            if (min == nullptr) {
                return;
            }

            FibonacciHeap<key_type, value_type> *heap = min;

            if (_min == nullptr) {
                _min = heap;

                do {
                    ++_rank;
                    _addToSize(heap->_size + 1);
                    heap->_parent = this;

                    if(heap == this){
                        break;
                    }

                    heap = heap->_next;
                } while (heap != min);

                return;
            }

            FibonacciHeap<key_type, value_type> *current = _min;
            FibonacciHeap<key_type, value_type> *first = min;
            FibonacciHeap<key_type, value_type> *last = first;

            for(size_t i = 0; ; ++i){
                if(heap->_key > current->_key){
                    if(i > 0){
                        FibonacciHeap<key_type, value_type>::_insertBefore(first, last, current);
                    }

                    while(current->_key < heap->_key){
                        current = current->_next;

                        if(current == _min || current == nullptr){
                            if(current == nullptr)
                            {
                                current = _min;
                                std:: cout << _min->_prev << std::endl;
                                std::cerr << "Heap problem" << std::endl;
                            }
                            break;
                        }
                    }

                    first = heap;
                }

                last = heap;
                heap = heap->_next;

                if(heap == min){
                    FibonacciHeap<key_type, value_type>::_insertBefore(first, last, current);
                    break;
                }
            }
        }

        void _cut() {
            if (_parent != nullptr) {
                if (_parent->_min == this) {
                    _parent->_min = (_next == this ? nullptr : _next);
                }

                --_parent->_rank;
                _parent->_addToSize(-_size - 1);
                _parent = nullptr;
            }

            _prev->_next = _next;
            _next->_prev = _prev;
            _prev = this;
            _next = this;
        }

        FibonacciHeap<key_type, value_type> *_getRoot() {
            auto parent = _parent;

            if (parent == nullptr) {
                return this;
            }

            return parent->_getRoot();
        }

        void _decreaseKey(key_type content) {
            auto root = _getRoot();
            auto parent = _parent;

            _cut();
            root->_union(this);

            while (parent->_mark == MARK::ONE_CHILD_REMOVED) {
                parent->_mark = MARK::NONE;
                parent->_cut();
                root->_union(parent);

                parent = parent->_parent;
            }
        }

        void _remove(){
            _decreaseKey(0);
            _cut();
        }

        void _addToSize(int64_t val){
            _size += val;

            if(_parent != nullptr){
                _parent->_addToSize(val);
            }
        }

        void _consolidate() {
            auto heap = _min;

            if(heap == nullptr){
                return;
            }

            std::map<size_t, FibonacciHeap<key_type, value_type>*> ranks;

            do{
                typename std::map<size_t, FibonacciHeap<key_type, value_type>*>::iterator iter = ranks.find(heap->_rank);
                heap->_cut();

                if(iter == ranks.end()){
                    ranks.insert(std::pair<size_t, FibonacciHeap<key_type, value_type>*>(heap->_rank, heap));
                }else{
                    auto ref = iter->second;

                    if(heap->_key <= ref->_key){
                        heap->_union(ref);
                    }else{
                        ref->_union(heap);
                        heap = ref;
                    }

                    ranks.erase(iter);
                    continue;
                }

                heap = _min;
            }while(heap != nullptr);

            for(auto rank : ranks){
                _union(rank.second);
            }
        }

    public:
        explicit FibonacciHeap(key_type key, value_type value) {
            _parent = nullptr;
            _min = nullptr;
            _prev = this;
            _next = this;
            _rank = 0;
            _size = 0;
            _key = key;
            _value = value;
            _mark = MARK::ROOT;
            _actions = 0;
            _consolidateFreq = 0;
        }

        virtual ~FibonacciHeap(){
            //lock_guard<mutex> lock(_lock);

            for(auto heap = _min; heap != nullptr; ){
                auto next = heap->_next;
                delete heap;

                if(next == _min){
                    break;
                }

                heap = next;
            }
        }

        virtual void setCleanupFrequency(size_t consolidateFreq){
            _consolidateFreq = consolidateFreq;
        }

        virtual void push(key_type key, value_type value){
            auto heap = new FibonacciHeap<key_type, value_type>(key, value);

            std::unique_lock<std::mutex> lock(_lock);

            bool newEntry = (this->_size == 0);

            _union(heap);

            ++_actions;

            if(_consolidateFreq > 0 && (_actions % _consolidateFreq) == 0){
                _consolidate();
            }

            if(newEntry){
                lock.unlock();

                _newEntry.notify_all();
            }
        }

        virtual bool pop(value_type &val, std::chrono::milliseconds maxTime = std::chrono::milliseconds::zero()){
            std::unique_lock<std::mutex> lock(_lock);

            if(_min == nullptr){
                if(!_newEntry.wait_for(lock, maxTime, [this] {
                    return _min != nullptr;
                })){
                    return false;
                }
            }

            auto min = _min;
            min->_cut();
            _union(min->_min);

            ++_actions;

            if(_consolidateFreq > 0 && (_actions % _consolidateFreq) == 0){
                _consolidate();
            }

            lock.unlock();

            val = min->_value;
            min->_min = nullptr;
            delete min;

            return true;
        }

        size_t size(){
            return _size;
        }
    };
}

#endif //TILE_PROVIDER_FIBONACCIHEAP_H
