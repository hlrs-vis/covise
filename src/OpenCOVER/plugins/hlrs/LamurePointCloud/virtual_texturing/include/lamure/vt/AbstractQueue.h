// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_ABSTRACTQUEUE_H
#define VT_ABSTRACTQUEUE_H

#include <cstddef>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace vt {

    template<typename content_type>
    class AbstractQueue;

    template<typename content_type>
    class AbstractQueueEntry {
    protected:
        std::atomic<AbstractQueueEntry<content_type> *> _prev;
        std::atomic<AbstractQueueEntry<content_type> *> _next;

        std::atomic<AbstractQueue<content_type> *> _queue;
        std::atomic<content_type> _content;

        friend class AbstractQueue<content_type>;

    public:
        explicit AbstractQueueEntry(content_type content, AbstractQueue<content_type> *queue) {
            _content.store(content);
            _queue.store(queue);
        }

        virtual ~AbstractQueueEntry() {

        }

        virtual void setPrev(AbstractQueueEntry<content_type> *prev) {
            _prev.store(prev);
        }

        virtual void setNext(AbstractQueueEntry<content_type> *next) {
            _next.store(next);
        }

        virtual AbstractQueueEntry<content_type> *getPrev() {
            return _prev.load();
        }

        virtual AbstractQueueEntry<content_type> *getNext() {
            return _next.load();
        }

        virtual content_type getContent() {
            return _content.load();
        }

        virtual void remove() {
            _queue.load()->remove(*this);
        }

        virtual AbstractQueue<content_type> *getQueue(){
            return _queue.load();
        }
    };

    template<typename content_type>
    class AbstractQueue {
    protected:
        alignas(64) std::condition_variable _newEntry;

        alignas(64) std::mutex _lock;
        std::atomic<size_t> _size;
        std::atomic<AbstractQueueEntry<content_type> *> _first;
        std::atomic<AbstractQueueEntry<content_type> *> _last;

        virtual void _extractUnsafe(AbstractQueueEntry<content_type> &entry) {
            auto prev = entry.getPrev();
            auto next = entry.getNext();

            // remove
            if (prev == nullptr) {
                _first.store(next);
            } else {
                prev->setNext(next);
            }

            if (next == nullptr) {
                _last.store(prev);
            } else {
                next->setPrev(prev);
            }
        }

        virtual AbstractQueueEntry<content_type> *_popFrontUnsafe() {
            auto first = _first.load();

            if (first != nullptr) {
                _extractUnsafe(*first);
            }

            return first;
        }

        virtual AbstractQueueEntry<content_type> *_popBackUnsafe() {
            auto last = _last.load();

            if (last != nullptr) {
                _extractUnsafe(*last);
            }

            return last;
        }

        void _incrementSize(int8_t add){
            _size.store(_size.load() + add);
        }

    public:
        AbstractQueue() {
            _size.store(0);
            _first.store(nullptr);
            _last.store(nullptr);
        }

        virtual ~AbstractQueue() {
            auto entry = _first.load();

            while (entry != nullptr) {
                auto next = entry->getNext();
                delete entry;
                entry = next;
            }
        }

        virtual void push(content_type &content) = 0;

        virtual bool pop(content_type &content, const std::chrono::milliseconds maxTime) = 0;

        size_t getSize() {
            return _size.load();
        }

        virtual void remove(AbstractQueueEntry<content_type> &entry) {
            std::lock_guard<std::mutex> lock(_lock);
            _extractUnsafe(entry);
            delete &entry;
            _incrementSize(-1);
        }
    };
}

#endif //VT_ABSTRACTQUEUE_H
