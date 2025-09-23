// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef VT_TILEREQUESTPRIORITYQUEUE_H
#define VT_TILEREQUESTPRIORITYQUEUE_H

#include <lamure/vt/AbstractQueue.h>
#include <lamure/vt/ooc/TileRequest.h>

namespace vt{
    template<typename priority_type>
    class TileRequestPriorityQueue;

    template<typename priority_type>
    class TileRequestPriorityQueueEntry : public AbstractQueueEntry<vt::ooc::TileRequest*>{
    protected:
        std::mutex _deleteLock;
    public:
        explicit TileRequestPriorityQueueEntry(ooc::TileRequest *req, TileRequestPriorityQueue<priority_type> &queue);

        ~TileRequestPriorityQueueEntry() {
        }

        virtual priority_type getPriority(){
            return this->_content.load()->getPriority();
        }
    };

    template<typename priority_type>
    TileRequestPriorityQueueEntry<priority_type>::TileRequestPriorityQueueEntry(ooc::TileRequest *req,
                                                          TileRequestPriorityQueue<priority_type> &queue) :
            AbstractQueueEntry<ooc::TileRequest*>(req, &queue){
    }

    template<typename priority_type>
    class TileRequestPriorityQueue : public AbstractQueue<ooc::TileRequest*>{
    protected:
        virtual void _insertUnsafe(TileRequestPriorityQueueEntry<priority_type> &entry){
            auto next = (TileRequestPriorityQueueEntry<priority_type>*)this->_first.load();
            TileRequestPriorityQueueEntry<priority_type> *prev = nullptr;

            while(next != nullptr){
                if(next->getPriority() >= entry.getPriority()){
                    break;
                }

                prev = next;
                next = (TileRequestPriorityQueueEntry<priority_type>*)next->getNext();
            }

            entry.setPrev(prev);
            entry.setNext(next);

            if(prev == nullptr){
                this->_first.store(&entry);
            }else{
                prev->setNext(&entry);
            }

            if(next == nullptr){
                this->_last.store(&entry);
            }else{
                next->setPrev(&entry);
            }
        }

    public:
        virtual void reinsert(TileRequestPriorityQueueEntry<priority_type> &entry){
            std::lock_guard<std::mutex> lock(this->_lock);

            this->_extractUnsafe(entry);
            this->_insertUnsafe(entry);
        }

        virtual void push(ooc::TileRequest *&content){
            auto entry = new TileRequestPriorityQueueEntry<priority_type>(content, *this);

            std::lock_guard<std::mutex> lock(this->_lock);

            this->_insertUnsafe(*entry);
            this->_incrementSize(1);

        }

        virtual bool pop(ooc::TileRequest *&content, const std::chrono::milliseconds maxTime){
            std::unique_lock<std::mutex> lock(this->_lock);

            if(!this->_newEntry.wait_for(lock, maxTime, [this]{
                return this->_first.load() != nullptr;
            })){
                return false;
            }

            auto entry = (TileRequestPriorityQueueEntry<priority_type>*)this->_popBackUnsafe();

            if(entry == nullptr){
                return false;
            }

            content = entry->getContent();
            delete entry;

            this->_incrementSize(-1);

            return true;
        }

        virtual bool popLeast(ooc::TileRequest *&content, const std::chrono::milliseconds maxTime){
            std::unique_lock<std::mutex> lock(this->_lock);

            if(!this->_newEntry.wait_for(lock, maxTime, [this]{
                return this->_last.load() != nullptr;
            })){
                return false;
            }

            auto entry = (TileRequestPriorityQueueEntry<priority_type>*)this->_popFrontUnsafe();

            if(entry == nullptr){
                return false;
            }

            content = entry->getContent();
            delete entry;

            this->_incrementSize(-1);

            return true;
        }

        bool contains(ooc::TileRequest *content){
            std::lock_guard<std::mutex> lock(this->_lock);

            for(auto entry = this->_first.load(); entry != nullptr; entry = entry->getNext()){
                if(entry->getContent() == content){
                    return true;
                }
            }

            return false;
        }
    };

}

#endif //VT_TILEREQUESTPRIORITYQUEUE_H
