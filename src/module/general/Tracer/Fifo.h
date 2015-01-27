/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FIFO_H_
#define _FIFO_H_

// template for a fifo of
// objects which have a default constructor
template <class T>
class Fifo
{
private:
    struct myPair
    {
        T object_;
        myPair *next_;
    } *first_;

public:
    Fifo()
    {
        first_ = 0;
    }
    int isEmpty()
    {
        return (first_ == 0);
    }
    T extract()
    {
        myPair *extracted;
        // What if first_ == 0 ???????
        // Well, then the programme crashes, so, don't do it!!!
        extracted = first_;
        first_ = first_->next_;
        T ret = extracted->object_;
        delete extracted;
        return ret;
    }
    void add(T new_obj)
    {
        myPair *new_pair = new myPair;
        new_pair->object_ = new_obj;
        new_pair->next_ = first_;
        first_ = new_pair;
    }
    void clean()
    {
        while (first_)
        {
            myPair *next = first_->next_;
            delete first_;
            first_ = next;
        }
    }
    ~Fifo()
    {
        clean();
    }
};
#endif
