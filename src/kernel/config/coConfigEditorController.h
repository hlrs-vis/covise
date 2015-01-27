/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//templates here. Observer for entry is coConfigEditorEntry
#ifndef COCONFIGOBSERVER
#define COCONFIGOBSERVER
#include <vector>
namespace covise
{

template <class T>
class Observer
{

public:
    Observer()
    {
    }
    virtual ~Observer()
    {
    }

    virtual void update(T *subject) = 0;
};

template <class T>
class Subject
{
public:
    Subject()
    {
    }
    virtual ~Subject()
    {
    }

    void attach(Observer<T> &observer)
    {
        this->observers.push_back(&observer);
    }

    void notify()
    {
        //       vector<Observer<T> *>::iterator it;

        typename std::vector<Observer<T> *>::iterator it;
        for (it = this->observers.begin();
             it != this->observers.end();
             ++it)
            (*it)->update(static_cast<T *>(this));
    }

private:
    std::vector<Observer<T> *> observers;
};
}
#endif
