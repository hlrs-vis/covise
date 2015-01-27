/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TOOLS_HELPER_H_
#define __TOOLS_HELPER_H_

#include "Singleton.h"

using namespace std;

namespace Tools
{
class Helper : public Singleton<Helper>
{
    friend class Singleton<Helper>::InstanceHolder;

private:
    Helper();

public:
    virtual ~Helper();

    template <class T>
    void clearArray(T *array, int length)
    {
        if ((length <= 0) || (array == NULL))
            return;

        for (int i = 0; i < length; i++)
            if (array[i] != NULL)
            {
                delete array[i];
                array[i] = NULL;
            }
    }

    template <class T>
    T *vectorToArray(vector<T> values)
    {
        if (values.size() == 0)
            return NULL;

        T *array = new T[values.size()];

        for (int i = 0; i < values.size(); i++)
            array[i] = values[i];

        return array;
    }

    template <class T>
    vector<T> arrayToVector(T *values, int length)
    {
        vector<T> list;

        if (length > 0)
        {
            list.assign(length, values[0]);

            for (int i = 0; i < length; i++)
                list[i] = values[i];
        }

        return list;
    }
};
};
#endif
