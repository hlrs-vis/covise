/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INFINITE_ARRAY_H_
#define _INFINITE_ARRAY_H_

template <class T>
class ia
{
private:
    T *array;
    int length;
    int max_reference;
    ia(const ia &); // copy constr. and Zuweisung nicht implementiert:
    // Nachmacher sind nicht erwuenscht
    ia &operator=(const ia &);

public:
    void crop(int i)
    {
        if (i <= 0)
            return;
        max_reference -= i;
        if (max_reference < -1)
            max_reference = -1;
    }

    T *getArray()
    {
        return array;
    }

    const T *getArray() const
    {
        return array;
    }

    int size() const
    {
        return max_reference + 1;
    }

    ia(int i = 0)
    {
        array = 0;
        if (i < 0)
            i = 0;
        length = i;
        if (i)
        {
            array = new T[i];
        }
        max_reference = -1;
    }

    void clean()
    {
        max_reference = -1;
    }

    void schleifen()
    {
        delete[] array;
        array = 0;
        length = 0;
        clean();
    }
    ~ia()
    {
        delete[] array;
    }

    void reserve(int new_length, int resize = 0)
    {
        if (length < new_length && new_length > 0)
        {
            T *tmp = new T[new_length];
            if (length)
            {
                memcpy(tmp, array, sizeof(T) * length);
            }
            delete[] array;
            array = tmp;
            length = new_length;
            if (resize)
            {
                max_reference = length - 1;
            }
        }
    }

    // This might be more efficient if we assume that
    // we are not accessing elements beyond the actual
    // array limit, in other words, if we restrict this
    // operator as done in the vector template class of the STL.
    T &operator[](int i)
    {
        if (i > max_reference)
        {
            max_reference = i;
        }
        if (i >= length)
        {
            T *tmp;
            int new_length;
            if (length == 0)
            {
                new_length = i + 1;
            }
            else
            {
                new_length = ((i / length) + 1) * length;
            }
            tmp = new T[new_length];
            if (length)
            {
                // only simple types!!!!
                memcpy(tmp, array, sizeof(T) * length); // Not valid for classes!!!
                delete[] array;
            }
            array = tmp;
            length = new_length;
        }
        return array[i];
    }
    void push_back(const T &obj)
    {
        // easy way, but not very efficient
        // operator[](size())=obj;
        ++max_reference;
        if (max_reference >= length)
        {
            T *tmp;
            int new_length;
            if (length == 0)
            {
                new_length = max_reference + 1;
            }
            else
            {
                new_length = ((max_reference / length) + 1) * length;
            }
            tmp = new T[new_length];
            // only simple types!!!!
            memcpy(tmp, array, sizeof(T) * length); // Not valid for classes!!!
            delete[] array;
            array = tmp;
            length = new_length;
        }
        array[max_reference] = obj;
    }
};
#endif
