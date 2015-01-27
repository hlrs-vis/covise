/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_MULTIVECTOR_H
#define __GAALET_MULTIVECTOR_H

#include "configuration_list.h"
#include "expression.h"
#include "multivector_element.h"

#include <algorithm>
#include <cstring>
//C++0x only: #include <array>

namespace gaalet
{

//multivector struct
//template<typename CL, typename SL=sl::sl_null>
//struct multivector : public expression<multivector<CL, SL>>
template <typename CL, typename M, typename T>
struct multivector : public expression<multivector<CL, M, T> >
{
    typedef CL clist;
    static const conf_t size = clist::size;

    typedef M metric;

    typedef T element_t;

    //initialization
    GAALET_CUDA_HOST_DEVICE
    multivector()
    {
        memset(data, 0, size * sizeof(element_t));
        //std::fill(data, data+size, 0.0);
        //std::fill(data.begin(), data.end(), 0.0);
    }

    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0)
    {
        memset(data, 0, (size - 1) * sizeof(element_t));
        //std::fill(data, data+size, 0.0);
        data[0] = c0;
        //std::fill(data.begin(), data.end(), 0.0);
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1)
    {
        memset(data, 0, (size - 2) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2)
    {
        memset(data, 0, (size - 3) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3)
    {
        memset(data, 0, (size - 4) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4)
    {
        memset(data, 0, (size - 5) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5)
    {
        memset(data, 0, (size - 6) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5, const element_t &c6)
    {
        memset(data, 0, (size - 7) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
        data[6] = c6;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5, const element_t &c6, const element_t &c7)
    {
        memset(data, 0, (size - 8) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
        data[6] = c6;
        data[7] = c7;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5, const element_t &c6, const element_t &c7, const element_t &c8)
    {
        memset(data, 0, (size - 9) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
        data[6] = c6;
        data[7] = c7;
        data[8] = c8;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5, const element_t &c6, const element_t &c7, const element_t &c8, const element_t &c9)
    {
        memset(data, 0, (size - 10) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
        data[6] = c6;
        data[7] = c7;
        data[8] = c8;
        data[9] = c9;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5, const element_t &c6, const element_t &c7, const element_t &c8, const element_t &c9, const element_t &c10)
    {
        memset(data, 0, (size - 11) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
        data[6] = c6;
        data[7] = c7;
        data[8] = c8;
        data[9] = c9;
        data[10] = c10;
    }
    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &c0, const element_t &c1, const element_t &c2, const element_t &c3, const element_t &c4, const element_t &c5, const element_t &c6, const element_t &c7, const element_t &c8, const element_t &c9, const element_t &c10, const element_t &c11)
    {
        memset(data, 0, (size - 12) * sizeof(element_t));
        data[0] = c0;
        data[1] = c1;
        data[2] = c2;
        data[3] = c3;
        data[4] = c4;
        data[5] = c5;
        data[6] = c6;
        data[7] = c7;
        data[8] = c8;
        data[9] = c9;
        data[10] = c10;
        data[11] = c11;
    }

    /*C++0x only: multivector(std::initializer_list<element_t> s)
   {
      element_t* last = std::copy(s.begin(), (s.size()<=size) ? s.end() : (s.begin()+size), data);
      //C++0x only: typename std::array<element_t, size>::iterator last = std::copy(s.begin(), (s.size()<=size) ? s.end() : (s.begin()+size), data.begin());
      std::fill(last, data+size, 0.0);
      //std::fill(last, data.end(), 0.0);
   }*/

    //return element by index, index known at runtime
    GAALET_CUDA_HOST_DEVICE
    const element_t &operator[](const conf_t &index) const
    {
        return data[index];
    }
    GAALET_CUDA_HOST_DEVICE
    element_t &operator[](const conf_t &index)
    {
        return data[index];
    }

    //return element by index, index known at compile time
    template <conf_t index>
    GAALET_CUDA_HOST_DEVICE const element_t &get() const
    {
        return data[index];
    }
    template <conf_t index>
    GAALET_CUDA_HOST_DEVICE
        element_t &
        get()
    {
        return data[index];
    }

    //return element by configuration (basis vector), configuration known at compile time
    template <conf_t conf>
    ////reference return (const element_t& element() const) not applicable because of possible return of 0.0;
    GAALET_CUDA_HOST_DEVICE
        element_t
            element() const
    {
        const conf_t index = search_element<conf, clist>::index;
        //static_assert(index<size, "element<conf_t>(): no such element in configuration list");
        return (index < size) ? data[index] : 0.0;
    }

    //evaluation
    static const conf_t lastElementIndex = size - 1;
    template <typename E, conf_t index = lastElementIndex>
    struct ElementEvaluation
    {
        static const conf_t elementBitmap = get_element<index, clist>::value;
        static const conf_t nextElementIndex = index - 1;

        //v no reference to pointer *& with gcc4.5 possible... What's going on?
        GAALET_CUDA_HOST_DEVICE
        static void eval(element_t *const data, const E &e)
        {
            data[index] = e.template element<elementBitmap>();
            ElementEvaluation<E, nextElementIndex>::eval(data, e);
        }
        /*C++0x only: static void eval(std::array<element_t, size>& data, const E& e) {
         std::get<index>(data) = e.element<get_element<index, clist>::value>();
         ElementEvaluation<E, index+1>::eval(data, e);
      }*/
    };
    template <typename E>
    struct ElementEvaluation<E, 0>
    {
        GAALET_CUDA_HOST_DEVICE
        static void eval(element_t *const data, const E &e)
        {
            data[0] = e.template element<get_element<0, clist>::value>();
        }
        /*C++0x only: static void eval(std::array<element_t, size>& data, const E& e) {
         std::get<size-1>(data) = e.element<get_element<size-1, clist>::value>();
      }*/
    };

    //   constructor evaluation
    template <class E>
    GAALET_CUDA_HOST_DEVICE
    multivector(const expression<E> &e_)
    {
        const E &e(e_);
        ElementEvaluation<E>::eval(data, e);
    }

    //copy --- seems slower with global eval function (overrides rvalue reference assignment operator?)
    /*void operator=(const multivector& mv)
   {
      std::copy(mv.data, mv.data+size, data);
   }*/

    //assignment evaluation
    template <class E>
    GAALET_CUDA_HOST_DEVICE void operator=(const expression<E> &e_)
    {
        //const E& e(e_);
        //ElementEvaluation<E>::eval(data, e);
        //multivector mv(e_);
        //*this = std::move(mv);
        *this = multivector(e_);
        //data = std::move(mv.data);
        //std::copy(mv.data, mv.data+size, data);

        //element_t temp_data[size];
        //std::copy(temp_data, temp_data+size, data);
        //
        //std::array<element_t, size> temp_data;
        //ElementEvaluation<E>::eval(temp_data, e);
        //std::copy(temp_data.begin(), temp_data.end(), data.begin());
        //data = std::move(temp_data);
    }

    //assignment without temporary
    template <class E>
    GAALET_CUDA_HOST_DEVICE void assign(const expression<E> &e_)
    {
        const E &e(e_);
        ElementEvaluation<E>::eval(data, e);
    }

protected:
    element_t data[size];
    //C++0x only: std::array<element_t, size> data;
};

//specialization for scalar multivector type
template <typename M, typename T>
struct multivector<configuration_list<0x00, cl_null>, M, T> : public expression<multivector<configuration_list<0x00, cl_null>, M, T> >
{
    typedef configuration_list<0x00, cl_null> clist;
    static const conf_t size = clist::size;

    typedef M metric;

    typedef T element_t;

    //initialization
    GAALET_CUDA_HOST_DEVICE
    multivector()
        : value(0.0)
    {
    }

    GAALET_CUDA_HOST_DEVICE
    multivector(const element_t &setValue)
        : value(setValue)
    {
    }

    /*C++0x only: multivector(std::initializer_list<element_t> s)
      :  value(*s.begin())
   { }*/

    //conversion operator
    GAALET_CUDA_HOST_DEVICE
    operator element_t()
    {
        return value;
    }

    //return element by index, index known at runtime
    GAALET_CUDA_HOST_DEVICE
    const element_t &operator[](const conf_t &) const
    {
        return value;
    }
    GAALET_CUDA_HOST_DEVICE
    element_t &operator[](const conf_t &index)
    {
        return value;
    }

    //return element by index, index known at compile time
    template <conf_t index>
    GAALET_CUDA_HOST_DEVICE const element_t &get() const
    {
        return value;
    }

    //return element by configuration (basis vector), configuration known at compile time
    template <conf_t conf>
    ////reference return (const element_t& element() const) not applicable because of possible return of 0.0;
    GAALET_CUDA_HOST_DEVICE
        element_t
            element() const
    {
        //static const conf_t index = search_element<conf, clist>::index;
        //static_assert(index<size, "element<conf_t>(): no such element in configuration list");
        return (conf == 0x00) ? value : 0.0;
    }

    //   constructor evaluation
    template <class E>
    GAALET_CUDA_HOST_DEVICE
    multivector(const expression<E> &e_)
    {
        const E &e(e_);
        value = e.template element<0x00>();
    }

    //   assignment evaluation
    template <class E>
    GAALET_CUDA_HOST_DEVICE void operator=(const expression<E> &e_)
    {
        const E &e(e_);
        value = e.template element<0x00>();
    }

protected:
    element_t value;
};

} //end namespace gaalet

template <class A>
inline GAALET_CUDA_HOST_DEVICE
    gaalet::multivector<typename A::clist, typename A::metric, typename A::element_t>
    eval(const gaalet::expression<A> &a)
{
    return gaalet::multivector<typename A::clist, typename A::metric, typename A::element_t>(a);
}

#endif
