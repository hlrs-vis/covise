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
#include <array>

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
    multivector()
    {
        //std::fill(data, data+size, 0.0);
        //std::fill(data.begin(), data.end(), null_element);
        for (unsigned int i = 0; i < size; ++i)
            data[i] = null_element<element_t>::value();
    }

    multivector(std::initializer_list<element_t> s)
    {
        //element_t* last = std::copy(s.begin(), (s.size()<=size) ? s.end() : (s.begin()+size), data);
        typename std::array<element_t, size>::iterator last = std::copy(s.begin(), (s.size() <= size) ? s.end() : (s.begin() + size), data.begin());
        //std::fill(last, data+size, 0.0);
        //std::fill(last, data.end(), null_element);
        for (typename std::array<element_t, size>::iterator it = last; it != data.end(); ++it)
            (*it) = null_element<element_t>::value();
    }

    //return element by index, index known at runtime
    const element_t &operator[](const conf_t &index) const
    {
        return data[index];
    }
    element_t &operator[](const conf_t &index)
    {
        return data[index];
    }

    //return element by index, index known at compile time
    template <conf_t index>
    const element_t &get() const
    {
        return data[index];
    }
    template <conf_t index>
    element_t &get()
    {
        return data[index];
    }

    //return element by configuration (basis vector), configuration known at compile time
    template <conf_t conf>
    ////reference return (const element_t& element() const) not applicable because of possible return of 0.0;
    element_t element() const
    {
        static const conf_t index = search_element<conf, clist>::index;
        //static_assert(index<size, "element<conf_t>(): no such element in configuration list");
        return (index < size) ? data[index] : null_element<element_t>::value();
    }

    //evaluation
    template <typename E, conf_t index = 0>
    struct ElementEvaluation
    { //v no reference to pointer *& with gcc4.5 possible... What's going on?
        /*static void eval(element_t* const data, const E& e) {
         data[index] = e.element<get_element<index, clist>::value>();
         ElementEvaluation<E, index+1>::eval(data, e);
      }*/
        static void eval(std::array<element_t, size> &data, const E &e)
        {
            std::get<index>(data) = e.element<get_element<index, clist>::value>();
            ElementEvaluation<E, index + 1>::eval(data, e);
        }
    };
    template <typename E>
    struct ElementEvaluation<E, size - 1>
    {
        /*static void eval(element_t* const data, const E& e) {
         data[size-1] = e.element<get_element<size-1, clist>::value>();
      }*/
        static void eval(std::array<element_t, size> &data, const E &e)
        {
            std::get<size - 1>(data) = e.element<get_element<size - 1, clist>::value>();
        }
    };

    //   constructor evaluation
    template <class E>
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
    void operator=(const expression<E> &e_)
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
    void assign(const expression<E> &e_)
    {
        const E &e(e_);
        ElementEvaluation<E>::eval(data, e);
    }

protected:
    //element_t data[size];
    std::array<element_t, size> data;
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
    multivector()
        : value(null_element<element_t>::value())
    {
    }

    multivector(const element_t &setValue)
        : value(setValue)
    {
    }

    multivector(std::initializer_list<element_t> s)
        : value(*s.begin())
    {
    }

    //conversion operator
    operator element_t()
    {
        return value;
    }

    //return element by index, index known at runtime
    const element_t &operator[](const conf_t &) const
    {
        return value;
    }
    element_t &operator[](const conf_t &index)
    {
        return value;
    }

    //return element by index, index known at compile time
    template <conf_t index>
    const element_t &get() const
    {
        return value;
    }

    //return element by configuration (basis vector), configuration known at compile time
    template <conf_t conf>
    ////reference return (const element_t& element() const) not applicable because of possible return of 0.0;
    element_t element() const
    {
        //static const conf_t index = search_element<conf, clist>::index;
        //static_assert(index<size, "element<conf_t>(): no such element in configuration list");
        return (conf == 0x00) ? value : null_element<element_t>::value();
    }

    //   constructor evaluation
    template <class E>
    multivector(const expression<E> &e_)
    {
        const E &e(e_);
        value = e.element<0x00>();
    }

    //   assignment evaluation
    template <class E>
    void operator=(const expression<E> &e_)
    {
        const E &e(e_);
        value = e.element<0x00>();
    }

protected:
    element_t value;
};

} //end namespace gaalet

template <class A>
inline gaalet::multivector<typename A::clist, typename A::metric, typename A::element_t>
eval(const gaalet::expression<A> &a)
{
    return gaalet::multivector<typename A::clist, typename A::metric, typename A::element_t>(a);
}

#endif
