/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_PART_H
#define __GAALET_PART_H

#include "utility.h"

namespace gaalet
{

/*C++0x only: template<class A, conf_t... elements>
struct part : public expression<part<A, elements...> >
{
   typedef typename mv<elements...>::type::clist clist;

   typedef typename A::metric metric;

   part(const A& a_)
      :  a(a_)
   { }

   template<conf_t conf>
   element_t element() const {
      return a.element<conf>();
   }

protected:
   const A& a;
};*/

template <class T, class A>
struct Part_type : public expression<Part_type<T, A> >
{
    typedef typename T::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    Part_type(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    GAALET_CUDA_HOST_DEVICE
        element_t
            element() const
    {
        return (search_element<conf, clist>::index >= clist::size) ? 0.0 : a.element<conf>();
    }

protected:
    const A &a;
};

} //end namespace gaalet

/*C++0x only: template<gaalet::conf_t... elements, class A> inline
gaalet::part<A, elements...>
part(const gaalet::expression<A>& a) {
   return gaalet::part<A, elements...>(a);
}*/

template <class T, class A>
inline GAALET_CUDA_HOST_DEVICE
    gaalet::Part_type<T, A>
    part_type(const gaalet::expression<A> &a)
{
    return gaalet::Part_type<T, A>(a);
}

#endif
