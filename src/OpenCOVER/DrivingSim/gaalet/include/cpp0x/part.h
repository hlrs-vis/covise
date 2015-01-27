/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_PART_H
#define __GAALET_PART_H

#include "utility.h"

namespace gaalet
{

template <class A, conf_t... elements>
struct part : public expression<part<A, elements...> >
{
    typedef typename mv<elements...>::type::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    part(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return (search_element<conf, clist>::index >= clist::size) ? 0.0 : a.element<conf>();
    }

protected:
    const A &a;
};

template <class T, class A>
struct part_type : public expression<part_type<T, A> >
{
    typedef typename T::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    part_type(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return (search_element<conf, clist>::index >= clist::size) ? 0.0 : a.element<conf>();
    }

protected:
    const A &a;
};

} //end namespace gaalet

/// Projection on a sub-space of a multivector.
/**
 * Returns the elements of the multivector belonging to a sub-space defined by variadic template parameter \p elements. \p elements is a list of configuration bitmaps, e.g. part<1,2,4>().
 * \param elements List of elements defining the sub-spaces to project onto.
 */
/// \ingroup ga_ops
template <gaalet::conf_t... elements, class A>
inline gaalet::part<A, elements...>
part(const gaalet::expression<A> &a)
{
    return gaalet::part<A, elements...>(a);
}

/// Projection on a sub-space of a multivector.
/**
 * Returns the elements of the multivector belonging to a sub-space defined by configuration list of multivector \p T.
 * \param T Multivector with configuration list defining the sub-spaces to project onto.
 */
/// \ingroup ga_ops
template <class T, class A>
inline gaalet::part_type<T, A>
part_type(const gaalet::expression<A> &a)
{
    return gaalet::part_type<T, A>(a);
}

#endif
