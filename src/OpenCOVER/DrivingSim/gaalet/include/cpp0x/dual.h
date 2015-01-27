/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_DUAL_H
#define __GAALET_DUAL_H

#include "utility.h"

namespace gaalet
{

template <conf_t I, typename list, typename colist = cl_null>
struct dual_list
{
    typedef typename dual_list<I, typename list::tail, typename insert_element<I ^ list::head, colist>::clist>::clist clist;
};
template <conf_t I, typename colist>
struct dual_list<I, cl_null, colist>
{
    typedef colist clist;
};

template <class A>
struct dual : public expression<dual<A> >
{
    static const conf_t I = Power<2, A::metric::dimension>::value - 1;

    typedef typename dual_list<I, typename A::clist>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    dual(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return (search_element<conf, clist>::index >= clist::size) ? 0.0 : a.element<I ^ conf>()
                                                                           * (Power<-1, BitCount<I>::value *(BitCount<I>::value - 1) / 2>::value
                                                                              * CanonicalReorderingSign<I ^ conf, I>::value
                                                                              * ((BitCount<metric::signature_bitmap &((I ^ conf) & I)>::value % 2) ? -1 : 1));
    }

protected:
    const A &a;
};

} //end namespace gaalet

/// Dual of a multivector.
/** Depends on metric and should work with a non-degenerate algebra. When operated within degenerate algebra, a positive bilinear form of the corresponding basis vectors is assumed.
  */
/// \ingroup ga_ops

template <class A>
inline gaalet::dual<A>
dual(const gaalet::expression<A> &a)
{
    return gaalet::dual<A>(a);
}

#endif
