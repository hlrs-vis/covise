/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_SCALAR_H
#define __GAALET_SCALAR_H

#include "grade.h"
#include "geometric_product.h"

namespace gaalet
{

template <class L, class R>
struct scalar : public expression<scalar<L, R> >
{
    typedef gaalet::grade<0, gaalet::geometric_product<L, R> > E;

    typedef typename E::clist clist;

    typedef typename E::metric metric;

    typedef typename E::element_t element_t;

    scalar(const L &l_, const R &r_)
        //:  l(l_), r(r_)
        : e(::grade<0>(l_ *r_))
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return e.element<conf>();
        //return l.element<conf>() - r.element<conf>();
    }

protected:
    //const L& l;
    //const R& r;
    const E &e;
};

} //end namespace gaalet

/*template <class L, class R> inline
gaalet::scalar<L, R>
scalar(const gaalet::expression<L>& l, const gaalet::expression<R>& r)
{
   return gaalet::scalar<L, R>(l, r);
}*/

template <class L, class R>
inline auto scalar(const gaalet::expression<L> &l, const gaalet::expression<R> &r) -> decltype(grade<0>(l *r))
{
    return grade<0>(l * r);
}

#endif
