/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_INVERSE_H
#define __GAALET_INVERSE_H

#include "part.h"
#include "reverse.h"

namespace gaalet
{

//check for versor
template <typename CL, bool parent_even_grade = (BitCount<CL::head>::value % 2)>
struct check_versor
{
    static const bool even_grade = (BitCount<CL::head>::value % 2);
    static const bool value = (parent_even_grade == even_grade) ? check_versor<typename CL::tail, even_grade>::value : false;
};
template <bool parent_even_grade>
struct check_versor<cl_null, parent_even_grade>
{
    static const bool even_grade = parent_even_grade;
    static const bool value = true;
};

//go through inversion evaluation type checks
//value=1 - versor inversion
template <class A>
struct inverse_evaluation_type
{
    static const int value = (check_versor<typename A::clist>::value) ? 1 : -1;
};

template <class A, int ET = inverse_evaluation_type<A>::value>
struct inverse : public expression<inverse<A> >
{
    static_assert(ET != -1, "no method for evaluating this type of multivector implemented");
};

//versor inversion
template <class A>
struct inverse<A, 1> : public expression<inverse<A> >
{
    typedef typename A::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    inverse(const A &a_)
        : a(a_)
        , first_eval(true)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        //review: don't evaluate on definition workaround: will only work if arguments stay the same (thus attention with variables)
        if (first_eval)
        {
            div = 1.0 / ((~a) * a).template element<0x00>();
            first_eval = false;
        };
        return a.element<conf>() * div * Power<-1, BitCount<conf>::value *(BitCount<conf>::value - 1) / 2>::value;
    }

protected:
    const A &a;
    mutable element_t div;
    mutable bool first_eval;
};

} //end namespace gaalet

/// Inverse of a multivector.
/**
 * Only implemented for versors. Undefined for degenerate algebra.
 */
/// \ingroup ga_ops
template <class A>
inline gaalet::inverse<A>
operator!(const gaalet::expression<A> &a)
{
    return gaalet::inverse<A>(a);
}

/*template <class A> inline
auto operator!(const gaalet::expression<A>& a) -> decltype((~a)*(1.0/(a*(~a)).template element<0x00>()))
{
   return (~a)*(1.0/(a*(~a)).template element<0x00>());
}*/

template <class A>
inline
    //auto operator!(const gaalet::expression<A>& a) -> decltype(eval(a*gaalet::element_t()))
    auto inverse(const gaalet::expression<A> &a) -> decltype(eval(a *typename A::element_t()))
{
    typename A::element_t div = 1.0 / ((~a) * a).template element<0x00>();
    return eval(a * div);
}

#endif
