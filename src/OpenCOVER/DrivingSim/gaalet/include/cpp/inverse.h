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
//Strange error with CUDA 3.2 and 4.0 for "BitCount<CL::head>" -> workaround "BitCount<CL::head+0>"
template <typename CL, bool parent_even_grade = (BitCount<CL::head + 0>::value % 2)>
struct check_versor_op
{
    static const bool even_grade = (BitCount<CL::head + 0>::value % 2);
    static const bool value = (parent_even_grade == even_grade) ? check_versor_op<typename CL::tail, even_grade>::value : false;
};
template <bool parent_even_grade>
struct check_versor_op<cl_null, parent_even_grade>
{
    static const bool even_grade = parent_even_grade;
    static const bool value = true;
};
template <typename CL>
struct check_versor
{
    static const bool parent_even_grade = (BitCount<CL::head + 0>::value % 2);
    static const bool value = check_versor_op<CL, parent_even_grade>::value;
};

//go through inversion evaluation type checks
//value=1 - versor inversion
template <class A>
struct inverse_evaluation_type
{
    static const bool is_versor = check_versor<typename A::clist>::value;
    static const int value = (is_versor) ? 1 : -1;
};

template <class A, int ET = inverse_evaluation_type<A>::value>
struct inverse : public expression<inverse<A> >
{
    //C++0x only: static_assert(ET!=-1, "no method for evaluating this type of multivector implemented");
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

    //review: don't evaluate on definition workaround: will only work if arguments stay the same (thus attention with variables)
    template <conf_t conf>
    GAALET_CUDA_HOST_DEVICE
        element_t
            element() const
    {
        if (first_eval)
        {
            //div = 1.0/((~a)*a).template element<0x00>();
            element_t inv_div = eval(grade<0>((~a) * a));
            div = 1.0 / inv_div;
            first_eval = false;
        }
        return a.template element<conf>() * div * Power<-1, BitCount<conf>::value *(BitCount<conf>::value - 1) / 2>::value;
    }

protected:
    const A &a;
    mutable element_t div;
    mutable bool first_eval;
};

} //end namespace gaalet

template <class A>
inline GAALET_CUDA_HOST_DEVICE
    gaalet::inverse<A>
    operator!(const gaalet::expression<A> &a)
{
    return gaalet::inverse<A>(a);
}

/*template <class A> inline
auto operator!(const gaalet::expression<A>& a) -> decltype((~a)*(1.0/(a*(~a)).template element<0x00>()))
{
   return (~a)*(1.0/(a*(~a)).template element<0x00>());
}*/

/*C++0x only: template <class A> inline
//auto operator!(const gaalet::expression<A>& a) -> decltype(eval(a*gaalet::element_t()))
auto inverse(const gaalet::expression<A>& a) -> decltype(eval(a*gaalet::element_t()))
{
   gaalet::element_t div = 1.0/((~a)*a).template element<0x00>();
   return eval(a*div);
}*/

#endif
