/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_EXPONENTIAL_H
#define __GAALET_EXPONENTIAL_H

#include <cmath>

#include "grade.h"
#include "geometric_product.h"

namespace gaalet
{

//check for bivector
template <typename CL>
struct check_bivector
{
    //Strange error with CUDA 3.2 and 4.0 for "BitCount<CL::head>" -> workaround "BitCount<CL::head+0>"
    static const bool value = (BitCount<CL::head + 0>::value == 2) ? check_bivector<typename CL::tail>::value : false;
};
template <>
struct check_bivector<cl_null>
{
    static const bool value = true;
};

//go through inversion evaluation type checks
//value=1 - bivector exponential
template <class A>
struct exponential_evaluation_type
{
    static const int value = (check_bivector<typename A::clist>::value) ? 1 : -1;
};

template <class A, int ET = exponential_evaluation_type<A>::value>
struct exponential : public expression<exponential<A> >
{
    //C++0x only: static_assert(ET!=-1, "no method for evaluating this type of multivector implemented");
};

//bivector exponential
template <class A>
struct exponential<A, 1> : public expression<exponential<A> >
{
    typedef typename insert_element<0, typename A::clist>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    //dangerous implementation: constructor only called when expression is defined, not when evaluated
    exponential(const A &a_)
        : a(a_)
        , first_eval(true)
    {
    }

    //review: don't evaluate on definition workaround: will only work if arguments stay the same (thus attention with variables)
    template <conf_t conf>
    element_t element() const
    {
        if (first_eval)
        {
            element_t alpha_square = eval(grade<0>(a * a));
            if (alpha_square < 0.0)
            {
                element_t alpha = sqrt(-alpha_square);
                ca = cos(alpha);
                sada = sin(alpha) / alpha;
            }
            else if (alpha_square == 0.0 || alpha_square == -0.0)
            {
                ca = 1.0;
                sada = 1.0;
            }
            //else if(alpha_square > 0.0) {
            else
            {
                element_t alpha = sqrt(alpha_square);
                ca = cosh(alpha);
                sada = sinh(alpha) / alpha;
            }
            first_eval = false;
        }

        if (conf != 0)
            return a.element<conf>() * sada;
        else
            return ca;
    }
    /*template<>
   element_t element<0>() const {
      return ca;
   }*/

protected:
    const A &a;
    mutable element_t ca;
    mutable element_t sada;
    mutable bool first_eval;
};

} //end namespace gaalet

template <class A>
inline gaalet::exponential<A>
exp(const gaalet::expression<A> &a)
{
    return gaalet::exponential<A>(a);
}

#endif
