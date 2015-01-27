/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_LOGARITHM_H
#define __GAALET_LOGARITHM_H

#include <cmath>

#include "grade.h"
#include "geometric_product.h"
#include "inverse.h"
#include "magnitude.h"

namespace gaalet
{

//check for spinor
template <typename CL>
struct check_spinor
{
    static const bool value = (BitCount<CL::head>::value == 2 || BitCount<CL::head>::value == 0)
                                  ? check_spinor<typename CL::tail>::value
                                  : false;
};
template <>
struct check_spinor<cl_null>
{
    static const bool value = true;
};

//go through inversion evaluation type checks
//value=1 - bivector logarithm
//value=0 - scalar logarithm
//value=2 - spinor logarithm
template <class A>
struct logarithm_evaluation_type
{
    static const int value = (check_bivector<typename A::clist>::value) ? 1 : (check_scalar<typename A::clist>::value) ? 0 : (check_spinor<typename A::clist>::value) ? 2 : -1;
};

template <class A, int ET = logarithm_evaluation_type<A>::value>
struct logarithm : public expression<logarithm<A> >
{
    static_assert(ET != -1, "no method for evaluating this type of multivector implemented");
};

//bivector logarithm
template <class A>
struct logarithm<A, 1> : public expression<logarithm<A> >
{
    typedef typename insert_element<0, typename A::clist>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    //dangerous implementation: constructor only called when expression is defined, not when evaluated
    logarithm(const A &a_)
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
            auto b = eval(::grade<2>(a));
            typedef decltype(b) b_type;
            element_t b_square = 0.0;
            for (int bIt = 0; bIt < b_type::size; ++bIt)
            {
                b_square += b[bIt] * b[bIt];
            }
            inv_mag_b = 1.0 / sqrt(b_square);
            element_t r = a.element<0>();
            mag_s = sqrt(r * r + b_square);
            first_eval = false;
        }

        return conf == 0x00 ? log(mag_s) : a.element<conf>() * inv_mag_b * acos(0);
    }
    /*template<>
   element_t element<0>() const {
      return ca;
   }*/

protected:
    const A &a;
    mutable element_t mag_s;
    mutable element_t inv_mag_b;
    mutable bool first_eval;
};

//scalar logarithm
template <class A>
struct logarithm<A, 0> : public expression<logarithm<A> >
{
    typedef typename insert_element<0, cl_null>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    logarithm(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return (conf == 0) ? log(a.element<conf>()) : 0.0;
    }

protected:
    const A &a;
};

//spinor logarithm
template <class A>
struct logarithm<A, 2> : public expression<logarithm<A> >
{
    typedef typename insert_element<0, typename A::clist>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    //dangerous implementation: constructor only called when expression is defined, not when evaluated
    logarithm(const A &a_)
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
            //element_t b_square = eval(::magnitude(::grade<2>(a)));
            auto b = eval(::grade<2>(a));
            typedef decltype(b) b_type;
            element_t b_square = 0.0;
            for (unsigned int bIt = 0; bIt < b_type::size; ++bIt)
            {
                b_square += b[bIt] * b[bIt];
            }
            element_t r = a.element<0>();
            mag_s = sqrt(r * r + b_square);
            b_acos_r_s = acos(r / mag_s) / sqrt(b_square);
            first_eval = false;
        }

        return conf == 0x00 ? log(mag_s) : a.element<conf>() * b_acos_r_s;
    }
    /*template<>
   element_t element<0>() const {
      return ca;
   }*/

protected:
    const A &a;
    mutable element_t mag_s;
    mutable element_t b_acos_r_s;
    mutable bool first_eval;
};

} //end namespace gaalet

/// Logarithm of a multivector.
/**
 * Only implemented for scalars, bivectors and spinors.
 */
/// \ingroup ga_ops
template <class A>
inline gaalet::logarithm<A>
log(const gaalet::expression<A> &a)
{
    return gaalet::logarithm<A>(a);
}

#endif
