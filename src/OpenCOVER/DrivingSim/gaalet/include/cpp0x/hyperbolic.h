/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_HYPERBOLIC_H
#define __GAALET_HYPERBOLIC_H

#include <cmath>

#include "grade.h"
#include "geometric_product.h"

namespace gaalet
{

//go through inversion evaluation type checks
//value=1 - bivector hyperbolic
//value=2 - scalar hyperbolic
template <class A>
struct hyperbolic_evaluation_type
{
    static const int value = (check_bivector<typename A::clist>::value) ? 1 : (check_scalar<typename A::clist>::value) ? 0 : -1;
};

template <class A, int ET = hyperbolic_evaluation_type<A>::value>
struct sinh : public expression<sinh<A> >
{
    static_assert(ET != -1, "no method for evaluating this type of multivector implemented");
};

//bivector sinh
template <class A>
struct sinh<A, 1> : public expression<sinh<A> >
{
    typedef typename A::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    //dangerous implementation: constructor only called when expression is defined, not when evaluated
    sinh(const A &a_)
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
            element_t alpha_square = eval(grade<0, decltype(a * a)>(a * a));
            if (alpha_square == 0.0 || alpha_square == -0.0)
            {
                sada = 0.0;
            }
            else
            {
                element_t alpha = sqrt(-alpha_square);
                sada = sin(alpha) / alpha;
            }
            first_eval = false;
        }

        return a.element<conf>() * sada;
    }

protected:
    const A &a;
    mutable element_t sada;
    mutable bool first_eval;
};

//scalar sinh
template <class A>
struct sinh<A, 0> : public expression<sinh<A> >
{
    typedef typename insert_element<0, cl_null>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    sinh(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return (conf == 0) ? sinh(a.element<conf>()) : 0.0;
    }

protected:
    const A &a;
};

template <class A, int ET = hyperbolic_evaluation_type<A>::value>
struct cosh : public expression<cosh<A> >
{
    static_assert(ET != -1, "no method for evaluating this type of multivector implemented");
};

//bivector cosh
template <class A>
struct cosh<A, 1> : public expression<cosh<A> >
{
    typedef typename insert_element<0, cl_null>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    //dangerous implementation: constructor only called when expression is defined, not when evaluated
    cosh(const A &a_)
        : a(a_)
    {
    }

    //review: don't evaluate on definition workaround: will only work if arguments stay the same (thus attention with variables)
    template <conf_t conf>
    element_t element() const
    {
        element_t alpha_square = eval(grade<0, decltype(a * a)>(a * a));
        element_t alpha = sqrt(-alpha_square);
        return (conf == 0) ? cos(alpha) : 0.0;
    }

protected:
    const A &a;
};

//scalar cosh
template <class A>
struct cosh<A, 0> : public expression<cosh<A> >
{
    typedef typename insert_element<0, cl_null>::clist clist;

    typedef typename A::metric metric;

    typedef typename A::element_t element_t;

    cosh(const A &a_)
        : a(a_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return (conf == 0) ? cosh(a.element<conf>()) : 0.0;
    }

protected:
    const A &a;
};

} //end namespace gaalet

/// Hyperbolic sine of a multivector.
/**
 * Only implemented for scalars and bivectors.
 */
/// \ingroup ga_ops
template <class A>
inline gaalet::sinh<A>
sinh(const gaalet::expression<A> &a)
{
    return gaalet::sinh<A>(a);
}

/// Hyperbolic cosine of a multivector.
/**
 * Only implemented for scalars and bivectors.
 */
/// \ingroup ga_ops
template <class A>
inline gaalet::cosh<A>
cosh(const gaalet::expression<A> &a)
{
    return gaalet::cosh<A>(a);
}

#endif
