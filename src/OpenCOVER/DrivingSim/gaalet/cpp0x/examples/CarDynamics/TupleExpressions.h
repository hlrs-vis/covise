/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TUPLE_EXPRESSIONS_H
#define __TUPLE_EXPRESSIONS_H

#include <tuple>

//Wrapper class for CRTP
template <class E>
struct tuple_expression
{
    operator const E &() const
    {
        return *static_cast<const E *>(this);
    }
};

template <class T>
struct tuple_wrapper : public tuple_expression<tuple_wrapper<T> >
{
    typedef T result_t;

    tuple_wrapper(const T &t_)
        : t(t_)
    {
    }

    template <int I>
    //auto element() -> decltype(std::get<I>(T())) const {
    const typename std::tuple_element<I, T>::type &element() const
    {
        return std::get<I>(t);
    }

    const T &t;
};

template <typename L, typename R>
struct tuple_addition : public tuple_expression<tuple_addition<L, R> >
{
    typedef typename L::result_t result_t;

    tuple_addition(const L &l_, const R &r_)
        : l(l_)
        , r(r_)
    {
    }

    template <int I>
    //auto element() -> decltype(L().template element<I>() + R().template element<I>()) const {
    typename std::tuple_element<I, result_t>::type element() const
    {
        return l.template element<I>() + r.template element<I>();
    }

    const L &l;
    const R &r;
};

template <typename L>
struct tuple_scalar_product : public tuple_expression<tuple_scalar_product<L> >
{
    typedef typename L::result_t result_t;

    tuple_scalar_product(const L &l_, const double &r_)
        : l(l_)
        , r(r_)
    {
    }

    template <int I>
    //auto element() -> decltype(L().template element<I>()*double()) const {
    typename std::tuple_element<I, result_t>::type element() const
    {
        return l.template element<I>() * r;
    }

    const L &l;
    double r;
};

template <class... A>
inline tuple_wrapper<std::tuple<A...> >
wrap(const std::tuple<A...> &t)
{
    return tuple_wrapper<std::tuple<A...> >(t);
}

template <class L, class R>
inline tuple_addition<L, R>
operator+(const tuple_expression<L> &l, const tuple_expression<R> &r)
{
    return tuple_addition<L, R>(l, r);
}

template <class T>
inline tuple_scalar_product<T>
operator*(const tuple_expression<T> &l, const double &r)
{
    return tuple_scalar_product<T>(l, r);
}

template <class T>
inline tuple_scalar_product<T>
operator*(const double &l, const tuple_expression<T> &r)
{
    return tuple_scalar_product<T>(r, l);
}

template <typename E, typename R, int I = std::tuple_size<R>::value - 1>
struct tuple_expression_evaluation
{
    inline static void operate(const E &e, R &r)
    {
        tuple_expression_evaluation<E, R, I - 1>::operate(e, r);
        std::get<I>(r) = e.template element<I>();
    }
};

template <typename E, typename R>
struct tuple_expression_evaluation<E, R, 0>
{
    inline static void operate(const E &e, R &r)
    {
        std::get<0>(r) = e.template element<0>();
    }
};

template <typename R, typename E>
inline R eval(const tuple_expression<E> &e_)
{
    R r;
    const E &e(e_);
    tuple_expression_evaluation<E, R>::operate(e, r);
    return move(r);
}

#endif
