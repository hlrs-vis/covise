/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_MULTIVECTOR_ELEMENT_H
#define __GAALET_MULTIVECTOR_ELEMENT_H

#include <limits>

namespace gaalet
{

//default multivector element type
typedef double default_element_t;

template <typename EL, typename ER, bool EL_SIZE = (sizeof(EL) >= sizeof(ER))>
struct element_type_size_compare_traits;
template <typename EL, typename ER>
struct element_type_size_compare_traits<EL, ER, true>
{
    typedef EL element_t;
};
template <typename EL, typename ER>
struct element_type_size_compare_traits<EL, ER, false>
{
    typedef ER element_t;
};

template <typename EL, typename ER, bool EL_INT = std::numeric_limits<EL>::is_signed, bool ER_INT = std::numeric_limits<ER>::is_signed>
struct element_type_signed_traits;

template <typename EL, typename ER>
struct element_type_signed_traits<EL, ER, true, true>
{
    typedef typename element_type_size_compare_traits<EL, ER>::element_t element_t;
};
template <typename EL, typename ER>
struct element_type_signed_traits<EL, ER, false, false>
{
    typedef typename element_type_size_compare_traits<EL, ER>::element_t element_t;
};
template <typename EL, typename ER>
struct element_type_signed_traits<EL, ER, true, false>
{
    typedef EL element_t;
};
template <typename EL, typename ER>
struct element_type_signed_traits<EL, ER, false, true>
{
    typedef ER element_t;
};

template <typename EL, typename ER, bool EL_INT = std::numeric_limits<EL>::is_integer, bool ER_INT = std::numeric_limits<ER>::is_integer>
struct element_type_integer_traits;

template <typename EL, typename ER>
struct element_type_integer_traits<EL, ER, true, true>
{
    typedef typename element_type_signed_traits<EL, ER>::element_t element_t;
};
template <typename EL, typename ER>
struct element_type_integer_traits<EL, ER, false, false>
{
    typedef typename element_type_signed_traits<EL, ER>::element_t element_t;
};
template <typename EL, typename ER>
struct element_type_integer_traits<EL, ER, false, true>
{
    typedef EL element_t;
};
template <typename EL, typename ER>
struct element_type_integer_traits<EL, ER, true, false>
{
    typedef ER element_t;
};

template <typename EL, typename ER>
struct element_type_combination_traits
{
    typedef typename element_type_integer_traits<EL, ER>::element_t element_t;
};
template <typename E>
struct element_type_combination_traits<E, E>
{
    typedef E element_t;
};

//null element traits
template <typename E, bool is_integer>
struct null_element_type_traits
{
    static constexpr E value = 0;
};
template <typename E>
struct null_element_type_traits<E, false>
{
    static constexpr E value = 0.0;
};

template <typename E>
struct null_element
{
    static constexpr E value()
    {
        return null_element_type_traits<E, std::numeric_limits<E>::is_integer>::value;
    }
};

} //end namespace gaalet

#endif
