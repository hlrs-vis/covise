// This code is derived from the SafeInt library (http://safeint.codeplex.com/)
// Original license follows:

/*
SafeInt.hpp
Version 3.0.17p

This software is licensed under the Microsoft Public License (Ms-PL).
For more information about Microsoft open source licenses, refer to
http://www.microsoft.com/opensource/licenses.mspx

This license governs use of the accompanying software. If you use the software,
you accept this license. If you do not accept the license, do not use the
software.

1. Definitions

The terms "reproduce," "reproduction," "derivative works," and "distribution"
have the same meaning here as under U.S. copyright law. A "contribution" is the
original software, or any additions or changes to the software. A "contributor"
is any person that distributes its contribution under this license. "Licensed
patents" are a contributor's patent claims that read directly on its
contribution.

2. Grant of Rights

(A) Copyright Grant- Subject to the terms of this license, including the license
    conditions and limitations in section 3, each contributor grants you a
    non-exclusive, worldwide, royalty-free copyright license to reproduce its
    contribution, prepare derivative works of its contribution, and distribute
    its contribution or any derivative works that you create.
(B) Patent Grant- Subject to the terms of this license, including the license
    conditions and limitations in section 3, each contributor grants you a
    non-exclusive, worldwide, royalty-free license under its licensed patents to
    make, have made, use, sell, offer for sale, import, and/or otherwise dispose
    of its contribution in the software or derivative works of the contribution
    in the software.

3. Conditions and Limitations

(A) No Trademark License- This license does not grant you rights to use any
    contributors' name, logo, or trademarks.
(B) If you bring a patent claim against any contributor over patents that you
    claim are infringed by the software, your patent license from such
    contributor to the software ends automatically.
(C) If you distribute any portion of the software, you must retain all
    copyright, patent, trademark, and attribution notices that are present in
    the software.
(D) If you distribute any portion of the software in source code form, you may
    do so only under this license by including a complete copy of this license
    with your distribution. If you distribute any portion of the software in
    compiled or object code form, you may only do so under a license that
    complies with this license.
(E) The software is licensed "as-is." You bear the risk of using it. The
    contributors give no express warranties, guarantees or conditions. You may
    have additional consumer rights under your local laws which this license
    cannot change. To the extent permitted under your local laws, the
    contributors exclude the implied warranties of merchantability, fitness for
    a particular purpose and non-infringement.

Copyright (c) Microsoft Corporation.  All rights reserved.
*/

#ifndef VIRVO_PRIVATE_SAFE_INT_CAST_H
#define VIRVO_PRIVATE_SAFE_INT_CAST_H

#include <cassert>
#include <limits>
#include <stdexcept>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/make_signed.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/remove_cv.hpp>

namespace virvo
{

namespace details
{

    enum CastMethod {
        CastOK,
        CastCheckGEZero,
        CastCheckLEMax,
        CastCheckMinMaxUnsigned,
        CastCheckMinMaxSigned
    };

    template <class T/*To*/, class F/*From*/>
    struct GetCastMethod
    {
        static const bool T_is_signed = std::numeric_limits<T>::is_signed;
        static const bool F_is_signed = std::numeric_limits<F>::is_signed;

        static const CastMethod Method
            = ((boost::is_same<F, bool>::value || boost::is_same<T, bool>::value)
                    || (T_is_signed == F_is_signed && sizeof(T) >= sizeof(F))
                    || (T_is_signed && sizeof(T) > sizeof(F)))
                ? CastOK
            : ((T_is_signed && !F_is_signed && sizeof(F) >= sizeof(T))
                    || (!T_is_signed && !F_is_signed && sizeof(F) > sizeof(T)))
                ? CastCheckLEMax
            : (!T_is_signed && F_is_signed && sizeof(T) >= sizeof(F))
                ? CastCheckGEZero
            : (!T_is_signed)
                ? CastCheckMinMaxUnsigned
                : CastCheckMinMaxSigned;
    };

    template <class T>
    struct GetCastMethod<T, T>
    {
        static const CastMethod Method = CastOK;
    };

    template <class T, class F, CastMethod = GetCastMethod<T, F>::Method>
    struct CheckCastImpl;

    template <class T, class F>
    struct CheckCastImpl<T, F, CastOK>
    {
        static bool test(F /*f*/)
        {
            return true;
        }
    };

    template <class T, class F>
    struct CheckCastImpl<T, F, CastCheckGEZero>
    {
        static bool test(F f) // F is signed, T is unsigned
        {
            return 0 <= f;
        }
    };

    template <class T, class F>
    struct CheckCastImpl<T, F, CastCheckLEMax>
    {
        static bool test(F f) // F is unsigned
        {
            return f <= static_cast<F>(std::numeric_limits<T>::max());
        }
    };

    template <class T, class F>
    struct CheckCastImpl<T, F, CastCheckMinMaxUnsigned>
    {
        static bool test(F f) // F is signed, T is unsigned
        {
            return 0 <= f && f <= std::numeric_limits<T>::max();
        }
    };

    template <class T, class F>
    struct CheckCastImpl<T, F, CastCheckMinMaxSigned>
    {
        static bool test(F f) // F, T are signed
        {
            return f >= std::numeric_limits<T>::min()
                && f <= std::numeric_limits<T>::max();
        }
    };

    template <class T, class F>
    struct CheckCast
        : CheckCastImpl<typename boost::remove_cv<T>::type,
                        typename boost::remove_cv<F>::type>
    {
    };

    template <class T, class F>
    inline bool IsSafeIntCastable(F f)
    {
        BOOST_STATIC_ASSERT_MSG(boost::is_integral<F>::value,
            "F must be an integral type");
        BOOST_STATIC_ASSERT_MSG(boost::is_integral<T>::value,
            "T must be an integral type");

        return CheckCast<T, F>::test(f);
    }

} // namespace details

//------------------------------------------------------------------------------
// BadIntCast
//
// Exception thrown on invalid integer-to-integer casts
//
class BadIntCast
    : public std::runtime_error
{
public:
    BadIntCast()
        : std::runtime_error("bad int cast")
    {
    }
};

//------------------------------------------------------------------------------
// checked_int_cast
//
// Asserts that casting f from type F to type T is valid.
// Returns static_cast<T>(f).
//
template <class T, class F>
inline T checked_int_cast(F f)
{
    assert(details::IsSafeIntCastable<T>(f) && "bad int cast");
    return static_cast<T>(f);
}

//------------------------------------------------------------------------------
// safe_int_cast
//
// Like checked_int_cast, but throws BadIntCast if the cast is invalid.
//
template <class T, class F>
inline T safe_int_cast(F f)
{
    if (details::IsSafeIntCastable<T>(f))
        return static_cast<T>(f);

    throw BadIntCast();
}

//------------------------------------------------------------------------------
// as_signed_int
//
// Casts f from signed or unsigned type F to the signed type of the same size.
// Throws BadIntCast if the cast is invalid.
//
// This might be useful to safely cast from size_t to ssize_t.
//
template <class F>
inline typename boost::make_signed<F>::type as_signed_int(F f)
{
    return safe_int_cast<typename boost::make_signed<F>::type>(f);
}

//------------------------------------------------------------------------------
// as_unsigned_int
//
// Casts f from signed or unsigned type F to the unsigned type of the same size.
// Throws BadIntCast if the cast is invalid.
//
// This might be usefule to safely cast from ssize_t to size_t.
//
template <class F>
inline typename boost::make_unsigned<F>::type as_unsigned_int(F f)
{
    return safe_int_cast<typename boost::make_unsigned<F>::type>(f);
}

namespace details
{
    template <class F>
    struct ImplicitIntCast
    {
        F f;

        explicit ImplicitIntCast(F f) : f(f)
        {
        }

        template <class T>
        operator T() const
        {
            BOOST_STATIC_ASSERT_MSG(boost::is_integral<T>::value,
                "T must be an integral type");

            return safe_int_cast<T>(f); // return checked_int_cast<T>(f);
        }
    };
}

//------------------------------------------------------------------------------
// implicit_int_cast
//
// Returns an object which is implicitly convertible to an integral type.
// Asserts that the cast is valid when the conversion operator is invoked.
//
// This might be useful in cases where the target type is tedious to write down.
//
template <class F>
inline details::ImplicitIntCast<F> implicit_int_cast(F f)
{
    return details::ImplicitIntCast<F>(f);
}

} // namespace virvo

#endif // !VIRVO_PRIVATE_SAFE_INT_CAST_H
