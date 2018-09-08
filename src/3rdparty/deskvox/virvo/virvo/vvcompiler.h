// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#ifndef VV_COMPILER_H
#define VV_COMPILER_H


//
// Determine compiler
//

#if defined(__clang__)
#define VV_CXX_CLANG    (100 * __clang_major__ + __clang_minor__)
#elif defined(__INTEL_COMPILER) && defined(__GNUC__)
#define VV_CXX_INTEL    (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(__GNUC__)
#define VV_CXX_GCC      (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(_MSC_VER)
#define VV_CXX_MSVC     (_MSC_VER)
#endif

#ifndef VV_CXX_CLANG
#define VV_CXX_CLANG 0
#endif
#ifndef VV_CXX_INTEL
#define VV_CXX_INTEL 0
#endif
#ifndef VV_CXX_GCC
#define VV_CXX_GCC 0
#endif
#ifndef VV_CXX_MSVC
#define VV_CXX_MSVC 0
#endif

// MinGW on Windows?
#if defined(_WIN32) && (defined(__MINGW32__) || defined(__MINGW64__))
#define VV_CXX_MINGW VV_CXX_GCC
#else
#define VV_CXX_MINGW 0
#endif


//
// Determine available C++11 features
//

// VV_CXX_HAS_ALIAS_TEMPLATES
// VV_CXX_HAS_AUTO
// VV_CXX_HAS_CONSTEXPR
// VV_CXX_HAS_DECLTYPE
// VV_CXX_HAS_EXPLICIT_CONVERSIONS
// VV_CXX_HAS_GENERALIZED_INITIALIZERS
// VV_CXX_HAS_LAMBDAS
// VV_CXX_HAS_NULLPTR
// VV_CXX_HAS_OVERRIDE_CONTROL
// VV_CXX_HAS_RANGE_FOR
// VV_CXX_HAS_RVALUE_REFERENCES
// VV_CXX_HAS_STATIC_ASSERT
// VV_CXX_HAS_VARIADIC_TEMPLATES

#if VV_CXX_CLANG
#   if __has_feature(cxx_alias_templates)
#       define VV_CXX_HAS_ALIAS_TEMPLATES
#   endif
#   if __has_feature(cxx_auto_type)
#       define VV_CXX_HAS_AUTO
#   endif
#   if __has_feature(cxx_constexpr)
#       define VV_CXX_HAS_CONSTEXPR
#   endif
#   if __has_feature(cxx_decltype)
#       define VV_CXX_HAS_DECLTYPE
#   endif
#   if __has_feature(cxx_explicit_conversions)
#       define VV_CXX_HAS_EXPLICIT_CONVERSIONS
#   endif
#   if __has_feature(cxx_generalized_initializers)
#       define VV_CXX_HAS_GENERALIZED_INITIALIZERS
#   endif
#   if __has_feature(cxx_lambdas)
#       define VV_CXX_HAS_LAMBDAS
#   endif
#   if __has_feature(cxx_nullptr)
#       define VV_CXX_HAS_NULLPTR
#   endif
#   if __has_feature(cxx_override_control)
#       define VV_CXX_HAS_OVERRIDE_CONTROL
#   endif
#   if __has_feature(cxx_range_for)
#       define VV_CXX_HAS_RANGE_FOR
#   endif
#   if __has_feature(cxx_rvalue_references)
#       define VV_CXX_HAS_RVALUE_REFERENCES
#   endif
#   if __has_feature(cxx_static_assert)
#       define VV_CXX_HAS_STATIC_ASSERT
#   endif
#   if __has_feature(cxx_variadic_templates)
#       define VV_CXX_HAS_VARIADIC_TEMPLATES
#   endif
#elif VV_CXX_GCC
#   ifdef __GXX_EXPERIMENTAL_CXX0X__
#       if VV_CXX_GCC >= 403
#           define VV_CXX_HAS_DECLTYPE
#           define VV_CXX_HAS_RVALUE_REFERENCES
#           define VV_CXX_HAS_STATIC_ASSERT
#           define VV_CXX_HAS_VARIADIC_TEMPLATES
#       endif
#       if VV_CXX_GCC >= 404
#           define VV_CXX_HAS_AUTO
#           define VV_CXX_HAS_GENERALIZED_INITIALIZERS
#       endif
#       if VV_CXX_GCC >= 405
#           define VV_CXX_HAS_EXPLICIT_CONVERSIONS
#           define VV_CXX_HAS_LAMBDAS
#       endif
#       if VV_CXX_GCC >= 406
#           define VV_CXX_HAS_CONSTEXPR
#           define VV_CXX_HAS_NULLPTR
#           define VV_CXX_HAS_RANGE_FOR
#       endif
#       if VV_CXX_GCC >= 407
#           define VV_CXX_HAS_ALIAS_TEMPLATES
#           define VV_CXX_HAS_OVERRIDE_CONTROL
#       endif
#   endif
#elif VV_CXX_MSVC
#   if VV_CXX_MSVC >= 1600 // Visual C++ 10.0 (2010)
#       define VV_CXX_HAS_AUTO
#       define VV_CXX_HAS_DECLTYPE
#       define VV_CXX_HAS_LAMBDAS
#       define VV_CXX_HAS_NULLPTR
#       define VV_CXX_HAS_RVALUE_REFERENCES
#       define VV_CXX_HAS_STATIC_ASSERT
#   endif
#   if VV_CXX_MSVC >= 1700 // Visual C++ 11.0 (2012)
#       define VV_CXX_HAS_OVERRIDE_CONTROL
#       define VV_CXX_HAS_RANGE_FOR
#   endif
#   if _MSC_FULL_VER == 170051025 // Visual C++ 12.0 November CTP
#       define VV_CXX_HAS_EXPLICIT_CONVERSIONS
#       define VV_CXX_HAS_GENERALIZED_INITIALIZERS
#       define VV_CXX_HAS_VARIADIC_TEMPLATES
#   endif
#endif


#ifdef VV_CXX_HAS_OVERRIDE_CONTROL
#define VV_OVERRIDE override
#define VV_FINAL final
#else
#if VV_CXX_MSVC
#define VV_OVERRIDE override
#else
#define VV_OVERRIDE
#endif
#define VV_FINAL
#endif


//-------------------------------------------------------------------------------------------------
// Detect stdlib features
//

// VV_CXXLIB_HAS_HDR_ARRAY
// VV_CXXLIB_HAS_HDR_CHRONO
// VV_CXXLIB_HAS_HDR_TYPE_TRAITS
// VV_CXXLIB_HAS_COPYSIGN
// VV_CXXLIB_HAS_ROUND
// VV_CXXLIB_HAS_TRUNC

#if defined(_CPPLIB_VER)
#  if (_CPPLIB_VER >= 520) // vc10
#    define VV_CXXLIB_HAS_HDR_ARRAY 1
#    define VV_CXXLIB_HAS_HDR_TYPE_TRAITS 1
#  endif
#  if (_CPPLIB_VER >= 540) // vc11
#    define VV_CXXLIB_HAS_HDR_CHRONO 1
#  endif
#  if (_CPPLIB_VER >= 610) // vc12
#    define VV_CXXLIB_HAS_COPYSIGN 1
#    define VV_CXXLIB_HAS_ROUND 1
#    define VV_CXXLIB_HAS_TRUNC 1
#  endif
#endif

#if defined(__GLIBCXX__)
#  ifdef __GXX_EXPERIMENTAL_CXX0X__
#    define VV_CXXLIB_HAS_HDR_ARRAY 1
#    define VV_CXXLIB_HAS_HDR_CHRONO 1
#    define VV_CXXLIB_HAS_HDR_TYPE_TRAITS 1
#    define VV_CXXLIB_HAS_COPYSIGN 1
#    define VV_CXXLIB_HAS_ROUND 1
#    define MATH_CXXLIB_HAS_TRUNC 1
#  endif
#endif

#if defined(_LIBCPP_VERSION)
#  define VV_CXXLIB_HAS_HDR_ARRAY 1
#  define VV_CXXLIB_HAS_HDR_CHRONO 1
#  define VV_CXXLIB_HAS_HDR_TYPE_TRAITS 1
#  define VV_CXXLIB_HAS_COPYSIGN 1
#  define VV_CXXLIB_HAS_ROUND 1
#  define VV_CXXLIB_HAS_TRUNC 1
#endif


//
// Macros to work with compiler warnings/errors
//

// Use like: VV_CXX_MSVC_WARNING_DISABLE(4996)
// FIXME: Requires vc > ?
#if VV_CXX_MSVC
#   define VV_MSVC_WARNING_SUPPRESS(X) \
        __pragma(warning(suppress : X))
#   define VV_MSVC_WARNING_PUSH_LEVEL(X) \
        __pragma(warning(push, X))
#   define VV_MSVC_WARNING_PUSH() \
        __pragma(warning(push))
#   define VV_MSVC_WARNING_POP() \
        __pragma(warning(pop))
#   define VV_MSVC_WARNING_DEFAULT(X) \
        __pragma(warning(default : X))
#   define VV_MSVC_WARNING_DISABLE(X) \
        __pragma(warning(disable : X))
#   define VV_MSVC_WARNING_ERROR(X) \
        __pragma(warning(error : X))
#   define VV_MSVC_WARNING_PUSH_DISABLE(X) \
        __pragma(warning(push)) \
        __pragma(warning(disable : X))
#else
#   define VV_MSVC_WARNING_SUPPRESS(X)
#   define VV_MSVC_WARNING_PUSH_LEVEL(X)
#   define VV_MSVC_WARNING_PUSH()
#   define VV_MSVC_WARNING_POP()
#   define VV_MSVC_WARNING_DEFAULT(X)
#   define VV_MSVC_WARNING_DISABLE(X)
#   define VV_MSVC_WARNING_ERROR(X)
#   define VV_MSVC_WARNING_PUSH_DISABLE(X)
#endif


// Use like: VV_CXX_GCC_DIAGNOSTIC_IGNORE("-Wuninitialized")
// FIXME: Requires gcc > 4.6 or clang > ?.?
#if VV_CXX_CLANG || VV_CXX_GCC
#   define VV_GCC_DIAGNOSTIC_PUSH() \
        _Pragma("GCC diagnostic push")
#   define VV_GCC_DIAGNOSTIC_POP() \
        _Pragma("GCC diagnostic pop")
#   define VV_GCC_DIAGNOSTIC_IGNORE(X) \
        _Pragma("GCC diagnostic ignored \"" X "\"")
#   define VV_GCC_DIAGNOSTIC_WARNING(X) \
        _Pragma("GCC diagnostic warning \"" X "\"")
#   define VV_GCC_DIAGNOSTIC_ERROR(X) \
        _Pragma("GCC diagnostic error \"" X "\"")
#else
#   define VV_GCC_DIAGNOSTIC_PUSH()
#   define VV_GCC_DIAGNOSTIC_POP()
#   define VV_GCC_DIAGNOSTIC_IGNORE(X)
#   define VV_GCC_DIAGNOSTIC_WARNING(X)
#   define VV_GCC_DIAGNOSTIC_ERROR(X)
#endif


// Macro to mark classes and functions as deprecated. Usage:
//
// struct VV_DEPRECATED X {};
// struct X { void fun(); void VV_DEPRECATED fun(int); }
//
// void VV_DEPRECATED fun();
//
#if VV_CXX_MSVC
#   define VV_DEPRECATED __declspec(deprecated)
#elif VV_CXX_GCC || VV_CXX_CLANG
#   define VV_DEPRECATED __attribute__((deprecated))
#else
#   define VV_DEPRECATED
#endif


#if VV_CXX_MSVC
#   define VV_DLL_EXPORT __declspec(dllexport)
#   define VV_DLL_IMPORT __declspec(dllimport)
#elif VV_CXX_CLANG || VV_CXX_GCC
#   define VV_DLL_EXPORT __attribute__((visibility("default")))
#   define VV_DLL_IMPORT __attribute__((visibility("default")))
#else
#   define VV_DLL_EXPORT
#   define VV_DLL_IMPORT
#endif


#endif
