/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UNORDERERD_SET_H
#define UNORDERERD_SET_H

// for checking whether we use libc++
#include <ciso646>
#if defined(_LIBCPP_VERSION) || (__cplusplus >= 201103L)
#include <unordered_set>
#include <unordered_map>
using std::unordered_set;
using std::unordered_map;
#define HASH_NAMESPACE std

#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#include <tr1/unordered_set>
#include <tr1/unordered_map>
using std::tr1::unordered_set;
using std::tr1::unordered_map;
#define HASH_NAMESPACE std
#define HASH_NAMESPACE2 tr1

#elif((defined(__linux__) && !defined(CO_ia64icc) && !defined(CO_linux)) || defined(__APPLE__) || (defined(__linux__) && defined(CO_ia64icc) && (__GNUC__ >= 4)))
#include <ext/hash_map>
#include <ext/hash_set>
#if (defined(CO_ia64_glibc22) || (defined(CO_ia64icc) && (__GNUC__ < 4)))
#define unordered_map std::hash_map
#define unordered_set std::hash_set
#define HASH_NAMESPACE std
#else
#define unordered_map __gnu_cxx::hash_map
#define unordered_set __gnu_cxx::hash_set
#define HASH_NAMESPACE __gnu_cxx
#endif
#elif defined(_WIN32)
#if _MSC_VER < 1900
#include <hash_map>
#define unordered_map stdext::hash_map
#include <hash_set>
#define unordered_set stdext::hash_set
#else
#include <unordered_set>
#include <unordered_map>
using std::unordered_set;
using std::unordered_map;
#endif
#define HASH_NAMESPACE std
#else
#include <hash_map>
#define unordered_map std::hash_map
#include <hash_set>
#define unordered_set std::hash_set
#define HASH_NAMESPACE stdext
#endif

#endif
