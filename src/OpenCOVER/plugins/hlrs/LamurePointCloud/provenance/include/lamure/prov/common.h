// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PROV_COMMON_H
#define LAMURE_PROV_COMMON_H

#include <algorithm>
#include <assert.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/crc.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <fstream>
#include <lamure/prov/3rd_party/exif.h>
#include <lamure/prov/3rd_party/pdqsort.h>
#include <lamure/prov/3rd_party/tinyply.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#include <scm/core.h>
#include <scm/core/math.h>
#ifdef WIN32
#define BIG_ENDIAN 1
#define LITTLE_ENDIAN 0
#define BYTE_ORDER 0
#define IS_BIG_ENDIAN (*(WORD *)"\0\x2" != 0x200)
#endif

namespace lamure
{
namespace prov
{
// bool DEBUG = false;

typedef std::ifstream ifstream;
typedef std::ofstream ofstream;
typedef std::string string;

template <typename T, std::size_t SIZE>
using arr = std::array<T, SIZE>;

template <typename T>
using vec = std::vector<T>;

template <typename T>
using u_ptr = std::unique_ptr<T>;

template <typename T>
using s_ptr = std::shared_ptr<T>;

typedef scm::math::vec2f vec2f;
typedef scm::math::vec3f vec3f;
typedef scm::math::mat4f mat4f;
typedef scm::math::quatf quatf;

typedef boost::archive::text_oarchive text_oarchive;
typedef boost::archive::text_iarchive text_iarchive;

using namespace boost::sort::spreadsort;

template <typename T1, typename T2>
using pair = std::pair<T1, T2>;

template <typename T>
T swap(const T &arg, bool big_in_mem)
{
    if((BYTE_ORDER == BIG_ENDIAN) == big_in_mem)
    {
        return arg;
    }
    else
    {
        T ret;

        char *dst = reinterpret_cast<char *>(&ret);
        const char *src = reinterpret_cast<const char *>(&arg + 1);

        for(size_t i = 0; i < sizeof(T); i++)
            *dst++ = *--src;

        return ret;
    }
}

static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

static inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

} // namespace prov
} // namespace lamure
#endif // LAMURE_PROV_COMMON_H