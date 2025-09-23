// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

#include <lamure/platform.h>
#include <lamure/types.h>

#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <vector>

namespace lamure {

typedef boost::timer::auto_cpu_timer auto_timer;
typedef boost::timer::cpu_timer cpu_timer;

/**
 * converts the input vector to the vector of pointers.
 * The pointers refer to the base class of input's elements.
 */
template <class Base, class Derived>
std::vector<Base*> vector_to_ptr(std::vector<Derived>& input)
{
    std::vector<Base*> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(),
            [](Derived& i){ return &i; });
    return result;
}

//COMMON_DLL boost::filesystem::path add_to_path(const boost::filesystem::path& path,
//                                  const std::string& addition);

boost::filesystem::path add_to_path(const boost::filesystem::path& path,
    const std::string& addition);

} // namespace lamure

#endif // COMMON_UTILS_H_

