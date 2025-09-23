// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/utils.h>

namespace lamure {

boost::filesystem::path add_to_path(const boost::filesystem::path& path,
                                  const std::string& addition)
{
    return boost::filesystem::path(path) += addition;
}

} // namespace lamure

