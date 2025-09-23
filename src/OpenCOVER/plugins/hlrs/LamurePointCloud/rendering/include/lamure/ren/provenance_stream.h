// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_PROVENANCE_STREAM_H_
#define REN_PROVENANCE_STREAM_H_

#include <fstream>
#include <vector>
#include <string>
#include <cstdio>

#include <lamure/ren/platform.h>
#include <lamure/utils.h>
#include <lamure/ren/config.h>

namespace lamure {
namespace ren
{

class provenance_stream
{
public:
                        provenance_stream();
                        provenance_stream(const provenance_stream&) = delete;
                        provenance_stream& operator=(const provenance_stream&) = delete;
    virtual             ~provenance_stream();


    void                open(const std::string& file_name);
    void                open_for_writing(const std::string& file_name);
    void                close();
    const bool          is_file_open() const { return is_file_open_; };
    const std::string&  file_name() const { return file_name_; };

    void                read(char* const data,
                            const size_t start_in_file,
                            const size_t length_in_bytes) const;
                            
    // void                write(char* const data,
    //                         const size_t start_in_file,
    //                         const size_t length_in_bytes);

private:
    mutable std::ifstream stream_;

    std::string         file_name_;
    bool                is_file_open_;
};

} } // namespace lamure

#endif // REN_PROVENANCE_STREAM_H_

