// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "utils.h"
#include <boost/filesystem.hpp>

namespace {

scm::math::mat4f loadMatrix(const std::string& filename)
{

    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Unable to open transformation file: \"" 
            << filename << "\"\n";
        return scm::math::mat4f::identity();
    }
    scm::math::mat4f mat = scm::math::mat4f::identity();
    std::string matrix_values_string;
    std::getline(f, matrix_values_string);
    std::stringstream sstr(matrix_values_string);
    for (int i = 0; i < 16; ++i)
        sstr >> mat[i];
    return scm::math::transpose(mat);
}

}

std::pair< std::vector<std::string>, std::vector<scm::math::mat4f> > 
read_model_string(std::string const& path_to_resource_file, 
                std::set<lamure::model_t>* visible_set, 
                std::set<lamure::model_t>* invisible_set)
{
    namespace fs = boost::filesystem;

    std::pair< std::vector<std::string>, std::vector<scm::math::mat4f> > model_vec_pair;

    auto filename = fs::canonical(path_to_resource_file);


    std::ifstream resource_file(filename.string());

    std::string one_line;

    if(resource_file.is_open() )
    {
        lamure::model_t model_id = 0;

        while(std::getline(resource_file, one_line))
        {


            std::istringstream model_ss(one_line);

            if(one_line.length() >= 2)
            {
                if( !(one_line.substr(0,2) == "//") )
                {
                  std::string model_path;
                  scm::math::mat4f model_transf = scm::math::mat4f::identity();

                  model_ss >> model_path;

                  while (model_ss.rdbuf()->in_avail() != 0)
                  {

                    std::string dollar;

                    model_ss >> dollar;

                    if (dollar.size() == 0) {
                        continue;
                    }

                    if (dollar[0] == '$') {
                        switch (dollar[1]) {
                            case 'v':
                                (*visible_set).insert(model_id);
                                break;

                            case 'i':
                                (*invisible_set).insert(model_id);
                                break;

                            default: break;

                        }
                    }
                    else {
                        auto tranf_filename = fs::absolute(fs::path(dollar), filename.parent_path());
                        model_transf = loadMatrix(tranf_filename.string());
                    }
                  }

                  auto model_filename = fs::canonical(fs::path(model_path), filename.parent_path());

                  model_vec_pair.first.push_back(model_filename.string());
                  model_vec_pair.second.push_back(model_transf);

                  ++model_id;
                }
            }

        }

        resource_file.close();
    }
    else
    {
        std::cerr << "Could not open resource file: \"" 
                  << filename.string() << "\"\n";
    }

    return model_vec_pair;

}


void create_scene_name_from_vector(std::vector<std::string> const& name_vector, std::string& name)
{

    std::string n = "";

    //pattern before and after the real name of the model in the resources file
    std::string pattern_before_substring = "/";
    std::string pattern_after_substring  = ".kdn";

    for(std::vector<std::string>::const_iterator it = name_vector.begin(); it != name_vector.end(); ++it )
    {
        int pos_after_first_pattern= (*it).find_last_of(pattern_before_substring) + 1;
        int substring_length = ( ( (*it).find_last_of(pattern_after_substring) - pattern_after_substring.length() ) + 1 ) - pos_after_first_pattern;

        n += (*it).substr(pos_after_first_pattern , substring_length);

        if(it != (name_vector.end()-1) )
            n += "_&_";
    }

    name = n;
}

void create_scene_name_from_camera_session_file(std::string const& session_file, std::string& name)
{


    //pattern before and after the real name of the model in the resources file
    std::string pattern_before_substring = "/";
    std::string pattern_after_substring  = ".csn";


        int pos_after_first_pattern= session_file.find_last_of(pattern_before_substring) + 1;
        int substring_length = ( ( session_file.find_last_of(pattern_after_substring) - pattern_after_substring.length() ) + 1 ) - pos_after_first_pattern;



    name = (session_file).substr(pos_after_first_pattern , substring_length);
}


