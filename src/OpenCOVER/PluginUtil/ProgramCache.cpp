/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ProgramCache.h"
#include <osgDB/fstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

ProgramCache *ProgramCache::instance()
{
    static ProgramCache *singleton = 0;
    if (!singleton)
        singleton = new ProgramCache();
    return singleton;
}

ProgramCache::ProgramCache()
{
}

ProgramCache::~ProgramCache()
{
}

osg::ref_ptr<osg::Program> ProgramCache::getProgram(std::string vertex, std::string fragment)
{
    ProgramMap::iterator progIt = _programMap.find(vertex + ";" + fragment);
    if (progIt != _programMap.end())
    {
        return progIt->second;
    }

    // create shader program
    osg::ref_ptr<osg::Program> program = new osg::Program();
    // sources for shader
    std::stringstream fragSource, vertSource;
    std::string line;
    std::ifstream fileFrag;
    fileFrag.open(fragment.c_str());
    if (fileFrag.is_open())
    {
        while (getline(fileFrag, line))
            fragSource << line << "\n";
    }
    else
        std::cerr << "Error reading fragment file\n";
    fileFrag.close();
    std::ifstream fileVert;
    fileVert.open(vertex.c_str());
    if (fileVert.is_open())
    {
        while (getline(fileVert, line))
            vertSource << line << "\n";
    }
    else
        std::cerr << "Error reading vertex file\n";
    fileVert.close();
    // create frag and vert shader and add them to program
    osg::Shader *frag = new osg::Shader(osg::Shader::FRAGMENT, fragSource.str());
    osg::Shader *vert = new osg::Shader(osg::Shader::VERTEX, vertSource.str());
    program->addShader(frag);
    program->addShader(vert);
    // add program to map
    _programMap.insert(std::pair<std::string, osg::ref_ptr<osg::Program> >(vertex + ";" + fragment, program));
    return program;
}

void ProgramCache::gc()
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        ProgramMap::iterator it = _programMap.begin();
        while (it != _programMap.end())
        {
            if (it->second.get()->referenceCount() == 1)
            {
                _programMap.erase(it);
                changed = true;
                break;
            }
            it++;
        }
    }
}
