/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROGRAM_CACHE_H
#define PROGRAM_CACHE_H

#include <util/common.h>

#include <map>

#include <osg/Program>

typedef std::map<std::string, osg::ref_ptr<osg::Program> > ProgramMap;

class PLUGIN_UTILEXPORT ProgramCache
{
public:
    static ProgramCache *instance();
    virtual ~ProgramCache();

    osg::ref_ptr<osg::Program> getProgram(std::string vertex, std::string fragment);
    void gc();

private:
    ProgramCache();

    ProgramMap _programMap;
};

#endif
