#ifndef PLUGINUTIL_STATIC_SEQUENCE_H
#define PLUGINUTIL_STATIC_SEQUENCE_H

#include <osg/Sequence>

#include <util/coExport.h>

class PLUGIN_UTILEXPORT StaticSequence: public osg::Sequence
{
 public:
   StaticSequence();
   StaticSequence(const osg::Sequence&, const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
};

#endif
