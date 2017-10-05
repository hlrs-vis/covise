#include "StaticSequence.h"

StaticSequence::StaticSequence()
{
    setNumChildrenRequiringUpdateTraversal(getNumChildrenRequiringUpdateTraversal()-1);
}

StaticSequence::StaticSequence(const osg::Sequence &other, const osg::CopyOp &copyop)
: Sequence(other, copyop)
{
    setNumChildrenRequiringUpdateTraversal(getNumChildrenRequiringUpdateTraversal()-1);
}
