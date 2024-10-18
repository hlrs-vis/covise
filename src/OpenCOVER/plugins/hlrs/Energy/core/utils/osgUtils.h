#ifndef _CORE_UTILS_OSGUTILS_H
#define _CORE_UTILS_OSGUTILS_H

#include <osg/Group>

namespace core::utils::osgUtils {
void deleteChildrenRecursive(osg::Group *grp);
}
#endif
