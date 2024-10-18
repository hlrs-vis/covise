#include "osgUtils.h"

namespace core::utils::osgUtils {
void deleteChildrenRecursive(osg::Group *grp) {
  if (!grp)
    return;

  for (int i = 0; i < grp->getNumChildren(); ++i) {
    auto child = grp->getChild(i);
    if (auto child_group = dynamic_cast<osg::Group *>(child))
      deleteChildrenRecursive(child_group);
    grp->removeChild(child);
  }
}
} // namespace core::utils::osgUtils
