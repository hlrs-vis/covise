#include "utils.h"
#include <osg/MatrixTransform>

osg::Matrix localToWorldBelowRoot(const osg::NodePath &path)
{
    osg::Matrix m;
    for (unsigned int i = 1; i < path.size(); ++i) // skip path[0] = root (maps root->root)
    {
        osg::Transform *t = path[i]->asTransform();
        if (t)
            t->computeLocalToWorldMatrix(m, 0); // virtual: handles RELATIVE/ABSOLUTE + per-class matrix
    }
    return m;
}
