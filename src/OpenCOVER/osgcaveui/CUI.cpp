/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <iostream>
#include <assert.h>

// OSG:
#include <osg/MatrixTransform>

// Local:
#include "CUI.h"

using namespace cui;
using namespace osg;

CUI::DisplayType CUI::_display = CUI::CAVE;

/** Compute transformation matrix from local node to root node.
 *   @return true and matrix in local2root; or false if root node.
 */
Matrix CUI::computeLocal2Root(const Node *local)
{
    Matrix local2root;

    // Get locl transformation matrix, if any:
    const MatrixTransform *matrixTransform = dynamic_cast<const MatrixTransform *>(local);
    if (matrixTransform)
        local2root = matrixTransform->getMatrix();

    // Recurse to root:
    if (local->getNumParents() > 0)
    {
        const Group *parent = local->getParent(0);
        assert(parent);
        Matrix parent2root = computeLocal2Root(parent);
        local2root = local2root * parent2root;
    }

    return local2root;
}

/** @return true if child is a child of parent. This works over
 *   multiple levels of inheritance!
 *   */
bool CUI::isChild(Node *child, Node *parent)
{
    Node *testParent;
    unsigned int i;

    for (i = 0; i < child->getNumParents(); ++i)
    {
        testParent = child->getParent(i);
        if (testParent == parent)
            return true;
        else
            return isChild(testParent, parent);
    }
    return false;
}
