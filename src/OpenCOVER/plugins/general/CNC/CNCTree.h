/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CNCTREE_H
#define CNCTREE_H
 /****************************************************************************\
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
 **    Visualises path and workpiece of CNC machining                        **
 **                                                                          **
 ** Author: A.Kaiser		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** April-23  v2	    				       		                         **
 **                                                                          **
 **                                                                          **
 \****************************************************************************/

#include <stdexcept>

// Used to hold details of a point
struct Point {
    int x;
    int y;
    Point(int _x, int _y)
    {
        x = _x;
        y = _y;
    }
    Point()
    {
        x = 0;
        y = 0;
    }
};

// The main tree class
class TreeNode {
    // Hold details of the boundary of this node
    Point topLeft;
    Point botRight;

    // Contains details of node
    double z;
    int primitivePos;
    vector<bool> sideWalls;
    bool valid;         //if node is still valid

    // Children of this tree
    TreeNode* parentTree;
    int level;
    int numDescendants;
    vector<TreeNode*> childTrees;
    vector<int> millTimesteps;      //the timesteps where this Node is milled
    
public:
    TreeNode(Point _topL, Point _botR, int _level)
    {
        level = _level;
        topLeft = _topL;
        botRight = _botR;
        valid = true;
    }
    TreeNode(Point _topL, Point _botR, int _level, double _z, TreeNode* _parentTree)
    {
        z = _z;
        level = _level;
        topLeft = _topL;
        botRight = _botR;
        parentTree = _parentTree;
        valid = true;
    }
    TreeNode(TreeNode* _tree1, TreeNode* _tree2)
    {
        topLeft = Point(std::min(_tree1->topLeft.x, _tree2->topLeft.x), std::min(_tree1->topLeft.y, _tree2->topLeft.y));
        botRight = Point(std::max(_tree1->botRight.x, _tree2->botRight.x), std::max(_tree1->botRight.y, _tree2->botRight.y));
        valid = true;
    }

    vector<TreeNode*> getChildTrees();
    vector<int> getMillTimesteps();
    Point getTopLeft();
    Point getBotRight();
    int getLevel();
    bool isValid();
    vector<bool> getSideWalls();
    int getPrimitivePos();
    void setPrimitivePos(int _primitivePos);
    bool inBoundary(Point);
    bool unitArea();
    bool areVectorsEqual(std::vector<int> vec1, std::vector<int> vec2);

    void millQuad(int _ix, int _iy, double _z, int t);
    TreeNode* insert(int _ix, int _iy);
    void addChildren();
    TreeNode* search(Point);
    TreeNode* searchCommonParent(TreeNode*, TreeNode*);
    vector<TreeNode*> searchAncestry(TreeNode*, TreeNode*);
    vector<TreeNode*> searchAllNeighbors(TreeNode*, int);


    bool compareForAllCombine();
    void combineAllSiblings();
    bool combine2Siblings(TreeNode*, TreeNode*);
    bool combine2TreeNodes(TreeNode*, TreeNode*);
    bool combineMultiTreeNodes(vector<TreeNode*>);
    bool traverseAndCombineMillQuads();
    bool traverseAndCombineAll();


    bool reorganise3generations(TreeNode*, TreeNode*, TreeNode*, TreeNode*, TreeNode*);
    void rebaseChildrenToParent();
    void rebaseChildrenToParent(TreeNode*);
    bool borderCheck(TreeNode*, TreeNode*);
    void eliminateSingularTrees();
    void eliminateSingleMillTrees();
    TreeNode* searchForInvalid();
    void setLevels();
    void setDescendants();
    void sortChildren();
    void setSideWalls();
    vector<vector<TreeNode*>> writeTimestepVector(vector<vector<TreeNode*>>);
};

inline vector<TreeNode*> TreeNode::getChildTrees()
{
    return childTrees;
}
inline vector<int> TreeNode::getMillTimesteps()
{
    return millTimesteps;
}

inline Point TreeNode::getTopLeft()
{
    return topLeft;
}
inline Point TreeNode::getBotRight()
{
    return botRight;
}
inline int TreeNode::getLevel()
{
    return level;
}
inline bool TreeNode::isValid()
{
    return valid;
}
inline vector<bool> TreeNode::getSideWalls()
{
    return sideWalls;
}
inline int TreeNode::getPrimitivePos()
{
    return primitivePos;
}
inline void TreeNode::setPrimitivePos(int _primitivePos)
{
    primitivePos = _primitivePos;
    return;
}

// Check if current quadtree contains the point
inline bool TreeNode::inBoundary(Point p)
{
    return (p.x > topLeft.x && p.x <= botRight.x
        && p.y > topLeft.y && p.y <= botRight.y);
}

// Check if current we are at a quad of unit area
// We cannot subdivide this quad further
inline bool TreeNode::unitArea()
{
    return (topLeft.x + 1 == botRight.x
        && topLeft.y + 1 == botRight.y);
}

// Check if two vectors contain identical elements
inline bool TreeNode::areVectorsEqual(std::vector<int> vec1, std::vector<int> vec2) {

    // Check if the vectors have the same size
    if (vec1.size() != vec2.size()) {
        return false;
    }
    // Compare the elements of the vectors
    for (int i = 0; i < vec1.size(); i++) {
        if (vec1[i] != vec2[i]) {
            return false;
        }
    }
    return true;
}

// Timestep t is added to millTimesteps for Quad _ix,_iy if _z < z
inline void TreeNode::millQuad(int _ix, int _iy, double _z, int t)
{
    TreeNode* tree = search(Point(_ix, _iy));
    if (_z < tree->z)
    {
        tree = tree->insert(_ix, _iy);
        tree->z = _z;
        tree->millTimesteps.push_back(t);
        //tree->combineQuads();

        bool co = tree->parentTree->compareForAllCombine();
        while (co)
        {
            tree->parentTree->combineAllSiblings();
            tree = search(Point(_ix, _iy));
            co = tree->parentTree->compareForAllCombine();
        }
    }
    return;
}


// Insert a node into the quadtree and return the TreeNode.
// if node already exist: no insertion performed
inline TreeNode* TreeNode::insert(int _ix, int _iy)
{
    // Current quad cannot contain it
    if (!inBoundary(Point(_ix, _iy)))
    {
        throw std::invalid_argument("TreeNode::insert, out of Boundary");
        return nullptr;
    }

    TreeNode* tree = search(Point(_ix, _iy));

    while (!tree->unitArea())
    {
        tree->addChildren();
        tree = search(Point(_ix, _iy));
    }
    return tree;
}

// Adds 4 (or2) Children to Current Node which devide the Quad into 4 new Quads.
inline void TreeNode::addChildren()
{   
    bool unitAreaX = (topLeft.x + 1 == botRight.x);
    bool unitAreaY = (topLeft.y + 1 == botRight.y);

    if (!unitAreaX && !unitAreaY)
    {   //topLeftTree 
        TreeNode* child = new TreeNode(Point(topLeft.x, topLeft.y),
            Point(floor((topLeft.x + botRight.x) / 2.0), floor((topLeft.y + botRight.y) / 2.0)), level + 1, z, this);   // floor needed for topLeft = -1, botRight = 0
        child->millTimesteps = this->getMillTimesteps();
        this->childTrees.push_back(child);
    }
    if (!unitAreaX)
    {   //botLeftTree
        TreeNode* child = new TreeNode(Point(topLeft.x, floor((topLeft.y + botRight.y) / 2.0)),
            Point(floor((topLeft.x + botRight.x) / 2.0), botRight.y), level + 1, z, this);
        child->millTimesteps = this->getMillTimesteps();
        this->childTrees.push_back(child);
    }
    if (!unitAreaY)
    {   //topRightTree
        TreeNode* child = new TreeNode(Point(floor((topLeft.x + botRight.x) / 2.0), topLeft.y),
            Point(botRight.x, floor((topLeft.y + botRight.y) / 2.0)), level + 1, z, this);
        child->millTimesteps = this->getMillTimesteps();
        this->childTrees.push_back(child);
    }
    
    //botRightTree
    TreeNode* child = new TreeNode(Point(floor((topLeft.x + botRight.x) / 2.0), floor((topLeft.y + botRight.y) / 2.0)),
        Point(botRight.x, botRight.y), level + 1, z, this);
    child->millTimesteps = this->getMillTimesteps();
    this->childTrees.push_back(child);
    
    this->millTimesteps = vector<int>{};
}

// Find a node in a quadtree
inline TreeNode* TreeNode::search(Point p)
{
    // Current quad cannot contain it
    if (!inBoundary(p))
        return nullptr;

    // check if no children
    if (childTrees.size() == 0)     //(numChildren == 0)
    {
        return this;
    }

    for (TreeNode* childTree : childTrees)
    {
        if (childTree->inBoundary(p))
            return childTree->search(p);
    }
    return nullptr;

};
// Find the first ancestor of two nodes in the quadtree
inline TreeNode* TreeNode::searchCommonParent(TreeNode* tree1, TreeNode* tree2)
{   
    TreeNode* ancestor = tree1->parentTree;
    while(true)
    {
        if (ancestor->inBoundary(tree2->botRight))
            return ancestor;
        ancestor = ancestor->parentTree;
    }
};
// Returns a vector with all nodes from child towards ancestor.
inline vector<TreeNode*> TreeNode::searchAncestry(TreeNode* ancestor, TreeNode* child)
{
    TreeNode* parent = child->parentTree;
    vector<TreeNode*> parentLine;
    parentLine.push_back(child);
    while (ancestor != parent)
    {
        parentLine.push_back(parent);
        parent = parent->parentTree;
    }
    parentLine.push_back(ancestor);
    return parentLine;
};
// Returns a vector with all neighbors at the "side" border.
// side: 0 = xMin, 1 = yMin, 2 = xMax, 3 = yMax
inline vector<TreeNode*> TreeNode::searchAllNeighbors(TreeNode* treeRoot, int side)
{
    vector<TreeNode*> nbs = {};
    int xSearch;
    int ySearch;
    if (side == 0)
    {
        xSearch = this->topLeft.x;
        ySearch = this->topLeft.y + 1;
    }
    else if (side == 1)
    {
        xSearch = this->topLeft.x + 1;
        ySearch = this->topLeft.y;
    }
    else if (side == 2)
    {
        xSearch = this->botRight.x + 1;
        ySearch = this->topLeft.y + 1;
    }
    else if (side == 3)
    {
        xSearch = this->topLeft.x + 1;
        ySearch = this->botRight.y + 1;
    }
    TreeNode* nb1 = treeRoot->search(Point(xSearch, ySearch));
    if (nb1 == nullptr)
    {
        return nbs;
    }
    else
    {
        nbs.push_back(nb1);
        if (side == 0 || side == 2)
        {
            while (nbs.back()->botRight.y < this->botRight.y)
            {
                nbs.push_back(treeRoot->search(Point(xSearch, nbs.back()->botRight.y + 1)));
            }
        }
        else if (side == 1 || side == 3)
        {
            while (nbs.back()->botRight.x < this->botRight.x)
            {
                nbs.push_back(treeRoot->search(Point(nbs.back()->botRight.x + 1, ySearch)));
            }
        }
        return nbs;
    }
}


// Compares all children if they are leaves and have identical millTimesteps.
inline bool TreeNode::compareForAllCombine()
{
    bool combAll = true;
    vector<int> timeVec = childTrees[0]->millTimesteps;
    for (TreeNode* childTree : getChildTrees())
    {
        if (childTree->getChildTrees().size() != 0)
            combAll = false;
        if (!areVectorsEqual(timeVec, childTree->millTimesteps))
            combAll = false;
    }
    return combAll;
}
// Combines 4 (or2) Children into current Quad. No Tests are performed.
inline void TreeNode::combineAllSiblings()
{   
    z = childTrees[0]->z;
    millTimesteps = childTrees[0]->millTimesteps;
    for (TreeNode* child : childTrees)
    {
        child->valid = false;
    }
    childTrees = vector<TreeNode*>{};
    return;
}
// Compares and if possible combines the passed TreeNodes (if siblings and identical millTimesteps).
inline bool TreeNode::combine2Siblings(TreeNode* tree, TreeNode* sibling)
{
    if (tree != sibling && tree->parentTree == tree->parentTree && tree->valid && sibling->valid)
    {
        if (areVectorsEqual(tree->millTimesteps, sibling->millTimesteps))
        {
            //check border
            if (tree->topLeft.x == sibling->topLeft.x && tree->botRight.x == sibling->botRight.x)
            {
                if (tree->botRight.y == sibling->topLeft.y)
                {
                    tree->botRight.y = sibling->botRight.y;      // tree change y
                    tree->parentTree->childTrees.erase(std::remove(tree->parentTree->childTrees.begin(), tree->parentTree->childTrees.end(), sibling), tree->parentTree->childTrees.end());    // parent remove sibling
                    sibling->valid = false;
                    return true;
                }
                else if (tree->topLeft.y == sibling->botRight.y)
                {
                    tree->topLeft.y = sibling->topLeft.y;        // tree change y
                    tree->parentTree->childTrees.erase(std::remove(tree->parentTree->childTrees.begin(), tree->parentTree->childTrees.end(), sibling), tree->parentTree->childTrees.end());    // parent remove sibling
                    sibling->valid = false;
                    return true;
                }
            }
            else if (tree->topLeft.y == sibling->topLeft.y && tree->botRight.y == sibling->botRight.y)
            {
                if (tree->botRight.x == sibling->topLeft.x)
                {
                    tree->botRight.x = sibling->botRight.x;      // tree change x
                    tree->parentTree->childTrees.erase(std::remove(tree->parentTree->childTrees.begin(), tree->parentTree->childTrees.end(), sibling), tree->parentTree->childTrees.end());    // parent remove sibling
                    sibling->valid = false;
                    return true;
                }
                else if (tree->topLeft.x == sibling->botRight.x)
                {
                    tree->topLeft.x = sibling->topLeft.x;        // tree change x
                    tree->parentTree->childTrees.erase(std::remove(tree->parentTree->childTrees.begin(), tree->parentTree->childTrees.end(), sibling), tree->parentTree->childTrees.end());    // parent remove sibling
                    sibling->valid = false;
                    return true;
                }
            }
        }
    }
    return false;
}
// Compares and if possible combines the passed TreeNodes (any relation, only identical millTimesteps and same border required).
inline bool TreeNode::combine2TreeNodes(TreeNode* tree, TreeNode* neighbor)
{
    if (tree != neighbor && tree->valid && neighbor->valid)
    {
        if (areVectorsEqual(tree->millTimesteps, neighbor->millTimesteps))
        {
            if (borderCheck(tree, neighbor))
            {
                TreeNode* ancestor = searchCommonParent(tree, neighbor);
                vector<TreeNode*> treeParents = searchAncestry(ancestor, tree);
                vector<TreeNode*> neighborParents = searchAncestry(ancestor, neighbor);
                bool reorgaed = true;
                while (treeParents.size() >= 3 && neighborParents.size() >= 3 && reorgaed)
                {
                    int sizeT = treeParents.size();
                    int sizeN = neighborParents.size();
                    reorgaed = reorganise3generations(treeParents.back(), treeParents[sizeT - 2], neighborParents[sizeN - 2], treeParents[sizeT - 3], neighborParents[sizeN - 3]);
                    ancestor->eliminateSingularTrees();
                    ancestor = searchCommonParent(tree, neighbor);
                    treeParents = searchAncestry(ancestor, tree);
                    neighborParents = searchAncestry(ancestor, neighbor);
                }
                while (treeParents.size() >= 3)
                {
                    treeParents[1]->rebaseChildrenToParent(treeParents[0]);
                    treeParents = searchAncestry(ancestor, tree);
                }
                while (neighborParents.size() >= 3)
                {
                    neighborParents[1]->rebaseChildrenToParent(neighborParents[0]);
                    neighborParents = searchAncestry(ancestor, neighbor);
                }
                return combine2Siblings(tree, neighbor);
            }
        }
    }
    return false;
}
// Compares and if possible combines the passed TreeNodes (any relation, only identical millTimesteps and same border required).
inline bool TreeNode::combineMultiTreeNodes(vector<TreeNode*> treeVector)
{
    bool combinedAny = false;
    bool combinedLoop = true;
    while (combinedLoop)
    {   
        combinedLoop = false;
        for (TreeNode* child : treeVector)
        {   
            if (child != nullptr)
            {
                for (TreeNode* neighbor : treeVector)
                {
                    if (neighbor != nullptr)
                    {
                        if (child != neighbor && child->valid && neighbor->valid)
                        {
                            if (borderCheck(child, neighbor))
                            {
                                bool comb = combine2TreeNodes(child, neighbor);
                                combinedLoop = (combinedLoop || comb);
                            }
                        }
                    }
                }
            }
        }
        combinedAny = (combinedAny || combinedLoop);
    }
    return combinedAny;
}

// Traverses the whole tree and combines quads with millTimesteps if possible.
inline bool TreeNode::traverseAndCombineMillQuads()
{
    bool combinedAny = false;
    bool combinedLoop = true;
    while (combinedLoop)    // only combines siblings
    {   
        combinedLoop = false;
        std::stack<TreeNode*> nodeStack;
        nodeStack.push(this);
        while (!nodeStack.empty())
        {
            // traversiere Baum
            TreeNode* node = nodeStack.top();
            nodeStack.pop();
            vector<TreeNode*> millChildren;
            if (node->valid)
            {
                for (TreeNode* childTree : node->getChildTrees())
                {
                    if (childTree->getChildTrees().size() != 0)
                    {
                        nodeStack.push(childTree);
                    }
                    else if (childTree->millTimesteps.size() != 0)
                    {
                        millChildren.push_back(childTree);
                    }
                }
                bool comb = combineMultiTreeNodes(millChildren);
                combinedLoop = (combinedLoop || comb);
            }
        }
        combinedAny = (combinedAny || combinedLoop);
        this->eliminateSingleMillTrees();
    }
    combinedLoop = true;
    while (combinedLoop)    // combines neighbors
    {
        combinedLoop = false;
        std::stack<TreeNode*> nodeStack;
        nodeStack.push(this);
        while (!nodeStack.empty())
        {
            // traversiere Baum
            TreeNode* node = nodeStack.top();
            nodeStack.pop();
            std::stack<TreeNode*> millChildrenStack;
            if (node->valid)
            {
                for (TreeNode* childTree : node->getChildTrees())
                {
                    if (childTree->getChildTrees().size() != 0)
                    {
                        nodeStack.push(childTree);
                    }
                    else if (childTree->millTimesteps.size() != 0)
                    {
                        millChildrenStack.push(childTree);
                    }
                }
                while (!millChildrenStack.empty())
                {
                    TreeNode* millChild = millChildrenStack.top();
                    millChildrenStack.pop();
                    if (millChild->valid)
                    {
                        vector<TreeNode*> millNeighbors;    // could be nullptr !!
                        millNeighbors.push_back(millChild);
                        millNeighbors.push_back(search(Point(millChild->topLeft.x - 1, millChild->topLeft.y)));
                        millNeighbors.push_back(search(Point(millChild->topLeft.x, millChild->topLeft.y - 1)));
                        millNeighbors.push_back(search(Point(millChild->botRight.x + 1, millChild->botRight.y)));
                        millNeighbors.push_back(search(Point(millChild->botRight.x, millChild->botRight.y + 1)));
                        bool comb = combineMultiTreeNodes(millNeighbors);
                        combinedLoop = (combinedLoop || comb);
                    }
                }
            }
        }
        combinedAny = (combinedAny || combinedLoop);
    }
    this->eliminateSingularTrees();
    this->eliminateSingleMillTrees();
    return combinedAny;
}

// Traverses the whole tree and combines all sorts of Quads if possible.
inline bool TreeNode::traverseAndCombineAll()
{
    bool combinedAny = false;
    bool combinedLoop = true;
    while (combinedLoop)    // only combines siblings
    {
        combinedLoop = false;
        std::stack<TreeNode*> nodeStack;
        nodeStack.push(this);
        while (!nodeStack.empty())
        {
            // traversiere Baum
            TreeNode* node = nodeStack.top();
            nodeStack.pop();
            vector<TreeNode*> leafChildren;
            if (node->valid)
            {
                for (TreeNode* childTree : node->getChildTrees())
                {
                    if (childTree->getChildTrees().size() != 0)
                    {
                        nodeStack.push(childTree);
                    }
                    else
                    {
                        leafChildren.push_back(childTree);
                    }
                }
                bool comb = combineMultiTreeNodes(leafChildren);
                combinedLoop = (combinedLoop || comb);
            }
        }
        combinedAny = (combinedAny || combinedLoop);
        this->eliminateSingularTrees();
        this->eliminateSingleMillTrees();
    }
    //combinedLoop = true;
    //while (combinedLoop)    // combines neighbors
    //{
    //    combinedLoop = false;
    //    std::stack<TreeNode*> nodeStack;
    //    nodeStack.push(this);
    //    while (!nodeStack.empty())
    //    {
    //        // traversiere Baum
    //        TreeNode* node = nodeStack.top();
    //        nodeStack.pop();
    //        std::stack<TreeNode*> leafChildrenStack;
    //        if (node->valid)
    //        {
    //            for (TreeNode* childTree : node->getChildTrees())
    //            {
    //                if (childTree->getChildTrees().size() != 0)
    //                {
    //                    nodeStack.push(childTree);
    //                }
    //                else
    //                {
    //                    leafChildrenStack.push(childTree);
    //                }
    //            }
    //            while (!leafChildrenStack.empty())
    //            {
    //                TreeNode* leafChild = leafChildrenStack.top();
    //                leafChildrenStack.pop();
    //                if (leafChild->valid)
    //                {
    //                    vector<TreeNode*> leafNeighbors;    // could be nullptr !!
    //                    leafNeighbors.push_back(leafChild);
    //                    leafNeighbors.push_back(search(Point(leafChild->topLeft.x - 1, leafChild->topLeft.y)));
    //                    leafNeighbors.push_back(search(Point(leafChild->topLeft.x, leafChild->topLeft.y - 1)));
    //                    leafNeighbors.push_back(search(Point(leafChild->botRight.x + 1, leafChild->botRight.y)));
    //                    leafNeighbors.push_back(search(Point(leafChild->botRight.x, leafChild->botRight.y + 1)));
    //                    bool comb = combineMultiTreeNodes(leafNeighbors);
    //                    combinedLoop = (combinedLoop || comb);
    //                }
    //            }
    //        }
    //    }
    //    combinedAny = (combinedAny || combinedLoop);
    //    this->eliminateSingularTrees();
    //    this->eliminateSingleMillTrees();
    //}
    return combinedAny;
}

// Reorganises the tree: Quads child1+2 share a grandparent, but have different parents. After reorga they have the same parent.
// Siblings from child1+2 change parents.
// Checks if parent1+2 borders match.
inline bool TreeNode::reorganise3generations(TreeNode* grandp, TreeNode* parent1, TreeNode* parent2, TreeNode* child1, TreeNode* child2)
{
    bool parentMatch = borderCheck(parent1, parent2);
    bool childMatch = borderCheck(child1, child2);
    if (parentMatch && childMatch)
    {
        grandp->childTrees.erase(std::remove(grandp->childTrees.begin(), grandp->childTrees.end(), parent1), grandp->childTrees.end());    // grandparent remove parent1
        grandp->childTrees.erase(std::remove(grandp->childTrees.begin(), grandp->childTrees.end(), parent2), grandp->childTrees.end());    // grandparent remove parent2
        parent1->valid = false;
        parent2->valid = false;
        vector<TreeNode*> childLevel = parent1->childTrees;
        childLevel.insert(childLevel.end(), parent2->childTrees.begin(), parent2->childTrees.end());
        childLevel.erase(std::remove(childLevel.begin(), childLevel.end(), child1), childLevel.end());    // childLevel remove child1
        childLevel.erase(std::remove(childLevel.begin(), childLevel.end(), child2), childLevel.end());    // childLevel remove child2
        TreeNode* parentOneTwo = new TreeNode(child1, child2);
        parentOneTwo->childTrees.push_back(child1);
        parentOneTwo->childTrees.push_back(child2);
        for (TreeNode* childSibling : childLevel)
        {
            if (borderCheck(parentOneTwo, childSibling))
            {
                vector<TreeNode*> tempChildTrees = parentOneTwo->childTrees;
                parentOneTwo = new TreeNode(parentOneTwo, childSibling);
                childLevel.erase(std::remove(childLevel.begin(), childLevel.end(), childSibling), childLevel.end());    // childLevel remove childSibling
                parentOneTwo->childTrees = tempChildTrees;
                parentOneTwo->childTrees.push_back(childSibling);
            }
        }
        for (TreeNode* childOneTwo : parentOneTwo->childTrees)
        {
            childOneTwo->parentTree = parentOneTwo;
        }
        parentOneTwo->parentTree = grandp;
        parentOneTwo->level = grandp->level + 1;
        grandp->childTrees.push_back(parentOneTwo);

        while (childLevel.size() != 0)
        {
            TreeNode* childNext = childLevel.back();
            TreeNode* parentNext = new TreeNode(childNext, childNext);
            parentNext->childTrees.push_back(childNext);
            childLevel.pop_back();
            bool nextCombined;
            do {
                nextCombined = false;

                for (TreeNode* childSibling : childLevel)
                {
                    if (borderCheck(parentNext, childSibling))
                    {
                        vector<TreeNode*> tempChildTrees = parentNext->childTrees;
                        parentNext = new TreeNode(parentNext, childSibling);
                        childLevel.erase(std::remove(childLevel.begin(), childLevel.end(), childSibling), childLevel.end());    // childLevel remove childSibling
                        parentNext->childTrees = tempChildTrees;
                        parentNext->childTrees.push_back(childSibling);
                        nextCombined = true;
                    }
                }
            } while (nextCombined);
            if (parentNext->childTrees.size() <= 1)     // if no matching sibling found for childNext
            {
                parentNext = childNext;
            }
            for (TreeNode* childNext : parentNext->childTrees)
            {
                childNext->parentTree = parentNext;
            }
            parentNext->parentTree = grandp;
            parentNext->level = grandp->level + 1;
            grandp->childTrees.push_back(parentNext);
        }
    }
    return (parentMatch && childMatch);
};
// Removes this treeNode and adds all children to parent.
inline void TreeNode::rebaseChildrenToParent()
{
    for (TreeNode* child : this->childTrees)
    {
        child->parentTree = this->parentTree;
        this->parentTree->childTrees.push_back(child);
    }
    this->valid = false;
    this->parentTree->childTrees.erase(std::remove(this->parentTree->childTrees.begin(), this->parentTree->childTrees.end(), this), this->parentTree->childTrees.end());    // parent remove this
};
// Changes this treeNode and adds passed child to parent, rest doesn't matter.
inline void TreeNode::rebaseChildrenToParent(TreeNode* child)
{   
    //if (std::find(this->childTrees.begin(), this->childTrees.end(), child) != this->childTrees.end())
    if (this->valid && child->valid)
    {
        child->parentTree = this->parentTree;
        this->parentTree->childTrees.push_back(child);
        this->childTrees.erase(std::remove(this->childTrees.begin(), this->childTrees.end(), child), this->childTrees.end());    // this remove child

        while (childTrees.size() != 0)
        {
            TreeNode* childNext = childTrees.back();
            TreeNode* parentNext = new TreeNode(childNext, childNext);
            parentNext->childTrees.push_back(childNext);
            childTrees.pop_back();
            bool nextCombined;
            do {
                nextCombined = false;

                for (TreeNode* childSibling : this->childTrees)
                {
                    if (borderCheck(parentNext, childSibling))
                    {
                        vector<TreeNode*> tempChildTrees = parentNext->childTrees;
                        parentNext = new TreeNode(parentNext, childSibling);
                        this->childTrees.erase(std::remove(this->childTrees.begin(), this->childTrees.end(), childSibling), this->childTrees.end());    // this->childTrees remove childSibling
                        parentNext->childTrees = tempChildTrees;
                        parentNext->childTrees.push_back(childSibling);
                        nextCombined = true;
                    }
                }
            } while (nextCombined);
            if (parentNext->childTrees.size() <= 1)     // if no matching sibling found for childNext
            {
                parentNext = childNext;
            }
            for (TreeNode* childNext : parentNext->childTrees)
            {
                childNext->parentTree = parentNext;
            }
            parentNext->parentTree = this->parentTree;
            parentNext->level = this->parentTree->level + 1;
            this->parentTree->childTrees.push_back(parentNext);
        }
        this->valid = false;
        this->parentTree->childTrees.erase(std::remove(this->parentTree->childTrees.begin(), this->parentTree->childTrees.end(), this), this->parentTree->childTrees.end());    // parent remove this
    }
};
// Checks if tree1 and tree2 share one border/edge.
inline bool TreeNode::borderCheck(TreeNode* tree1, TreeNode* tree2)
{
    //check border
    if (tree1->topLeft.x == tree2->topLeft.x && tree1->botRight.x == tree2->botRight.x)
    {
        if (tree1->botRight.y == tree2->topLeft.y)
        {
            return true;
        }
        else if (tree1->topLeft.y == tree2->botRight.y)
        {
            return true;
        }
    }
    else if (tree1->topLeft.y == tree2->topLeft.y && tree1->botRight.y == tree2->botRight.y)
    {
        if (tree1->botRight.x == tree2->topLeft.x)
        {
            return true;
        }
        else if (tree1->topLeft.x == tree2->botRight.x)
        {
            return true;
        }
    }
    return false;
}

// Traverses the whole tree and deletes treeNodes with only one child. 
inline void TreeNode::eliminateSingularTrees()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        if (node->valid)
        {
            for (TreeNode* childTree : node->getChildTrees())
            {
                nodeStack.push(childTree);
            }
            if (node->getChildTrees().size() == 1)
            {
                TreeNode* child = node->childTrees[0];
                if (node->topLeft.x == child->topLeft.x && node->topLeft.y == child->topLeft.y
                    && node->botRight.x == child->botRight.x && node->botRight.y == child->botRight.y)
                {
                    child->parentTree = node->parentTree;
                    child->parentTree->childTrees.push_back(child);
                    node->valid = false;
                    node->parentTree->childTrees.erase(std::remove(node->parentTree->childTrees.begin(), node->parentTree->childTrees.end(), node), node->parentTree->childTrees.end());    // parent remove node
                }
                else
                {
                    node = node;    // branch should never be reached.
                }
            }
        }
    }
}
// Traverses the whole tree and deletes treeNodes with 2 children of which 1 is a leaf whitout millStep. 
inline void TreeNode::eliminateSingleMillTrees()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        if (node->valid)
        {
            for (TreeNode* childTree : node->getChildTrees())
            {
                nodeStack.push(childTree);
            }
            if (node->getChildTrees().size() == 2)
            {
                if (node->getChildTrees()[0]->childTrees.size() == 0 && node->getChildTrees()[0]->millTimesteps.size() == 0)
                {
                        node->rebaseChildrenToParent();
                }
                else if (node->getChildTrees()[1]->childTrees.size() == 0 && node->getChildTrees()[1]->millTimesteps.size() == 0)
                {
                        node->rebaseChildrenToParent();
                }
            }
        }
    }
}

// Traverses the whole tree and deletes treeNodes with only one child. 
inline TreeNode* TreeNode::searchForInvalid()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        for (TreeNode* childTree : node->getChildTrees())
        {
            nodeStack.push(childTree);
        }
        if (!node->isValid())
        {
            return node;
        }
    }
    return nullptr;
}
// Traverses the whole tree and sets the level for every node. 
inline void TreeNode::setLevels()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        for (TreeNode* childTree : node->getChildTrees())
        {
            nodeStack.push(childTree);
        }
        if (node->parentTree != nullptr)
        {
            node->level = node->parentTree->level + 1;
        }
    }
}

// Traverses the whole tree and sets the number of descendants for every node. 
inline void TreeNode::setDescendants()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        for (TreeNode* childTree : node->getChildTrees())
        {
            nodeStack.push(childTree);
        }
        node->numDescendants = 0;
        if (node->childTrees.size() == 0)
        { 
            TreeNode* descendant = node;
            while (descendant->parentTree != nullptr)
            {
                descendant->parentTree->numDescendants += 1;
                descendant = descendant->parentTree;
            }
        }
    }
}

// Traverses the whole tree and sorts children without millTimesteps to the end of childTrees.
inline void TreeNode::sortChildren()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        for (TreeNode* childTree : node->getChildTrees())
        {   
            std::stack<TreeNode*> leafNoMillStack;
            if (childTree->getChildTrees().size() != 0)
            {
                nodeStack.push(childTree);
            }
            else if (childTree->getMillTimesteps().size() == 0)
            {
                leafNoMillStack.push(childTree);
            }
            while (!leafNoMillStack.empty())
            {
                TreeNode* leafNoMill = leafNoMillStack.top();
                leafNoMillStack.pop();
                node->childTrees.erase(std::remove(node->childTrees.begin(), node->childTrees.end(), leafNoMill), node->childTrees.end());    // node remove leafNoMill
                node->childTrees.push_back(leafNoMill);
            }
        }
    }
}

// Traverses the whole tree and sorts children without millTimesteps to the end of childTrees.
inline void TreeNode::setSideWalls()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        if (node->valid)
        {
            for (TreeNode* childTree : node->getChildTrees())
            {
                nodeStack.push(childTree);
            }
            if (node->getChildTrees().size() == 0)
            {
                node->sideWalls = { false, false, false, false };
                for (int side = 0; side < 4; side++)
                {
                    vector<TreeNode*> nbs = node->searchAllNeighbors(this, side);
                    if (nbs.size() == 0)
                        node->sideWalls[side] = true;
                    for (TreeNode* nb : nbs)
                    {
                        if (!areVectorsEqual(node->millTimesteps, nb->millTimesteps))
                        {
                            if (nb->millTimesteps.size() != 0)
                                node->sideWalls[side] = true;
                        }
                    }
                }
            }
        }
    }
}
// Check if current quadtree contains the point
inline vector<vector<TreeNode*>> TreeNode::writeTimestepVector(vector<vector<TreeNode*>> timestepVec)
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        if (node->valid)
        {
            for (TreeNode* childTree : node->getChildTrees())
            {
                nodeStack.push(childTree);
            }
            if (node->getChildTrees().size() == 0)
            {
                for (int time : node->millTimesteps)
                {
                    timestepVec[time].push_back(node);
                }
            }
        }
    }
    return timestepVec;
}

#endif