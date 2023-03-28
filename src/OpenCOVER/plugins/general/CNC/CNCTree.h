#ifndef CNCTREE_H
#define CNCTREE_H

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

/*
// The objects that we want stored in the quadtree
struct MillCoords {
    Point pos;
    int cuttedFaces;
    double z;
    int primitivePos;
    MillCoords(Point _pos, double _z)
    {
        pos = _pos;
        cuttedFaces = -1;
        z = _z;
        primitivePos = -1;
    }
    //MillCoords() { data = 0; }
};*/

// The main quadtree class
class TreeNode {
    // Hold details of the boundary of this node
    Point topLeft;
    Point botRight;

    // Contains details of node
    double z;
    int primitivePos;
    bool valid;         //if node is still valid
    //vector<MillCoords*> nodeCoords;

    // Children of this tree
    TreeNode* parentTree;
    int level;
    //int numChildren;            //0,1,2,3,4; -1 -> leafNode
    int numDescendants;
    vector<TreeNode*> childTrees;
    vector<int> millTimesteps;      //the timesteps where this Node is milled
    
public:
    TreeNode(Point _topL, Point _botR, int _level)
    {
        level = _level;
        //numChildren = 0;
        //numDescendants = 0;
        topLeft = _topL;
        botRight = _botR;
        valid = true;
    }
    TreeNode(Point _topL, Point _botR, int _level, double _z, TreeNode* _parentTree)//, int _numChildren)
    {
        //nodeCoords.push_back(_coords);
        z = _z;
        level = _level;
        //numChildren = _numChildren;
        //numDescendants = 0;
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
    //vector<TreeNode*> getSiblingTrees();
    //vector<TreeNode*> getSiblingLeafs();
    Point getTopLeft();
    Point getBotRight();
    int getLevel();
    bool isValid();
    /*
    double getNumChildren();
    void setZ(double _z);
    double getZ();
    void setPrimitivePos(int _primitivePos);
    int getPrimitivePos();
    */
    void millQuad(int _ix, int _iy, double _z, int t);
    TreeNode* insert(int _ix, int _iy);
    void addChildren();
    TreeNode* search(Point);
    TreeNode* searchCommonParent(TreeNode*, TreeNode*);
    vector<TreeNode*> searchAncestry(TreeNode*, TreeNode*);
//    TreeNode* searchParent(Point);
    bool inBoundary(Point);
    bool unitArea();
//    void traverseAndCombineQuads();
    bool compareForAllCombine();
    void combineAllSiblings();
    bool compareCombine2Siblings(vector<TreeNode*>);
    bool traverseAndCallCC2Siblings();
    bool compareCombineNeighbors(TreeNode*, vector<TreeNode*>);
    bool compareCombineAllNeighbors(TreeNode*, vector<TreeNode*>);
    void traverseAndCallCCNeighbor();
    void traverseAndCallCCAllNeighbor();
    bool reorganise3generations(TreeNode*, TreeNode*, TreeNode*, TreeNode*, TreeNode*);
    void rebaseChildrenToParent();
    void rebaseChildrenToParent(TreeNode*);
    bool borderCheck(TreeNode*, TreeNode*);
    void eliminateSingularTrees();
    TreeNode* searchForInvalid();
    void setLevels();
    void setDescendants();
    //bool combineOneSibling();
    bool areVectorsEqual(std::vector<int> vec1, std::vector<int> vec2);
    /*
    bool topLeftBoundary(Point);
    bool updateZ(int _ix, int _iy, double _z);
    */
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
/*
inline double TreeNode::getNumChildren()
{
    return numChildren;
}


inline void TreeNode::setZ(double _z)
{   
    n->z = _z;
    return;
}
inline double TreeNode::getZ()
{
    return n->z;
}

inline void TreeNode::setPrimitivePos(int _primitivePos)
{
    n->primitivePos = _primitivePos;
    return;
}
inline int TreeNode::getPrimitivePos()
{
    return n->primitivePos;
}
*/


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


// Insert a node into the quadtree
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
            Point((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2), level + 1, z, this);
        child->millTimesteps = this->getMillTimesteps();
        this->childTrees.push_back(child);
    }
    if (!unitAreaX)
    {   //botLeftTree
        TreeNode* child = new TreeNode(Point(topLeft.x, (topLeft.y + botRight.y) / 2),
            Point((topLeft.x + botRight.x) / 2, botRight.y), level + 1, z, this);
        child->millTimesteps = this->getMillTimesteps();
        this->childTrees.push_back(child);
    }
    if (!unitAreaY)
    {   //topRightTree
        TreeNode* child = new TreeNode(Point((topLeft.x + botRight.x) / 2, topLeft.y),
            Point(botRight.x, (topLeft.y + botRight.y) / 2), level + 1, z, this);
        child->millTimesteps = this->getMillTimesteps();
        this->childTrees.push_back(child);
    }
    
    //botRightTree
    TreeNode* child = new TreeNode(Point((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2),
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
        return NULL;

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
/*
// Find the parent of a node in a quadtree
inline TreeNode* TreeNode::searchParent(Point p)
{
    // Current quad cannot contain it
    if (!inBoundary(p))
        return NULL;

    // check if no children
    if (childTrees.size() == 0)     //(numChildren == 0)
    {
        return NULL;
    }

    for (TreeNode* childTree : childTrees)
    {
        if (childTree->inBoundary(p))
        {
            if (childTree->getChildTrees().size() == 0)
                return this;
            else
                return childTree->searchParent(p);
        }
            
    }

};
*/
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

// Combines multiple Quads into 1 new Quad.
/*inline void TreeNode::traverseAndCombineQuads()
{   
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        //for (TreeNode* childTree : node->getChildTrees())
        for (int i = 0; i < node->getChildTrees().size(); i++)
        {   
            //if (childTree->getChildTrees().size() != 0)
            if (node->getChildTrees()[i]->getChildTrees().size() != 0)
            {
                //nodeStack.push(childTree);
                nodeStack.push(node->getChildTrees()[i]);
            }
          
            //else if (childTree->millTimesteps.size() == 0)
            else if (node->getChildTrees()[i]->millTimesteps.size() == 0)
            {
                //childTree->combineOneSibling();
                //if (node->getChildTrees()[i]->combineOneSibling())
                //    nodeStack.push(node);
            }
        }
    }
}
*/

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
// Compares all children if they are leaves and have identical millTimesteps.
inline bool TreeNode::compareCombine2Siblings(vector<TreeNode*> leafSiblings)
{
    for (TreeNode* child : leafSiblings)
    {
        for (TreeNode* sibling : leafSiblings)
        {
            if (child != sibling)
            {
                if (areVectorsEqual(child->millTimesteps, sibling->millTimesteps))
                {
                    //check border
                    if (child->topLeft.x == sibling->topLeft.x && child->botRight.x == sibling->botRight.x)
                    {
                        if (child->botRight.y == sibling->topLeft.y)
                        {
                            child->botRight.y = sibling->botRight.y;      // child change y
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
                            sibling->valid = false;
                            return true;
                        }
                        else if (child->topLeft.y == sibling->botRight.y)
                        {
                            child->topLeft.y = sibling->topLeft.y;      // child change y
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
                            sibling->valid = false;
                            return true;
                        }
                    }
                    else if (child->topLeft.y == sibling->topLeft.y && child->botRight.y == sibling->botRight.y)
                    {
                        if (child->botRight.x == sibling->topLeft.x)
                        {
                            child->botRight.x = sibling->botRight.x;      // child change x
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
                            sibling->valid = false;
                            return true;
                        }
                        else if (child->topLeft.x == sibling->botRight.x)
                        {
                            child->topLeft.x = sibling->topLeft.x;      // child change x
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
                            sibling->valid = false;
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}
// Traverses the whole tree and combines two direct Siblings into one Quad if possible.
inline bool TreeNode::traverseAndCallCC2Siblings()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    bool combinedAny = false;
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
            bool comb = compareCombine2Siblings(leafChildren);
            combinedAny = (combinedAny || comb);
            while (comb)
            {
                vector<TreeNode*> leafChildren;
                for (TreeNode* childTree : node->getChildTrees())
                {
                    if (childTree->getChildTrees().size() == 0)
                    {
                        leafChildren.push_back(childTree);
                    }
                }
                comb = compareCombine2Siblings(leafChildren);
            }
        }
    }
    return combinedAny;
}


// Compares all children if they are leaves and have identical millTimesteps.
inline bool TreeNode::compareCombineNeighbors(TreeNode* leaf, vector<TreeNode*> leafNeighbors)
{
    bool combinedAny = false;
    for (TreeNode* neighbor : leafNeighbors)
    {   
        if (neighbor != nullptr && neighbor->valid && leaf->valid)// && neighbor->level > 0 && neighbor->level < 20 && neighbor->topLeft.x > -2 && neighbor->topLeft.x < 4000)// && neighbor->getMillTimesteps() != nullptr)
        {
            if (areVectorsEqual(leaf->millTimesteps, neighbor->millTimesteps))
            {   
                //check border
                bool combineNb = borderCheck(leaf, neighbor);
                if (combineNb)
                {
                    TreeNode* ancestor = searchCommonParent(leaf, neighbor);
                    vector<TreeNode*> leafParents = searchAncestry(ancestor, leaf);
                    vector<TreeNode*> neighborParents = searchAncestry(ancestor, neighbor);
                    if (leafParents.size() == neighborParents.size())
                    {   
                        bool reorgaed = true;
                        bool comb = false;
                        while (leafParents.size() >= 3 && neighborParents.size() >= 3 && reorgaed)
                        {   
                            int sizeL = leafParents.size();
                            int sizeN = neighborParents.size();
                            reorgaed = reorganise3generations(leafParents.back(), leafParents[sizeL - 2], neighborParents[sizeN - 2], leafParents[sizeL - 3], neighborParents[sizeN - 3]);
                            ancestor->eliminateSingularTrees();
                            ancestor = searchCommonParent(leaf, neighbor);
                            comb = (comb || ancestor->traverseAndCallCC2Siblings());
                            ancestor = searchCommonParent(leaf, neighbor);
                            leafParents = searchAncestry(ancestor, leaf);
                            neighborParents = searchAncestry(ancestor, neighbor);
                        }
                        return comb;
                        //return ancestor->traverseAndCallCC2Siblings();  // hier Optimierungspotenzial. welchen Node muss ich wie nochmal versuchen zu comparen?
                        //return true;
                    }
                    else if (leafParents.size() - 1 == neighborParents.size() || neighborParents.size() == leafParents.size() + 1)
                    {
                        bool reorgaed = true;
                        bool comb = false;
                        while (leafParents.size() >= 3 && neighborParents.size() >= 3 && reorgaed && neighbor->valid && leaf->valid)
                        {
                            int sizeL = leafParents.size();
                            int sizeN = neighborParents.size();
                            reorgaed = reorganise3generations(leafParents.back(), leafParents[sizeL - 2], neighborParents[sizeN - 2], leafParents[sizeL - 3], neighborParents[sizeN - 3]);
                            ancestor->eliminateSingularTrees();
                            ancestor = searchCommonParent(leaf, neighbor);
                            comb = (comb || ancestor->traverseAndCallCC2Siblings());
                            ancestor = searchCommonParent(leaf, neighbor);
                            leafParents = searchAncestry(ancestor, leaf);
                            neighborParents = searchAncestry(ancestor, neighbor);
                        }
                        if (leafParents.size() <= 3 && neighborParents.size() <= 3 && neighbor->valid && leaf->valid) // rebaseChildrenToParent erhöht Anzahl childTrees massiv? -> bessere Varianten notwendig?
                        {
                            if (leafParents.size() == 3)
                            {
                                leafParents[1]->rebaseChildrenToParent(leafParents[0]);
                            }
                            else if (neighborParents.size() == 3)
                            {
                                neighborParents[1]->rebaseChildrenToParent(neighborParents[0]);
                            }
                            ancestor = searchCommonParent(leaf, neighbor);
                            ancestor->eliminateSingularTrees();
                            comb = (comb || ancestor->traverseAndCallCC2Siblings());
                            ancestor->searchForInvalid();
                        }
                        return comb;
                    }
                }
            }
        }
    }
    return false;
}

// Compares all children if they are leaves and have identical millTimesteps.
inline bool TreeNode::compareCombineAllNeighbors(TreeNode* leaf, vector<TreeNode*> leafNeighbors)
{
    bool combinedAny = false;
    for (TreeNode* neighbor : leafNeighbors)
    {
        if (neighbor != nullptr && neighbor->valid && leaf->valid)// && neighbor->level > 0 && neighbor->level < 20 && neighbor->topLeft.x > -2 && neighbor->topLeft.x < 4000)// && neighbor->getMillTimesteps() != nullptr)
        {
            if (areVectorsEqual(leaf->millTimesteps, neighbor->millTimesteps))
            {
                //check border
                bool combineNb = borderCheck(leaf, neighbor);
                combinedAny = (combinedAny || combineNb);
                if (combineNb)
                {
                    TreeNode* ancestor = searchCommonParent(leaf, neighbor);
                    vector<TreeNode*> leafParents = searchAncestry(ancestor, leaf);
                    vector<TreeNode*> neighborParents = searchAncestry(ancestor, neighbor);

                    bool reorgaed = true;
                    while (leafParents.size() >= 3 && neighborParents.size() >= 3 && reorgaed)
                    {
                        int sizeL = leafParents.size();
                        int sizeN = neighborParents.size();
                        reorgaed = reorganise3generations(leafParents.back(), leafParents[sizeL - 2], neighborParents[sizeN - 2], leafParents[sizeL - 3], neighborParents[sizeN - 3]);
                        ancestor->eliminateSingularTrees();
                        ancestor = searchCommonParent(leaf, neighbor);
                        leafParents = searchAncestry(ancestor, leaf);
                        neighborParents = searchAncestry(ancestor, neighbor);
                    }
                    while (leafParents.size() >= 3)
                    {
                        leafParents[1]->rebaseChildrenToParent(leafParents[0]);
                        leafParents = searchAncestry(ancestor, leaf);
                    }
                    while (neighborParents.size() >= 3)
                    {
                        neighborParents[1]->rebaseChildrenToParent(neighborParents[0]);
                        neighborParents = searchAncestry(ancestor, neighbor);
                    }
                    ancestor->traverseAndCallCC2Siblings();
                    ancestor->eliminateSingularTrees();
                }
            }
        }
    }
    return combinedAny;
}

// Traverses the whole Tree and combines two neighbor quads (from different parents) if possible.
inline void TreeNode::traverseAndCallCCNeighbor()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    bool nbCombined = false;
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        std::stack<TreeNode*> leafChildrenStack;
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
                    leafChildrenStack.push(childTree);
                }
            }

            while (!leafChildrenStack.empty())
            {
                TreeNode* leaf = leafChildrenStack.top();
                leafChildrenStack.pop();
                if (leaf->valid)
                {
                    vector<TreeNode*> leafNeighbors;    // could be nullptr !!
                    leafNeighbors.push_back(search(Point(leaf->topLeft.x - 1, leaf->topLeft.y)));
                    leafNeighbors.push_back(search(Point(leaf->topLeft.x, leaf->topLeft.y - 1)));
                    leafNeighbors.push_back(search(Point(leaf->botRight.x + 1, leaf->botRight.y)));
                    leafNeighbors.push_back(search(Point(leaf->botRight.x, leaf->botRight.y + 1)));

                    bool leafComb = compareCombineNeighbors(leaf, leafNeighbors);
                    if (leafComb)
                        leafChildrenStack.push(leaf);
                    nbCombined = (leafComb || nbCombined);
                }
            }
        }
    }
    if (nbCombined)
        this->traverseAndCallCCNeighbor();
    else
        this->traverseAndCallCCAllNeighbor();
}
// Traverses the whole Tree and combines two neighbor quads (from different parents) if possible.
inline void TreeNode::traverseAndCallCCAllNeighbor()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    bool nbCombined = false;
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        std::stack<TreeNode*> leafChildrenStack;
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
                    leafChildrenStack.push(childTree);
                }
            }

            while (!leafChildrenStack.empty())
            {
                TreeNode* leaf = leafChildrenStack.top();
                leafChildrenStack.pop();
                if (leaf->valid)
                {
                    vector<TreeNode*> leafNeighbors;    // could be nullptr !!
                    leafNeighbors.push_back(search(Point(leaf->topLeft.x - 1, leaf->topLeft.y)));
                    leafNeighbors.push_back(search(Point(leaf->topLeft.x, leaf->topLeft.y - 1)));
                    leafNeighbors.push_back(search(Point(leaf->botRight.x + 1, leaf->botRight.y)));
                    leafNeighbors.push_back(search(Point(leaf->botRight.x, leaf->botRight.y + 1)));

                    bool leafComb = compareCombineAllNeighbors(leaf, leafNeighbors);
                    if (leafComb)
                        leafChildrenStack.push(leaf);
                    nbCombined = (leafComb || nbCombined);
                }
            }
        }
    }
    if (nbCombined)
        this->traverseAndCallCCNeighbor();
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
    for (TreeNode* child : childTrees)
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

/*
// Check if point is topLeft corner of current quadtree
inline bool TreeNode::topLeftBoundary(Point p)
{
    return (p.x == topLeft.x && p.y == topLeft.y);
}

// Search node/Millcords and update z if _z < current z,
// return true if z was updated, false if _z >= current z
inline bool TreeNode::updateZ(int _ix, int _iy, double _z)
{   
    MillCoords* coords = this->search(Point(_ix, _iy));
    if (_z < coords->z)
    {
        coords->z = _z;
        return true;
    }
    else
        return false;
}

*/

//getter und setter hinzufügen!


#endif