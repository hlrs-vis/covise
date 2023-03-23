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
    //vector<MillCoords*> nodeCoords;

    // Children of this tree
    TreeNode* parentTree;
    int level;
    //int numChildren;            //0,1,2,3,4; -1 -> leafNode
    //int numDescendants;           //incorrect if insertion fails
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
    }

    vector<TreeNode*> getChildTrees();
    vector<int> getMillTimesteps();
    //vector<TreeNode*> getSiblingTrees();
    //vector<TreeNode*> getSiblingLeafs();
    Point getTopLeft();
    Point getBotRight();
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
//    TreeNode* searchParent(Point);
    bool inBoundary(Point);
    bool unitArea();
//    void traverseAndCombineQuads();
    bool compareForAllCombine();
    void combineAllSiblings();
    bool compareCombine2Siblings(vector<TreeNode*>);
    void traverseAndCallCC2Siblings();
    bool compareCombineNeighbors(TreeNode*, vector<TreeNode*>);
    void traverseAndCallCCNeighbor();
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
// Find the first anchestor of two nodes in the quadtree
inline TreeNode* TreeNode::searchCommonParent(TreeNode* tree1, TreeNode* tree2)
{   
    TreeNode* anchestor = tree1->parentTree;
    while(true)
    {
        if (anchestor->inBoundary(tree2->botRight))
            return anchestor;
        anchestor = anchestor->parentTree;
    }
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
                            return true;
                        }
                        else if (child->topLeft.y == sibling->botRight.y)
                        {
                            child->topLeft.y = sibling->topLeft.y;      // child change y
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
                            return true;
                        }
                    }
                    else if (child->topLeft.y == sibling->topLeft.y && child->botRight.y == sibling->botRight.y)
                    {
                        if (child->botRight.x == sibling->topLeft.x)
                        {
                            child->botRight.x = sibling->botRight.x;      // child change x
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
                            return true;
                        }
                        else if (child->topLeft.x == sibling->botRight.x)
                        {
                            child->topLeft.x = sibling->topLeft.x;      // child change x
                            child->parentTree->childTrees.erase(std::remove(child->parentTree->childTrees.begin(), child->parentTree->childTrees.end(), sibling), child->parentTree->childTrees.end());    // parent remove sibling
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
inline void TreeNode::traverseAndCallCC2Siblings()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        vector<TreeNode*> leafChildren;
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


// Compares all children if they are leaves and have identical millTimesteps.
inline bool TreeNode::compareCombineNeighbors(TreeNode* leaf, vector<TreeNode*> leafNeighbors)
{
    
    for (TreeNode* neighbor : leafNeighbors)
    {   
        if (neighbor != nullptr && neighbor->level > 0 && neighbor->level < 20 && neighbor->topLeft.x > -2 && neighbor->topLeft.x < 4000)// && neighbor->getMillTimesteps() != nullptr)
        {
            if (areVectorsEqual(leaf->millTimesteps, neighbor->millTimesteps))
            {
                //check border
                if (leaf->topLeft.x == neighbor->topLeft.x && leaf->botRight.x == neighbor->botRight.x)
                {
                    if (leaf->botRight.y == neighbor->topLeft.y)
                    {
                        leaf->botRight.y = neighbor->botRight.y;      // leaf change y
                        TreeNode* anchestor = searchCommonParent(leaf, neighbor);
                        anchestor->childTrees.push_back(leaf);
                        leaf->parentTree->childTrees.erase(std::remove(leaf->parentTree->childTrees.begin(), leaf->parentTree->childTrees.end(), leaf), leaf->parentTree->childTrees.end());    // parent remove leaf
                        neighbor->parentTree->childTrees.erase(std::remove(neighbor->parentTree->childTrees.begin(), neighbor->parentTree->childTrees.end(), neighbor), neighbor->parentTree->childTrees.end());    // parent remove neighbor
                        return true;
                    }
                    else if (leaf->topLeft.y == neighbor->botRight.y)
                    {
                        leaf->topLeft.y = neighbor->topLeft.y;      // leaf change y
                        TreeNode* anchestor = searchCommonParent(leaf, neighbor);
                        anchestor->childTrees.push_back(leaf);
                        leaf->parentTree->childTrees.erase(std::remove(leaf->parentTree->childTrees.begin(), leaf->parentTree->childTrees.end(), leaf), leaf->parentTree->childTrees.end());    // parent remove leaf
                        neighbor->parentTree->childTrees.erase(std::remove(neighbor->parentTree->childTrees.begin(), neighbor->parentTree->childTrees.end(), neighbor), neighbor->parentTree->childTrees.end());    // parent remove neighbor
                        return true;
                    }
                }
                else if (leaf->topLeft.y == neighbor->topLeft.y && leaf->botRight.y == neighbor->botRight.y)
                {
                    if (leaf->botRight.x == neighbor->topLeft.x)
                    {
                        leaf->botRight.x = neighbor->botRight.x;      // leaf change x
                        TreeNode* anchestor = searchCommonParent(leaf, neighbor);
                        anchestor->childTrees.push_back(leaf);
                        leaf->parentTree->childTrees.erase(std::remove(leaf->parentTree->childTrees.begin(), leaf->parentTree->childTrees.end(), leaf), leaf->parentTree->childTrees.end());    // parent remove leaf
                        neighbor->parentTree->childTrees.erase(std::remove(neighbor->parentTree->childTrees.begin(), neighbor->parentTree->childTrees.end(), neighbor), neighbor->parentTree->childTrees.end());    // parent remove neighbor
                        return true;
                    }
                    else if (leaf->topLeft.x == neighbor->botRight.x)
                    {
                        leaf->topLeft.x = neighbor->topLeft.x;      // leaf change x
                        TreeNode* anchestor = searchCommonParent(leaf, neighbor);
                        anchestor->childTrees.push_back(leaf);
                        leaf->parentTree->childTrees.erase(std::remove(leaf->parentTree->childTrees.begin(), leaf->parentTree->childTrees.end(), leaf), leaf->parentTree->childTrees.end());    // parent remove leaf
                        neighbor->parentTree->childTrees.erase(std::remove(neighbor->parentTree->childTrees.begin(), neighbor->parentTree->childTrees.end(), neighbor), neighbor->parentTree->childTrees.end());    // parent remove neighbor
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

// Traverses the whole Tree and combines two neighbor quads (from different parents) if possible.
inline void TreeNode::traverseAndCallCCNeighbor()
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(this);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        std::stack<TreeNode*> leafChildrenStack;
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
            vector<TreeNode*> leafNeighbors;    // could be nullptr !!
            leafNeighbors.push_back(search(Point(leaf->topLeft.x - 1, leaf->topLeft.y)));
            leafNeighbors.push_back(search(Point(leaf->topLeft.x, leaf->topLeft.y - 1)));
            leafNeighbors.push_back(search(Point(leaf->botRight.x + 1, leaf->botRight.y)));
            leafNeighbors.push_back(search(Point(leaf->botRight.x, leaf->botRight.y + 1)));

            compareCombineNeighbors(leaf, leafNeighbors);
        }

    /*    bool comb = compareCombine2Siblings(leafChildren);

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
    */
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