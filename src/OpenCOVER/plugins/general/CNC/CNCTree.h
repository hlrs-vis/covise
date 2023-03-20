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
    TreeNode(Point _topL, Point _botR, int _level, double _z)//, int _numChildren)
    {
        //nodeCoords.push_back(_coords);
        z = _z;
        level = _level;
        //numChildren = _numChildren;
        //numDescendants = 0;
        topLeft = _topL;
        botRight = _botR;
    }

    vector<TreeNode*> getChildTrees();
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
    bool inBoundary(Point);
    bool unitArea();
    /*
    bool topLeftBoundary(Point);
    bool updateZ(int _ix, int _iy, double _z);
    */
};

inline vector<TreeNode*> TreeNode::getChildTrees()
{
    return childTrees;
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
            Point((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2), level + 1, z);
        this->childTrees.push_back(child);
    }
    if (!unitAreaX)
    {   //botLeftTree
        TreeNode* child = new TreeNode(Point(topLeft.x, (topLeft.y + botRight.y) / 2),
            Point((topLeft.x + botRight.x) / 2, botRight.y), level + 1, z);
        this->childTrees.push_back(child);
    }
    if (!unitAreaY)
    {   //topRightTree
        TreeNode* child = new TreeNode(Point((topLeft.x + botRight.x) / 2, topLeft.y),
            Point(botRight.x, (topLeft.y + botRight.y) / 2), level + 1, z);
        this->childTrees.push_back(child);
    }
    
    //botRightTree
    TreeNode* child = new TreeNode(Point((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2),
        Point(botRight.x, botRight.y), level + 1, z);
    this->childTrees.push_back(child);
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