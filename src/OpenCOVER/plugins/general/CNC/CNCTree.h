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
    int numChildren;            //0,1,2,3,4; -1 -> leafNode
    //int numDescendants;         //incorrect if insertion fails
    vector<TreeNode*> childTrees;
    vector<int> millTimesteps;  //the timesteps where this Node is milled
    
public:
    TreeNode(Point _topL, Point _botR, int _level)
    {
        level = _level;
        numChildren = 0;
        //numDescendants = 0;
        topLeft = _topL;
        botRight = _botR;
    }
    TreeNode(Point _topL, Point _botR, int _level, double _z, int _numChildren)
    {
        //nodeCoords.push_back(_coords);
        z = _z;
        level = _level;
        numChildren = _numChildren;
        //numDescendants = 0;
        topLeft = _topL;
        botRight = _botR;
    }
    /*
    Point getTopLeft();
    Point getBotRight();
    double getNumChildren();
    void setZ(double _z);
    double getZ();
    void setPrimitivePos(int _primitivePos);
    int getPrimitivePos();
    */
    void insert(int _ix, int _iy, double _z);
    void insert(int _ix, int _iy);
    TreeNode* search(Point);
    bool inBoundary(Point);
    /*
    bool topLeftBoundary(Point);
    bool updateZ(int _ix, int _iy, double _z);
    */
};
/*
inline Point TreeNode::getTopLeft()
{
    return topLeft;
}
inline Point TreeNode::getBotRight()
{
    return botRight;
}

inline double TreeNode::getNumChildren()
{
    return numChildren;
}

inline TreeNode* TreeNode::getTopLeftTree()
{
    return topLeftTree;
}
inline TreeNode* TreeNode::getTopRightTree()
{
    return topRightTree;
}
inline TreeNode* TreeNode::getBotLeftTree()
{
    return botLeftTree;
}
inline TreeNode* TreeNode::getBotRightTree()
{
    return botRightTree;
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
inline void TreeNode::insert(int _ix, int _iy, double _z)
{
  //  this->insert(new MillCoords(Point(_ix, _iy), _z));
    return;
}

// Insert a node into the quadtree
// if node already exist: no insertion performed
inline void TreeNode::insert(int _ix, int _iy)
{
    // Current quad cannot contain it
    if (!inBoundary(Point(_ix, _iy))
    {
        throw std::invalid_argument("TreeNode::insert, out of Boundary");
        return;
    }

    TreeNode* tree = search(Point(_ix, _iy));
/*
    // Actual insertion
    // We are at a quad of unit area
    // We cannot subdivide this quad further
    if (numChildren == 0)
    {
        for (MillCoords* coords : nodeCoords)
        {
            if (coords->pos.x == _millCoords->pos.x && coords->pos.y == _millCoords->pos.y)
                return coords;
        }
    }

    if ((topLeft.x + botRight.x) / 2 >= _millCoords->pos.x) {
        // Indicates topLeftTree
        if ((topLeft.y + botRight.y) / 2 >= _millCoords->pos.y) {
            if (topLeftTree == NULL)
            {
                topLeftTree = new TreeNode(Point(topLeft.x, topLeft.y),
                    Point((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2), this->level + 1);
                numChildren++;
            }
            topLeftTree->insert(_millCoords);
            numDescendants++;
        }

        // Indicates botLeftTree
        else {
            if (botLeftTree == NULL)
            {
                botLeftTree = new TreeNode(Point(topLeft.x, (topLeft.y + botRight.y) / 2), 
                    Point((topLeft.x + botRight.x) / 2, botRight.y), this->level + 1);
                numChildren++;
            }
            botLeftTree->insert(_millCoords);
            numDescendants++;
        }
    }
    else {
        // Indicates topRightTree
        if ((topLeft.y + botRight.y) / 2 >= _millCoords->pos.y) {
            if (topRightTree == NULL)
            {
                topRightTree = new TreeNode(Point((topLeft.x + botRight.x) / 2, topLeft.y),
                    Point(botRight.x, (topLeft.y + botRight.y) / 2), this->level + 1);
                numChildren++;
            }
            topRightTree->insert(_millCoords);
            numDescendants++;
        }

        // Indicates botRightTree
        else {
            if (botRightTree == NULL)
            {
                botRightTree = new TreeNode(Point((topLeft.x + botRight.x) / 2, (topLeft.y + botRight.y) / 2),
                    Point(botRight.x, botRight.y),this->level+1);
                numChildren++;
            }
            botRightTree->insert(_millCoords);
            numDescendants++;
        }
    }
    return;
}
*/

// Find a node in a quadtree
inline TreeNode* TreeNode::search(Point p)
{
    // Current quad cannot contain it
    if (!inBoundary(p))
        return NULL;

    // check if no children
    if (numChildren == 0)
    {
        return this;
    }

    for (TreeNode* childTree : childTrees)
    {
        if (childTree->inBoundary(p))
            childTree->search(p);
    }

};

// Check if current quadtree contains the point
inline bool TreeNode::inBoundary(Point p)
{
    return (p.x > topLeft.x && p.x <= botRight.x
        && p.y > topLeft.y && p.y <= botRight.y);
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