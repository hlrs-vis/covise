/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLANE_H_INCLUDED
#define PLANE_H_INCLUDED
#include "Node.h"
#include "Element.h"
#include <list>
using namespace std;

class Plane
{

public:
    //Note: The dimesion of arrays should be 1 bigger than the subindex of the arrays.
    /*Node nodeCollector[11][11];*/
    list<Node> _nodeList;
    /*Element elementCollector[25];*/
    list<Element> _elementList;
    Node _referenceNode[8];

private:
    void xiEtaToXYZ(float xi, float eta, Node *myNodePtr);
    void shapeFunction(float xi, float eta, float formValue[8]);

public:
    Plane();
    void controllPlaneCreating();
    void targetPlaneCreating();
    void moveTo(float x, float y, float z);
    void rotate(float grad);

public:
    ~Plane(void);
};
#endif
