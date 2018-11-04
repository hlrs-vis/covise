/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#include "StdAfx.h"
#include "Plane.h"
#include <stdio.h>
#include <complex>

using namespace std;
Plane::Plane()
{
    //default refence nodes setting
    _referenceNode[0].x = 2;
    _referenceNode[0].y = -2;
    _referenceNode[0].z = -2;

    _referenceNode[1].x = 0;
    _referenceNode[1].y = -2;
    _referenceNode[1].z = -2;

    _referenceNode[2].x = -2;
    _referenceNode[2].y = -2;
    _referenceNode[2].z = -2;

    _referenceNode[3].x = -2;
    _referenceNode[3].y = -2;
    _referenceNode[3].z = 0;

    _referenceNode[4].x = -2;
    _referenceNode[4].y = -2;
    _referenceNode[4].z = 2;

    _referenceNode[5].x = 0;
    _referenceNode[5].y = -2;
    _referenceNode[5].z = 2;

    _referenceNode[6].x = 2;
    _referenceNode[6].y = -2;
    _referenceNode[6].z = 2;

    _referenceNode[7].x = 2;
    _referenceNode[7].y = -2;
    _referenceNode[7].z = 0;
}
void Plane::controllPlaneCreating()
{
    //The following codes creat the datas of the Controll Plane
    float xStart = -1;
    float yStart = -1;
    float zStart = 0;
    float width = 2;
    float length = 2;
    int numberOfRows = 4;
    int numberOfColumns = 4;

    float x = xStart;
    float y = yStart;
    float z = zStart;
    int rowMax = 2 * numberOfRows, colMax = 2 * numberOfColumns;
    bool flag1 = true, flag2 = true;
    int node_Num = 1;

    for (int row = 0; row <= rowMax; row++)
    {

        for (int column = 0; column <= colMax; column++)
        {
            Node newNode;
            newNode.x = x;
            newNode.y = y;
            newNode.z = z;

            x = x + length / (2 * numberOfColumns);
            if (flag1)
            {
                newNode.number = node_Num;
                node_Num++;
            }
            else
            {
                if (flag2)
                {
                    newNode.number = node_Num;
                    flag2 = false;
                    node_Num++;
                }
                else
                {
                    newNode.number = -1;
                    flag2 = true;
                }
            }
            _nodeList.push_back(newNode);
        }

        if (flag1)
            flag1 = false;
        else
            flag1 = true;
        flag2 = true;
        y = y + width / (2 * numberOfRows);
        x = xStart;
    }

    int elemNum = 1;
    for (int row = 0; row < numberOfRows; row += 1)
    {
        for (int column = 0; column < numberOfColumns; column += 1)
        {
            Element newElement;
            newElement.number = elemNum;

            newElement.numberofNode[0] = row * (3 * numberOfColumns + 2) + 2 * column + 1;
            newElement.numberofNode[1] = row * (3 * numberOfColumns + 2) + 2 * column + 1 + 1;
            newElement.numberofNode[2] = row * (3 * numberOfColumns + 2) + 2 * column + 2 + 1;
            newElement.numberofNode[3] = (row + 1) * (2 * numberOfColumns + 1) + row * (numberOfColumns + 1) + column + 1;
            newElement.numberofNode[4] = (row + 1) * (2 * numberOfColumns + 1) + row * (numberOfColumns + 1) + column + 1 + 1;
            newElement.numberofNode[5] = (row + 1) * (3 * numberOfColumns + 2) + 2 * column + 1;
            newElement.numberofNode[6] = (row + 1) * (3 * numberOfColumns + 2) + 2 * column + 1 + 1;
            newElement.numberofNode[7] = (row + 1) * (3 * numberOfColumns + 2) + 2 * column + 2 + 1;

            _elementList.push_back(newElement);
            elemNum++;
        }
    }
}
void Plane::targetPlaneCreating()
{
    //The following codes creat the datas of the Target Plane
    float originalX, originalY;
    list<Node>::iterator nodeList_Iter;
    nodeList_Iter = _nodeList.begin();
    for (int i = 0; i <= (int)_nodeList.size() - 1; i++)
    {
        originalX = nodeList_Iter->x;
        originalY = nodeList_Iter->y;
        Node tempNodePtr;
        //tempNodePtr = (*nodeList_Iter);
        xiEtaToXYZ(originalX, originalY, &(*nodeList_Iter));
        nodeList_Iter++;
    }
}
Plane::~Plane(void)
{
}

void Plane::moveTo(float /*x*/, float /*y*/, float /*z*/)
{
    /*int rowMax=2*numberOfRows,colMax=2*numberOfColumns;
	for(int row=0;row<=rowMax;row++)
	{
	  for(int column=0;column<=colMax;column++)
	  {
		  if(nodeCollector[column][row].number!=-1)
		  {
		  nodeCollector[column][row].x+=x;
          nodeCollector[column][row].y+=y;
		  nodeCollector[column][row].z+=z;
		  }
	   }  
	  
	}*/
}

void Plane::rotate(float grad)
{
    (void)grad;
    //float pi = 3.14159265359;
    //   int rowMax=2*numberOfRows,colMax=2*numberOfColumns;
    //for(int row=0;row<=rowMax;row++)
    //{
    //  for(int column=0;column<=colMax;column++)
    //  {
    //	  if(nodeCollector[column][row].number!=-1)
    //	  {
    //	  complex <float> tempComplex1 ( nodeCollector[column][row].x, nodeCollector[column][row].y );
    //	  complex <float> tempComplex2 = polar(abs(tempComplex1),arg(tempComplex1)+grad*pi/180);
    //	  nodeCollector[column][row].x=real(tempComplex2);
    //         nodeCollector[column][row].y=imag(tempComplex2);
    //	  //nodeCollector[column][row].z+=z;
    //	  }
    //   }
    //
    //}
}
void Plane::xiEtaToXYZ(float xi, float eta, Node *myNodePtr)
{
    myNodePtr->x = 0;
    myNodePtr->y = 0;
    myNodePtr->z = 0;

    //cout << "xi=" << xi << "\teta=" << eta << "\n";

    float formValue[8];

    shapeFunction(xi, eta, formValue);
    for (int i = 0; i <= 7; i++)
    {
        myNodePtr->x += formValue[i] * _referenceNode[i].x;
        myNodePtr->y += formValue[i] * _referenceNode[i].y;
        myNodePtr->z += formValue[i] * _referenceNode[i].z;
    }
}
void Plane::shapeFunction(float xi, float eta, float formValue[8])
{
    formValue[7] = 1 - xi;
    formValue[1] = 1 - eta;
    formValue[0] = 0.25f * formValue[7] * formValue[1] * (-1 - eta - xi);
    formValue[4] = 0.25f * (1 + xi);
    formValue[2] = formValue[4] * formValue[1] * (-1 - eta + xi);
    formValue[6] = 1 + eta;
    formValue[4] *= formValue[6] * (-1 + eta + xi);
    formValue[5] = 0.5f * (1 - xi * xi);
    formValue[1] *= formValue[5];
    formValue[5] *= formValue[6];
    formValue[6] *= 0.25f * formValue[7] * (-1 + eta - xi);
    formValue[3] = 0.5f * (1 - eta * eta);
    formValue[7] *= formValue[3];
    formValue[3] *= 1 + xi;
}
