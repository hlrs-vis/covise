/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CELL_MAKER_H_
#define _CELL_MAKER_H_

// Inspace:
#include <osgDrawObj.H>

// VTK:
#include "vtk/vtkActor.h"
#include "vtk/vtkActorCollection.h"
#include "vtk/vtkRenderWindow.h"

// OSG:
#include <osg/Geode>

//local
#include "VolumePickBox.H"

class CellMaker : public VolumePickBox
{
public:
    class SphereData
    {
    public:
        osg::Node *_node;

        float x;
        float y;
        float z;

        float radius;

        int markersInside;

        SphereData()
        {
            markersInside = 0;
        };
        void print()
        {
            cerr << " Sphere at x=" << x << ", y=" << y << ", z=" << z << ", radius=" << radius << endl;
        }
    };

    CellMaker(osgDrawObj *, cui::Interaction *);
    virtual ~CellMaker();
    void init();
    void displaySphere(SphereData *);
    void setNumCells(int num);
    int getNumCells()
    {
        return _numCells;
    }
    void setExtremes(float min, float max);
    void setColorRGB(osg::Vec4);
    void setColorHueToRGB(float hue);
    void showCells(bool);
    void trackballRotation(float, float, osg::Matrix &);
    void cursorUpdate(cui::InputDevice *dev);
    void updateSpheresColor();
    bool writeSpheresFile(const char *);
    bool readSpheresFile(const char *);
    osg::Node *makeOSGSphere(SphereData *);
    void addMarkerByHand(cui::Marker *);
    void checkMarkerCollisions();
    SphereData *collisionDetection(cui::Marker *);

private:
    osgDrawObj *_osgObj;
    std::vector<SphereData *> _mSpheres;

    FILE *_fp;
    int _numCells;
    float _minRadius;
    float _maxRadius;
    float _r;
    float _g;
    float _b;

    SphereData *makeRandomCell(float, float, float, float, float, float);
    osg::Vec3 getRandomPosition(float, float, float, float, float, float, float);
    osg::Vec3 getCollisionsFreeRandomPosition(float, float, float, float, float, float, float);
    bool isCollision(osg::Vec3, float, osg::Vec3, float);
    float getDistance(osg::Vec3, osg::Vec3);
    double getRandom(float min, float max);
    vtkActor *makeCell(SphereData *);
    void addSphere(double x, double y, double z, float radius);
    //    void writePositionAndRadii(std::string fileName);
    //    void writeOBJ(std::string dataSetName);
};

class DistanceVisitor : public osg::NodeVisitor
{ // traverses the scene graph to get the z (distance) value of each sphere
private:
    vector<float> _distances;
    int cyclesMT;
    osg::Matrix _baseMat; // Matrix of _scale node multiplied by the world root matrix

public:
    DistanceVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        cyclesMT = 0;
    }

    vector<float> getDistances()
    {
        return _distances;
    }

    virtual void apply(osg::Node &node)
    {
        traverse(node);
    }

    virtual void apply(osg::Geode &geode)
    { // compute local matrices to store z (distance) value
        if (geode.getNumDrawables() == 1)
        {
            osg::Drawable *sphereDrawable = geode.getDrawable(0);
            osg::Sphere *sphereShape = dynamic_cast<osg::Sphere *>(sphereDrawable->getShape());
            if (sphereShape)
            {
                osg::Matrix trans;
                trans.setTrans(sphereShape->getCenter());
                osg::MatrixTransform *scaleNode = dynamic_cast<osg::MatrixTransform *>(geode.getParent(0));
                if (scaleNode)
                {
                    osg::Matrix local2root;
                    local2root = trans * (scaleNode->getMatrix() * _baseMat);
                    _distances.push_back(local2root.getTrans().z());
                }
            }
        }
    }

    virtual void apply(osg::MatrixTransform &node)
    { // compute local matrices -- main part now is done in geodes (trans. part is stored in the spheres)
        cyclesMT++;
        if (cyclesMT == 1) // try to get (only) the _scale node by node.getNumParents() instead of checking the cycles
        {
            osg::MatrixTransform *transfNode = dynamic_cast<osg::MatrixTransform *>(node.getParent(0));
            if (transfNode)
            {
                _baseMat = node.getMatrix() * transfNode->getMatrix();
            }
        }
        /*else    
          {//cerr << node.getMatrix() << _baseMat << endl;
            osg::Matrix local2root;
            if (osgDrawObj::computeLocal2Root(&node, local2root))
              _distances.push_back(local2root.getTrans().z());
          }*/
        apply((osg::Node &)node);
    }
};

class SpheresColorVisitor : public osg::NodeVisitor
{ // traverses the scene graph to color each sphere with the distance scale made by the DistanceVisitor
private:
    int _geodeIndex;
    float _minZ;
    float _maxZ;
    vector<float> _distances;
    list<float> _sortedDistances;
    int cyclesN;
    int cyclesG;

public:
    SpheresColorVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        _geodeIndex = -1;
        cyclesN = 0;
        cyclesG = 0;
    }

    void setDistances(vector<float> distances)
    {
        _distances = distances;

        vector<float>::iterator iter;
        int i;
        for (iter = _distances.begin(), i = 0; iter != _distances.end(); ++iter, i++)
        {
            _sortedDistances.push_back(_distances[i]);
        }
        _sortedDistances.sort();
        _minZ = _sortedDistances.front();
        _maxZ = _sortedDistances.back();
    }

    osg::Vec4 computeColor(float z)
    { // change this algorithm as required to compute the color from the z (distance) value
        // currently: scale from light to dark (near to far) for green base color
        float colorScaling = (z - _maxZ) / (_minZ - _maxZ);
        //cerr << z << " " << _minZ << " " << _maxZ << " " << colorScaling << endl;
        return osg::Vec4(0.0f, colorScaling, 0.0f, 1.0f);
    }

    virtual void apply(osg::Node &node)
    {
        traverse(node);
    }

    virtual void apply(osg::Geode &geode)
    {
        if (geode.getNumDrawables() == 1)
        {
            osg::ShapeDrawable *sphereDrawable = dynamic_cast<osg::ShapeDrawable *>(geode.getDrawable(0));
            if (sphereDrawable)
            {
                //cyclesG++;
                //cerr << "G: " << cyclesG << endl;
                _geodeIndex++;
                sphereDrawable->setColor(computeColor(_distances[_geodeIndex]));
            }
        }
    }
};

#endif
