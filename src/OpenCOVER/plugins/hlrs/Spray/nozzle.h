/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NOZZLE_H
#define NOZZLE_H

namespace opencover
{
class coVR3DTransRotInteractor;
}
using namespace opencover;

#include <string>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coVRTui.h>
#include <osgDB/ReadFile>
#include <osg/Matrix>
#include <osg/Node>
#include <osg/ShapeDrawable>
#include <osg/BoundingBox>
#include <osg/Group>
#include <osg/Node>
#include <osg/Geode>

#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>

#include "gen.h"
#include "types.h"

#include "parser.h"

#define posToCount(x,y,s) (3*x+y*s)
#define getMin(x,y) x<y?x:y

class nozzle : public coVR3DTransRotInteractor
{
private:

    int counter = 0;
    int nozzleID = 0;

    int prevEmissionRate = 0;

    float initPressure_ = 2;
    float minimum = 0.000025;
    float deviation = 0.00005;
    float alpha = 0.4;

    osg::ref_ptr<osg::MatrixTransform> transform_;
    osg::Group* interactorGroup;
    osg::Geode* geode_;
    osg::Geode* nozzleGeode;
    osg::Vec3 boundingBox_ = osg::Vec3(200,200,200);
    osg::Cone* cone_;
    osg::ShapeDrawable* shapeDrawable_;
    osg::Vec4 nozzleColor = osg::Vec4(1,1,1,1);
    osg::Vec4 currentColor_ = osg::Vec4(1,1,0,1);
    osg::MatrixTransform* nozzleScale;

    void updateColor();

    bool initialized = false;
    bool labelRegistered = false;
    bool displayed = true;

    std::string param1 = "none";
    std::string param2 = "none";
    std::string type = "none";

    bool intersection = true;


protected:
    void createGeometry();
    int particleCount_ = 1000;
    float autoremoveCount = 0.9;
    std::string nozzleName_;
    std::list<class gen*> genList;

    bool failed = false;
public:

    nozzle(osg::Matrix initialMat, float size, std::string nozzleName);

    virtual ~nozzle();
    void display(bool state);

    virtual void createGen();

    void updateGen();

    void setColor(osg::Vec4 newColor){
        currentColor_ = newColor;
        updateColor();
    }

    osg::Vec4 getColor(){
        return currentColor_;
    }

    void setInitPressure(float newPressure)
    {
        initPressure_ = newPressure;
    }

    float getInitPressure()
    {
        return initPressure_;
    }

    void resizeBoundingBox(osg::Vec3 newSize){
        boundingBox_ = newSize;
    }

    virtual void save(std::string pathName, std::string fileName);

    void setID(int ID);
    int getID();

    std::string getName()
    {
        return nozzleName_;

    }

    void registerLabel()
    {
        labelRegistered = true;
    }

    bool isRegistered()
    {
        return labelRegistered;
    }

    void setNozzleColor(osg::Vec4 newColor)
    {
        nozzleColor = newColor;
        shapeDrawable_->setColor(nozzleColor);
    }

    bool isFailed()
    {
        return failed;
    }

    float getMinimum()
    {
        return minimum;
    }

    float getDeviation()
    {
        return deviation;
    }

    void setMinimum(float newMinimum)
    {
        minimum = newMinimum;
    }

    void setDeviation(float newDeviation)
    {
        deviation = newDeviation;
    }

    void keepSize();


    osg::MatrixTransform* getMatrixTransform()
    {
        return transform_->asMatrixTransform();
    }

    std::string getParam1()
    {
        return param1;
    }

    std::string getParam2()
    {
        return param2;
    }

    std::string getType()
    {
        return type;
    }

    void setParam1(std::string newParam1)
    {
        param1 = newParam1;
    }

    void setParam2(std::string newParam2)
    {
        param2 = newParam2;
    }

    void setType(std::string newType)
    {
        type = newType;
    }

    void autoremove(bool state);

    void setIntersection(bool state)
    {
        intersection = state;
    }

    bool getIntersection()
    {
        return intersection;
    }

    void setAlpha(float newAlpha)
    {
        alpha = newAlpha;
    }

    float getAlpha()
    {
        return alpha;
    }

    void setNozzleGeometryNode(osg::Node* node);

    bool setNozzleGeometryFile(std::string filename)
    {
        std::ifstream mystream(filename);
        //std::string filename2 = "test.txt";
        std::string line = "";
        std::string name = "";
        osg::Vec3Array* coordArray = new osg::Vec3Array();
        osg::Vec3Array* normalsArray = new osg::Vec3Array();
        osg::Vec3 coords = osg::Vec3(0,0,0);
        osg::Vec3 normals = osg::Vec3(0,0,0);

        if(mystream.is_open())
        {
            printf("File opened!\n");
            std::cout << filename << std::endl;
            while(std::getline(mystream,line))
            {

                std::cout << line << std::endl;
                int x = 0;
                    while(line[x] == ' ')
                         x++;
                    line.erase(0,x);
                std::stringstream ss(line);
                std::getline(ss,line,' ');

                if(line.compare("solid") == 0)
                {
                    getline(ss,line, '\n');
                    name = line;
                    std::cout << "My name is " << name << std::endl;
                }

                if(line.empty())
                    continue;

                if(line.compare("facet") == 0)
                {
                    getline(ss,line, ' ');
                    getline(ss,line, ' ');
                    normals.x() = stof(line);
                    getline(ss,line, ' ');
                    normals.y() = stof(line);
                    getline(ss,line, '\n');
                    normals.z() = stof(line);

                    printf("Normals of geometry %f %f %f \n", normals.x(), normals.y(), normals.z());
                    normalsArray->push_back(normals);
                    osg::Vec3Array::iterator itr = normalsArray->end();
                    printf("Vertices of geometry %f %f %f \n", (*itr).x(), (*itr).y(), (*itr).z());
                    continue;
                }

                if(line.compare("outer") == 0)
                    continue;

                if(line.compare("vertex") == 0)
                {
                    getline(ss,line, ' ');
                    coords.x() = stof(line);
                    getline(ss,line, ' ');
                    coords.y() = stof(line);
                    getline(ss,line, '\n');
                    coords.z() = stof(line);
                    printf("Vertices of geometry %f %f %f \n", coords.x(), coords.y(), coords.z());
                    coordArray->push_back(coords);
                    osg::Vec3Array::iterator itr = coordArray->end();
                    printf("Vertices of geometry %f %f %f \n", (*itr).x(), (*itr).y(), (*itr).z());

                    //TODO: Check why values don't apply


                    continue;
                }

                if(line.compare("endloop") == 0)
                    continue;

                if(line.compare("endfacet") == 0)
                    continue;

                if(line.compare("endsolid") == 0)
                    continue;

            }//mystream

            //cover->getObjectsRoot()->addChild(nozzleGeode);
        }
        else
            return false;

        mystream.close();

//        nozzleGeode = new osg::Geode();
//        scaleTransform->addChild(nozzleGeode);
//        osg::Geometry* geom = new osg::Geometry();

//        geom->setVertexArray(coordArray);

//        osg::Vec4Array *colors = new osg::Vec4Array;
//        colors->push_back(osg::Vec4(1,1,1,1));
//        geom->setColorArray(colors);
//        geom->setColorBinding(osg::Geometry::BIND_OVERALL);

//        osg::Vec3Array *normals = new osg::Vec3Array;
//        normals->push_back(osg::Vec3(0,0,-1));
//        geom->setNormalArray(normals);
//        geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

//        std::cout << coordArray->size() << std::endl;

//        geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,coordArray->size()));

//        geode_->addDrawable(geom);

//        return true;

        osg::Geometry* test = new osg::Geometry();
        osg::Vec3Array* jo = new osg::Vec3Array();

        jo->push_back(osg::Vec3(1,1,1));
        jo->push_back(osg::Vec3(1,-1,1));
        jo->push_back(osg::Vec3(1,1,-1));
        jo->push_back(osg::Vec3(-1,-1,-1));
        jo->push_back(osg::Vec3(-1,1,-1));
        jo->push_back(osg::Vec3(-1,-1,1));

        test->setVertexArray(coordArray);

        osg::Vec3Array *normalsA = new osg::Vec3Array;
        normalsA->push_back(osg::Vec3(0,0,-1));
        test->setNormalArray(normalsA);
        test->setNormalBinding(osg::Geometry::BIND_OVERALL);

        osg::Vec4Array *colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(1,1,1,1));
        test->setColorArray(colors);
        test->setColorBinding(osg::Geometry::BIND_OVERALL);

        test->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,coordArray->size()));
        geode_->addDrawable(test);

        //      Pattern of a STL file
        //        solid name
        //         facet normal n1 n2 n3
        //          outer loop
        //           vertex p1x p1y p1z
        //           vertex p2x p2y p2z
        //           vertex p3x p3y p3z
        //          endloop
        //         endfacet
        //        endsolid name
    }
};



class standardNozzle : public nozzle
{
private:
    float sprayAngle_ = 0;
    std::string decoy_;
public:
    standardNozzle(float sprayAngle, std::string decoy, osg::Matrix initialMat, float size, std::string nozzleName);
    ~standardNozzle(){}

    void createGen();
    void save(std::string pathName, std::string fileName);

    float getSprayAngle(){
        return sprayAngle_;
    }

    std::string getDecoy(){
        return decoy_;
    }
};

class imageNozzle : public nozzle
{
private:
    std::string fileName_;
    std::string pathName_;

    pImageBuffer iBuf;

    int samplingPoints = 1000;

    float pixel_to_mm_;
    float pixel_to_flow_;
    int colorDepth_;
    int colorThreshold_ = 100;


    bool square = false;
    bool circle = true;
    bool readImage();

public:

    imageNozzle(std::string pathName, std::string fileName, osg::Matrix initialMat, float size, std::string nozzleName);
    ~imageNozzle(){}

    void createGen();
    void save(std::string pathName, std::string fileName);

    pImageBuffer getPImageBuffer()
    {
        return iBuf;
    }

    std::string getParam1()
    {
        return pathName_;
    }

    std::string getParam2()
    {
        return fileName_;
    }

};

#endif // NOZZLE_H
