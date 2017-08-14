/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HumanVehicleGeometry.h"

#include <limits>
#include <cover/coVRPluginSupport.h>
using namespace opencover;

HumanVehicleGeometry::HumanVehicleGeometry(std::string nodeName)
    : vehicleNodeName(nodeName)
    , vehicleNode(NULL)
    , boundingCircleRadius(0.0)
    , vrmlTransform(Quaternion(M_PI * 0.5, Vector3D(1.0, 0.0, 0.0)))
    , vehicleTransform(Quaternion(M_PI * 0.5, Vector3D(0.0, 1.0, 0.0)) * Quaternion(-M_PI * 0.5, Vector3D(1.0, 0.0, 0.0)))
{

    /*osg::Matrix mat = cover->getObjectsXform()->getMatrix();
   osg::Vec3 translation = mat.getTrans();
   osg::Quat rotation = mat.getRotate();
   Transform objectTransform( Vector3D(translation.x(), translation.y(), translation.z()),
                              Quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z()) );*/
    //vrmlTransform = objectTransform * vrmlTransform;

    findVehicleNode();
}

HumanVehicleGeometry::~HumanVehicleGeometry()
{
}

void HumanVehicleGeometry::setTransform(Transform &, double)
{
}

double HumanVehicleGeometry::getBoundingCircleRadius()
{
    return boundingCircleRadius;
}

const osg::Matrix &HumanVehicleGeometry::getVehicleTransformMatrix()
{
    if (!vehicleNode)
    {
        findVehicleNode();
    }
    if (vehicleNode)
    {
        osg::Transform *transform = vehicleNode->asTransform();
        if (transform)
        {
            osg::MatrixTransform *matTrans = transform->asMatrixTransform();
            if (matTrans)
            {
                return matTrans->getMatrix();
            }
        }
    }

    return cover->getXformMat();
}

Transform HumanVehicleGeometry::getVehicleTransformation()
{
    if (!vehicleNode)
    {
        findVehicleNode();
    }
    if (vehicleNode)
    {
        osg::Transform *transform = vehicleNode->asTransform();
        if (transform)
        {
            osg::MatrixTransform *matTrans = transform->asMatrixTransform();
            if (matTrans)
            {
                //osg::Matrix mat = matTrans->getMatrix();
                //osg::Vec3 trans = mat.getTrans();
                //osg::Quat rot = mat.getRotate();
                //Vector3D transVec(trans.x(), trans.y(), trans.z());
                //transVec = (vrmlTransform * transVec * vrmlTransform.T()).getVector();
                //Quaternion transQuat(rot.w(), rot.x(), rot.y(), rot.z());
                //transQuat = vrmlTransform.T() * transQuat * vrmlTransform.T();
                //return Transform(transVec, transQuat);

                osg::Matrix mat = matTrans->getMatrix();
                osg::Vec3 translation = mat.getTrans();
                osg::Quat rotation = mat.getRotate();

                Transform trans(Vector3D(translation.x(), translation.y(), translation.z()),
                                Quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z()));

                /*std::cout << "vrmlTransform: " << std::endl; vrmlTransform.print();
            std::cout << "vehicleTransform: " << std::endl; vehicleTransform.print();
            std::cout << "trans: " << std::endl; trans.print();
            Transform doneTransform = vrmlTransform*trans*vehicleTransform;
            std::cout << "doneTransform: " << std::endl; doneTransform.print();
            std::cout << "doneTransform dir: " << (doneTransform.q()*Vector3D(1.0,0.0,0.0)*doneTransform.q().T()).getVector();
            //std::cout << std::endl;*/

                return vrmlTransform * trans * vehicleTransform;
            }
        }
    }
    return Transform(Vector3D(std::numeric_limits<float>::signaling_NaN()), Quaternion(std::numeric_limits<float>::signaling_NaN()));
}

void HumanVehicleGeometry::findVehicleNode()
{
    if (!vehicleNode)
    {
        vehicleNode = searchGroupByVehicleNodeName(cover->getObjectsXform());
        if (vehicleNode)
        {
            osg::Vec3 trans(0.0, 0.0, 0.0);
            osg::Transform *transform = vehicleNode->asTransform();
            if (transform)
            {
                osg::MatrixTransform *matTrans = transform->asMatrixTransform();
                if (matTrans)
                {
                    osg::Matrix mat = matTrans->getMatrix();
                    trans = mat.getTrans();
                }
            }
            osg::BoundingSphere boundingSphere = vehicleNode->computeBound();
            boundingCircleRadius = (boundingSphere.center() - trans).length() + boundingSphere.radius();
        }
    }
}

osg::Node *HumanVehicleGeometry::searchGroupByVehicleNodeName(osg::Group *group)
{
    for (unsigned int groupIt = 0; groupIt < group->getNumChildren(); ++groupIt)
    {
        osg::Node *node = group->getChild(groupIt);
        if (node)
        {
            if (node->getName().find(vehicleNodeName) != vehicleNodeName.npos)
            {
                return node;
            }
            else
            {
                osg::Group *childGroup = dynamic_cast<osg::Group *>(node);
                if (childGroup)
                {
                    osg::Node *childNode = searchGroupByVehicleNodeName(childGroup);
                    if (childNode)
                    {
                        return childNode;
                    }
                }
            }
        }
    }
    return NULL;
}
