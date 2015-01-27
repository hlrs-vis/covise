/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MassSpringDamperSystem.h"
#include <iostream>
#include <cstdlib>

double MassPoint::m = 3500.0;

MassPoint::MassPoint(const Vec3d &rset, const Vec3d &vset)
    : r(rset)
    , v(vset)
    , force(0.0, 0.0, 0.0)
{
}

MassPoint::MassPoint(const Vec3d &rset, const Vec3d &vset, const Vec3d &forceset)
    : r(rset)
    , v(vset)
    , force(forceset)
{
}

MassPoint MassPoint::operator+(const MassPoint &mp) const
{
    return MassPoint(r + mp.r, v + mp.v);
}

const MassPoint &MassPoint::operator+=(const MassPoint &mp)
{
    r += mp.r;
    v += mp.v;
    return *this;
}

MassPoint MassPoint::operator*(double h) const
{
    return MassPoint(r * h, v * h);
}

MassPoint MassPoint::dstate(double)
{
    MassPoint dmp(v, force * (1 / m));
    force.x = 0;
    force.y = 0;
    force.z = 0;
    return dmp;
}

void MassPoint::addForce(const Vec3d &addForce)
{
    force += addForce;
}

SpringDamperJoint::SpringDamperJoint(unsigned int Aset, unsigned int Bset, std::vector<MassPoint> *mpVec)
    : A(Aset)
    , B(Bset)
    , l(((*mpVec)[Aset].r - (*mpVec)[Bset].r).norm())
    , k(K / l)
{
}

void SpringDamperJoint::applyTo(std::vector<MassPoint> *mpVec)
{
    Vec3d rAB = (*mpVec)[B].r - (*mpVec)[A].r;
    Vec3d vAB = (*mpVec)[B].v - (*mpVec)[A].v;
    double rABSquareNorm = rAB.squarenorm();

    double fNorm = k * (1 - (l * l) / (rABSquareNorm)) + d * (vAB * rAB / rABSquareNorm);

    (*mpVec)[A].addForce(rAB * fNorm);
    (*mpVec)[B].addForce(rAB * (-fNorm));

    Vec3d rABfNorm = rAB * fNorm;
    /*std::cout << "A: (" << pointA->getR().x << ", " << pointA->getR().y << ", " << pointA->getR().z << ")" << std::endl;
   std::cout << "B: (" << pointB->getR().x << ", " << pointB->getR().y << ", " << pointB->getR().z << ")" << ", dAB: " << sqrt(rABSquareNorm) << std::endl;
   std::cout << "Force: " << fNorm << ", Vec: " << rABfNorm.x << ", " << rABfNorm.y << ", " << rABfNorm.z << std::endl;*/
}

GroundContactJoint::GroundContactJoint(unsigned int setP)
    : P(setP)
{
}

void GroundContactJoint::applyTo(std::vector<MassPoint> *mpVec)
{
    if ((*mpVec)[P].r.z < 0.0)
    {
        (*mpVec)[P].addForce(Vec3d(mu * (*mpVec)[P].r.z * tanh((*mpVec)[P].v.x), mu * (*mpVec)[P].r.z * tanh((*mpVec)[P].v.y), -(k * pow((*mpVec)[P].r.z, 3) + d * (*mpVec)[P].v.z)));
    }
}

GravityJoint::GravityJoint(unsigned int setP)
    : P(setP)
{
}

void GravityJoint::applyTo(std::vector<MassPoint> *mpVec)
{
    (*mpVec)[P].addForce(Vec3d(0.0, 0.0, -(g * MassPoint::m)));
}

void MassPointSystem::buildDoubleTetrahedron(const Vec3d &center, const Vec3d &vel, const Vec3d &angVel)
{
    massPointVector.push_back(MassPoint(center + Vec3d(0.0, 0.0, 4.082), vel + angVel.cross(Vec3d(0.0, 0.0, 4.082))));
    massPointVector.push_back(MassPoint(center + Vec3d(-1.443, -2.5, 0.0), vel + angVel.cross(Vec3d(-1.443, -2.5, 0.0))));
    massPointVector.push_back(MassPoint(center + Vec3d(-1.443, 2.5, 0.0), vel + angVel.cross(Vec3d(-1.443, 2.5, 0.0))));
    massPointVector.push_back(MassPoint(center + Vec3d(2.886, 0.0, 0.0), vel + angVel.cross(Vec3d(2.886, 0.0, 0.0))));
    massPointVector.push_back(MassPoint(center + Vec3d(0.0, 0.0, -4.082), vel + angVel.cross(Vec3d(0.0, 0.0, -4.082))));

    jointVector.push_back(new SpringDamperJoint(0, 1, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(0, 2, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(0, 3, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 2, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 3, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 4, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(2, 3, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(2, 4, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(3, 4, &massPointVector));
    for (unsigned int i = 0; i < massPointVector.size(); ++i)
    {
        jointVector.push_back(new GroundContactJoint(i));
        jointVector.push_back(new GravityJoint(i));
    }
}

void MassPointSystem::buildBall(const Vec3d &center, const Vec3d &vel, const Vec3d &angVel, double r)
{
    int quarterPoints = 3;
    int fullPoints = (quarterPoints * 4);
    int halfPoints = (quarterPoints * 2);
    double deltaVer = 2 * M_PI / fullPoints;

    //upper apex
    massPointVector.push_back(MassPoint(center + Vec3d(0.0, 0.0, r), vel));

    for (int ver = 1; ver < halfPoints; ++ver)
    {
        //int horNum = (pow(2, (quarterPoints - (abs(ver-quarterPoints))) - 1) * quarterPoints);
        int horNum = 4 * (quarterPoints - abs(ver - quarterPoints));
        double deltaHor = 2 * M_PI / horNum;
        int firstHor = massPointVector.size();
        for (int hor = 0; hor < horNum; ++hor)
        {
            massPointVector.push_back(MassPoint(center + Vec3d(r * sin(ver * deltaVer) * cos(hor * deltaHor), r * sin(ver * deltaVer) * sin(hor * deltaHor), r * cos(ver * deltaVer)), vel));
        }
    }

    //lower apex
    massPointVector.push_back(MassPoint(center + Vec3d(0.0, 0.0, -r), vel));

    jointVector.push_back(new SpringDamperJoint(0, 1, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(0, 2, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(0, 3, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(0, 4, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 2, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(2, 3, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(3, 4, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(4, 1, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(5, 6, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(6, 7, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(7, 8, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(9, 10, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(10, 11, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(11, 12, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(12, 5, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 12, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 5, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(1, 6, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(2, 6, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(2, 7, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(2, 8, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(3, 8, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(3, 9, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(3, 10, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(4, 10, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(4, 11, &massPointVector));
    jointVector.push_back(new SpringDamperJoint(4, 12, &massPointVector));

    for (unsigned int i = 0; i < massPointVector.size(); ++i)
    {
        jointVector.push_back(new GroundContactJoint(i));
        jointVector.push_back(new GravityJoint(i));
    }
}

std::vector<MassPoint> *MassPointSystem::getMassPointVector()
{
    return &massPointVector;
}

std::vector<Joint *> *MassPointSystem::getJointVector()
{
    return &jointVector;
}

/*MassSpringDamperSystem::MassSpringDamperSystem(unsigned int num)
   :  massPointVector(num),
      firstFreeElement(0)
{
}

MassSpringDamperSystem::~MassSpringDamperSystem()
{
   for(unsigned int i=0; i<firstFreeElement; ++i) {
      delete massPointVector[i];
   }
}

MassSpringDamperSystem MassSpringDamperSystem::operator+(const MassSpringDamperSystem& other) const
{
   MassSpringDamperSystem system(massPointVector.size());

   for(unsigned int i=0; i<firstFreeElement; ++i) {
      system.addMassPoint(new MassPoint(*massPointVector[i] + *other.getMassPoint(i)));
   }

   return system;
}

const MassSpringDamperSystem& MassSpringDamperSystem::operator+=(const MassSpringDamperSystem& other)
{
   for(unsigned int i=0; i<firstFreeElement; ++i) {
      *massPointVector[i] += *other.getMassPoint(i);
   }

   return *this;
}

MassSpringDamperSystem MassSpringDamperSystem::operator*(double h) const
{
   MassSpringDamperSystem system(massPointVector.size());
  
  for(unsigned int i=0; i<firstFreeElement; ++i) {
      system.addMassPoint((*massPointVector[i])*h);
   }

   return system;
}

const MassSpringDamperSystem& MassSpringDamperSystem::operator=(const MassSpringDamperSystem& other)
{
   for(unsigned int i=0; i<firstFreeElement; ++i) {
      *massPointVector[i] = *other.getMassPoint(i);
   }

   return *this;
}

void MassSpringDamperSystem::addMassPoint(MassPoint* mp)
{
   massPointVector[firstFreeElement] = mp;

   firstFreeElement++;
}

MassPoint* MassSpringDamperSystem::getMassPoint(unsigned int mpi) const
{
   return massPointVector[mpi];
}

void MassSpringDamperSystem::addJoint(Joint* joint)
{
   jointVector.push_back(joint);
}


int MassSpringDamperSystem::getMassPointVectorSize() const
{
   return massPointVector.size();
}

MassSpringDamperSystem MassSpringDamperSystem::dstate(double h) const
{
   MassSpringDamperSystem system(massPointVector.size());

   for(unsigned int i=0; i<firstFreeElement; ++i) {
      system.addMassPoint(massPointVector[i]->dstate(h));
   }

   return system;
}

void MassSpringDamperSystem::applyExternalForces() const
{
   for(unsigned int i=0; i<firstFreeElement; ++i) {
      massPointVector[i]->emptyExternalForce();
   }

   for(unsigned int i=0; i<jointVector.size(); ++i) {
      jointVector[i]->applyForce();
   }
}

*/
