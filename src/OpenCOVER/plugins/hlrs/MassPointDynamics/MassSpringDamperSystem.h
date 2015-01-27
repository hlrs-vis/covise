/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MassSpringDamperSystem_h
#define MassSpringDamperSystem_h

#include "LinAlgSupport.h"
#include <vector>

class MassPoint
{
public:
    MassPoint()
    {
    }
    MassPoint(const Vec3d &, const Vec3d &);
    MassPoint(const Vec3d &, const Vec3d &, const Vec3d &);

    MassPoint operator+(const MassPoint &) const;

    const MassPoint &operator+=(const MassPoint &);

    MassPoint operator*(double) const;

    MassPoint dstate(double);

    void addForce(const Vec3d &);

    static double m;

    Vec3d r;
    Vec3d v;

private:
    Vec3d force;
};

class Joint
{
public:
    virtual void applyTo(std::vector<MassPoint> *) = 0;
};

class SpringDamperJoint : public Joint
{
public:
    SpringDamperJoint(unsigned int, unsigned int, std::vector<MassPoint> *);

    void applyTo(std::vector<MassPoint> *);

private:
    unsigned int A;
    unsigned int B;

    double l;
    double k;

    static const double d = 10000;
    //static const double E = 10e6;
    static const double K = 1.3e7;
};

class GroundContactJoint : public Joint
{
public:
    GroundContactJoint(unsigned int);

    void applyTo(std::vector<MassPoint> *);

private:
    unsigned int P;

    static const double mu = 500000;
    static const double k = 1000000;
    static const double d = 50000;
};

class GravityJoint : public Joint
{
public:
    GravityJoint(unsigned int);

    void applyTo(std::vector<MassPoint> *);

private:
    unsigned int P;

    static const double g = 9.81;
};

class MassPointSystem : public std::vector<MassPoint>
{
public:
    void buildDoubleTetrahedron(const Vec3d &, const Vec3d &, const Vec3d &);

    void buildBall(const Vec3d &, const Vec3d &, const Vec3d &, double);

    std::vector<MassPoint> *getMassPointVector();
    std::vector<Joint *> *getJointVector();

protected:
    std::vector<MassPoint> massPointVector;
    std::vector<Joint *> jointVector;
};

/*class MassSpringDamperSystem
{
public:
   MassSpringDamperSystem(unsigned int num);
   ~MassSpringDamperSystem();

   MassSpringDamperSystem operator+(const MassSpringDamperSystem&) const;
   const MassSpringDamperSystem& operator+=(const MassSpringDamperSystem&);

   MassSpringDamperSystem operator*(double) const;

   const MassSpringDamperSystem& operator=(const MassSpringDamperSystem&);

   MassSpringDamperSystem dstate(double) const;

   void applyExternalForces() const;

   void addMassPoint(MassPoint*);
   MassPoint* getMassPoint(unsigned int) const;

   void addJoint(Joint*);

   int getMassPointVectorSize() const;

protected:
   std::vector<MassPoint*> massPointVector;
   unsigned int firstFreeElement;

   std::vector<Joint*> jointVector;
};*/

#endif
