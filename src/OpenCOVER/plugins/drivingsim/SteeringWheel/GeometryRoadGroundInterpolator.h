/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GeometryRoadGroundInterpolator_h
#define __GeometryRoadGroundInterpolator_h

#include "RoadSystem/RoadSystem.h"
#include "gaalet.h"

template <unsigned int HP>
class GeometryRoadGroundInterpolator
{
public:
    static const unsigned int NumHitPoints = HP;
    //vectors in osg reference frame
    typedef double VectorArrayType[NumHitPoints][3];

    typedef gaalet::algebra<gaalet::signature<3, 0> > em;
    typedef em::mv<1, 2, 4>::type Vec3d;

    GeometryRoadGroundInterpolator()
    {
        for (int i = 0; i < NumHitPoints; ++i)
        {
            currentRoad[i] = NULL;
            currentLongPos[i] = -1.0;
        }
        h_h = 0.02;
    }

    void init(const VectorArrayType &r)
    {
        for (int i = 0; i < NumHitPoints; ++i)
        {
            currentRoad[i] = NULL;
            currentLongPos[i] = -1.0;

            r_n[i][0] = r[i][0];
            r_n[i][1] = r[i][1];
            r_n[i][2] = r[i][2];
            r_o[i] = r_n[i];
            t_n[i] = Vec3d();
            t_o[i] = Vec3d();
        }
        h_h = 0.02;
    }

    std::pair<VectorArrayType, VectorArrayType> operator()(const VectorArrayType &rpArray, const double &dt)
    {
        Vec3d *r_h = static_cast<Vec3d *>(rhArray);
        Vec3d *n_h = static_cast<Vec3d *>(nhArray);

        for (int i = 0; i < NumHitPoints; ++i)
        {
            Vector2D v_c(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
            if (currentRoad[i])
            {
                Vector3D v_w(-rpArray[i][1], rpArray[i][0], rpArray[i][2]);

                //if(RoadSystem::Instance()) {
                v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad[i], currentLongPos[i]);
                //}
                if (!v_c.isNaV())
                {
                    RoadPoint point = currentRoad[i]->getRoadPoint(v_c.u(), v_c.v());
                    rhArray[i][0] = point.y();
                    rhArray[i][1] = -point.x();
                    rhArray[i][2] = point.z();
                    nhArray[i][0] = point.ny();
                    nhArray[i][1] = -point.nx();
                    nhArray[i][2] = point.nz();
                }
            }
            if (!currentRoad[i] || v_c.isNaV())
            {
                //double tau = (double)step*h/dt; if(tau>1.0) tau = 1.0;
                double tau = dt / h_h;
                if (tau > 1.0)
                    tau = 1.0;
                double ttau = tau * tau;
                double tttau = ttau * tau;
                r_h[i] = r_n[i] * (2.0 * tttau - 3.0 * ttau + 1.0) + t_n[i] * (tttau - 2.0 * ttau + tau) + r_o[i] * (-2.0 * tttau + 3.0 * ttau) + t_o[i] * (tttau - ttau);
                n_h[i][0] = 0.0;
                n_h[i][1] = 0.0;
                n_h[i][2] = 1.0;
            }
        }

        return std::make_pair(rhArray, nhArray);
    }

    //double h_h: prediction step time
    void update(const VectorArrayType &rArray, const VectorArrayType &vArray, const double *&n_, const double &h)
    {
        //double h_h = cover->frameDuration()*1.1;
        //double h_h = cover->frameDuration();

        const Vec3d *r_b = static_cast<Vec3d *>(rArray);
        const Vec3d *v_b = static_cast<Vec3d *>(vArray);

        const Vec3d &n_b = static_cast<Vec3d>(&n_);

        osgUtil::IntersectVisitor visitor;
        visitor.setTraversalMask(Isect::Collision);

        std::vector<osg::LineSegment *> normal(NumHitPoints);
        for (int i = 0; i < NumHitPoints; ++i)
        {
            Vec3d p = r_b[i] + v_b[i] * h_h;
            Vec3d pl = p - 5.0 * n_b;
            Vec3d pu = p + 0.2 * n_b;
            normal[i] = new osg::LineSegment(osg::Vec3(-pl[1], pl[0], pl[2]),
                                             osg::Vec3(-pu[1], pu[0], pu[2]));
            visitor.addLineSegment(normal[i]);
        }
        cover->getObjectsRoot()->accept(visitor);

        std::vector<Vec3d> r_i(NumHitPoints);
        std::vector<Vec3d> n_i(NumHitPoints);

        for (int i = 0; i < NumHitPoints; ++i)
        {
            if (visitor.getNumHits(normal[i]))
            {
                osgUtil::Hit &hit = visitor.getHitList(normal[i]).back();

                osg::Vec3d intersect = hit.getWorldIntersectPoint();
                r_i[i][0] = intersect.y();
                r_i[i][1] = -intersect.x();
                r_i[i][2] = intersect.z();

                osg::Vec3d normal = hit.getWorldIntersectNormal();
                n_i[i][0] = normal.y();
                n_i[i][1] = -normal.x();
                n_i[i][2] = normal.z();
            }
            else
            {
                n_i[i][0] = 0.0;
                n_i[i][1] = 0.0;
                n_i[i][2] = 1.0;
                r_i[i] = r_i[i] + (((r_b[i] + v_b[i] * h_h) - r_i[i]) ^ n_i[i]) * (!n_i[i]);
            }
        }

        for (int i = 0; i < 4; ++i)
        {
            //double tau = (double)step*h/dt;
            double tau = h / h_h;
            if (tau > 1.0)
            {
                //std::cerr << "FourWheelDynamicsRealtime::run(): Overlapped tau: " << (double)step*h/dt << ", setting to 1.0!" << std::endl;
                tau = 1.0;
            }
            double ttau = tau * tau;
            double tttau = ttau * tau;

            r_n[i] = r_n[i] * (2 * tttau - 3 * ttau + 1) + t_n[i] * (tttau - 2 * ttau + tau) + r_o[i] * (-2 * tttau + 3 * ttau) + t_o[i] * (tttau - ttau);
            t_n[i] = r_n[i] * (6 * ttau - 6 * tau) + t_n[i] * (3 * ttau - 4 * tau + 1) + r_o[i] * (-6 * ttau + 6 * tau) + t_o[i] * (3 * ttau - 2 * tau);

            r_o[i] = r_i[i];
            t_o[i] = ((v_b[i]) ^ n_i[i]) * (!n_i[i]) * h_h;
        }

        h_h = h;
    };

protected:
    Road *currentRoad[NumHitPoints];
    double currentLongPos[NumHitPoints];

    VectorArrayType rhArray;
    VectorArrayType nhArray;

    Vec3d r_n[NumHitPoints], t_n[NumHitPoints], r_o[NumHitPoints], t_o[NumHitPoints];

    double h_h;
};

#endif
