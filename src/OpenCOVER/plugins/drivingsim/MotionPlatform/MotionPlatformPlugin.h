/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MOTIONPLATFORMPLUGIN_H
#define _MOTIONPLATFORMPLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: MotionPlatform Plugin (does nothing)                        **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <coVRPluginSupport.h>
#include <VRSceneGraph.h>
#include <coVRFileManager.h>
#include <config/CoviseConfig.h>

#include <osgViewer/Viewer>
#include <osg/NodeCallback>
#include <osg/Camera>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>
#include <osg/NodeVisitor>

#include "InvertMatrix.hpp"
using namespace covise;
using namespace opencover;

namespace opencover
{
class coTUISlider;
class coTUIFloatSlider;
class coTUIToggleButton;
}

class MotionPlatformPlugin : public coVRPlugin, public coTUIListener
{
public:
    MotionPlatformPlugin();
    ~MotionPlatformPlugin();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(RenderObject *container, coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

    osg::ref_ptr<osg::MatrixTransform> InsertSceneElement(osg::Group *parent,
                                                          std::string filepath,
                                                          std::string nodename);
    //function containing the kinematics function
    boost::numeric::ublas::vector<double> F(const boost::numeric::ublas::vector<double> &X)
    {
        //Die vom Newtonverfahren gerade untersuchten Werte entnehmen:
        double lambda = X[0];
        double alpha = X[1];
        //Reduktion auf 2D Problem:
        boost::numeric::ublas::matrix<double> wRq2D(2, 2);
        boost::numeric::ublas::matrix<double> Ra(2, 2);

        wRq2D(0, 0) = wTq(0, 0);
        wRq2D(0, 1) = wTq(2, 0);
        wRq2D(1, 0) = wTq(0, 2);
        wRq2D(1, 1) = wTq(2, 2);

        Ra(0, 0) = cos(alpha);
        Ra(0, 1) = sin(alpha);
        Ra(1, 0) = -sin(alpha);
        Ra(1, 1) = cos(alpha);

        boost::numeric::ublas::vector<double> zL2D(2);
        zL2D(0) = zL[0];
        zL2D(1) = zL[2];

        boost::numeric::ublas::vector<double> wQ2D(2);
        wQ2D(0) = wTq(3, 0);
        wQ2D(1) = wTq(3, 2);

        boost::numeric::ublas::vector<double> la(2);
        la(0) = lambda;
        la(1) = 0;

        boost::numeric::ublas::vector<double> wZtmp(2);
        wZtmp(0) = wZ[0];
        wZtmp(1) = wZ[2];

        return (boost::numeric::ublas::vector<double>)(boost::numeric::ublas::prod(wRq2D, la) + wQ2D - wZtmp - boost::numeric::ublas::prod(Ra, zL2D));
    }
    boost::numeric::ublas::matrix<double> JacNum_4O(boost::numeric::ublas::vector<double> &input)
    {
        using namespace boost::numeric::ublas;
        //create a working copy of the input
        boost::numeric::ublas::vector<double> Q(input);
        int mQ = Q.size();
        boost::numeric::ublas::vector<double> X1(mQ);
        boost::numeric::ublas::vector<double> X2(mQ);
        boost::numeric::ublas::vector<double> X3(mQ);
        boost::numeric::ublas::vector<double> X4(mQ);
        boost::numeric::ublas::vector<double> Y1(mQ);
        boost::numeric::ublas::vector<double> Y2(mQ);
        boost::numeric::ublas::vector<double> Y3(mQ);
        boost::numeric::ublas::vector<double> Y4(mQ);
        boost::numeric::ublas::matrix<double> J(mQ, mQ);

        double t_eps = 1.0e-5;

        for (int n = 0; n < mQ; n++)
        {
            X1 = Q;
            X2 = Q;
            X3 = Q;
            X4 = Q;
            X1[n] = X1[n] - 2 * t_eps;
            X2[n] = X2[n] - t_eps;
            X3[n] = X3[n] + t_eps;
            X4[n] = X4[n] + 2 * t_eps;

            Y1 = F(X1);
            Y2 = F(X2);
            Y3 = F(X3);
            Y4 = F(X4);
            column(J, n) = (Y1 - 8.0 * Y2 + 8.0 * Y3 - Y4) / (12.0 * t_eps);
        }
        return J;
    }
    bool solve_4O(const ublas::vector<double> &input, boost::numeric::ublas::vector<double> &newsolution)
    {
        //create a working copy of the input
        boost::numeric::ublas::vector<double> xalt(input);
        boost::numeric::ublas::vector<double> xneu(xalt.size());

        double err = 1.0;
        int n = 0;

        // resize return value
        newsolution.resize(xalt.size());

        while ((err > this->eps) && (n <= this->nmax))
        {
            boost::numeric::ublas::matrix<double> Jac(this->JacNum_4O(xalt));

            boost::numeric::ublas::matrix<double> invJac(xalt.size(), xalt.size());

            if (InvertMatrix(Jac, invJac))
            {

                //std::cout <<"Jac : "   << Jac    << std::endl;
                //std::cout <<"invJac :" << invJac << std::endl;
            }
            else
            {
                std::cout << "not invertable! : " << Jac << std::endl;
            }
            boost::numeric::ublas::vector<double> fxalt(2);
            fxalt = this->F(xalt);

            //std::cout << "F(xalt): " << fxalt << std::endl;
            xneu = xalt - prod(invJac, fxalt);

            err = boost::numeric::ublas::norm_2(this->F(xneu));
            n++;
            xalt = xneu;
        }

        if (err <= this->eps)
        {
            newsolution.assign(xneu);
            return true;
        }
        else
        {
            newsolution.assign(input);
            std::cout << "No solution!" << std::endl;
            return false;
        }
    };
    double geteps() const
    {
        return eps;
    }
    void seteps(double val)
    {
        eps = val;
    }

    int getnmax() const
    {
        return nmax;
    }
    void setnmax(int val)
    {
        nmax = val;
    }

private:
    coTUITab *MotionPlatformTab;
    coTUIFloatSlider *leftSlider;
    coTUIFloatSlider *rightSlider;
    coTUIFloatSlider *backSlider;
    coTUIToggleButton *animateButton;
    // dL = hubhoehe der drei Linearmotoren;
    float dL;
    osg::Vec3 yawAxis, pitchAxis, rollAxis;
    // tmp-Werte der Linearmotoren
    osg::Matrix wTml, wTmr, wTmh;
    float tmp;
    // Nullhoehen der Linearmotoren;
    float hl0, hr0, hh0;
    // tmp-Werte der Kugelkoepfe
    osg::Matrix wTkl, wTkr, wTkh;
    //tmp-Werte der Querstange
    osg::Matrix wTq;
    //tmp-Werte der Zange
    osg::Vec3 wZ;
    osg::Matrix wTz;
    //tmp-Werte der Linearfuehrung
    osg::Matrix wTl;
    osg::Vec3 zL;
    //tmp-Werte des Aufbaus
    osg::Matrix wTa;

    //Newton-Instanz
    boost::numeric::ublas::vector<double> X0;
    boost::numeric::ublas::vector<double> X;

    // Zeitvariable
    float t;
    int counter;
    double eps;
    int nmax;
    // Path to the osg geometry files
    std::string filedirpath;

    osg::ref_ptr<osg::MatrixTransform> Aufbau;
    osg::ref_ptr<osg::MatrixTransform> Quertraeger;
    osg::ref_ptr<osg::MatrixTransform> LinMot_L;
    osg::ref_ptr<osg::MatrixTransform> LinMot_R;
    osg::ref_ptr<osg::MatrixTransform> LinMot_H;
    osg::ref_ptr<osg::MatrixTransform> KK_L;
    osg::ref_ptr<osg::MatrixTransform> KK_R;
    osg::ref_ptr<osg::MatrixTransform> KK_H;
    osg::ref_ptr<osg::MatrixTransform> Zange;
    osg::ref_ptr<osg::MatrixTransform> LinFuehr;
};
#endif
