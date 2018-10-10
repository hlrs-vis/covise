/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NurbsSurface_PLUGIN_H
#define _NurbsSurface_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: NurbsSurface OpenCOVER Plugin (draws a NurbsSurface)        **
 **                                                                          **
 **                                                                          **
 ** Author: F.Karle/ K.Ahmann                                                **
 **                                                                          **
 ** History:                                                                 **
 ** December 2017  v1                                                        **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <cover/coVRCommunication.h>
#include <net/message.h>


#include <string>
#include <cover/ui/Owner.h>
#include <cover/ui/Label.h>

#include "alglib/stdafx.h"
#include "alglib/interpolation.h"

#include "sisl.h"
#include "sisl_aux/sisl_aux.h"

#include <algorithm>
#include <functional>
#include <assert.h>



namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
namespace ui {
class Slider;
class Label;
}
}

using namespace opencover;
using namespace alglib;
using namespace osg;

class NurbsSurface : public coVRPlugin, public ui::Owner
{
public:
struct curveInfo{
  double startPar = 0.0;
  double endPar = 0.0;
  SISLCurve *curve = nullptr;
};
    NurbsSurface();
    ~NurbsSurface();
    bool init();

    void message(int toWhom, int type, int len, const void *buf); ///< handle incoming messages
    //int getorder_U();
    //void setorder_U(int order_U);

    struct surfaceInfo{
        void createRBFModel();
        rbfmodel model;
        osg::ref_ptr<osg::Geode> computeSurface(double* points);
        void updateSurface();
        //const int num_surf = 1;
        int surfaceIndex = 0;
        const int num_points_u = 3; // number of points in the u parameter direction
        const int num_points_v = 3; // number of points in the v parameter direction
        double u_par[3] = {0, 1, 2}; // point parametrization in u-direction
        double v_par[3] = {0, 1, 2}; // point parametrization in v-direction

        const int dim = 3; // dimension of the space we are working in
        int order_U = 2;
        int order_V = 2;
        osg::ref_ptr<osg::Geode> geode;
        std::vector<osg::Vec3> receivedPoints;
        osg::Matrixd rotationMatrixToWorld;
        osg::Matrixd rotationMatrixToLocal;
        std::vector<osg::Vec3> receivedPointsRotated;
        osg::ref_ptr<osg::Group> splinePointsGroup;
        std::vector<osg::MatrixTransform*> transformMatrices; //stores the highlighted Points
        curveInfo upper;
        curveInfo lower;
        curveInfo left;
        curveInfo right;
        int numberOfAllPoints;
        float maximum_x;
        float minimum_x;
        float maximum_y;
        float minimum_y;

        void updateModel();
        void resize();
        bool calcEdges();
        void evaluateCurveAtParam(curveInfo& curve, double paramFactor, std::vector<double>& point);
        std::vector<double> evaluateCurveAtParam(curveInfo& curve, double paramFactor);

        real_2d_array xy;

        bool curveCurveIntersection(SISLCurve* c1, double& c1Param, SISLCurve* c2, double& c2Param);
        double sphereSize = 10.0;
        virtual bool destroy();
        int edge(std::vector<osg::Vec3> all_points, int local_x, int local_y, int change, curveInfo &resultCurveInfo);
        int edgeByPoints(std::vector<osg::Vec3> all_points, osg::Vec3 pointBegin, osg::Vec3 pointEnd, curveInfo &resultCurveInfo);
        int numEdgeSectors = 5;
        void highlightPoint(osg::Vec3& newSelectedPoint);
    };

private:
    surfaceInfo* currentSurface = nullptr;
    std::vector<surfaceInfo> surfaces;

    std::vector<osg::ref_ptr<osg::Geode>> surfaceGeodes;

    void saveFile(const std::string &fileName);

    ui::Menu *NurbsSurfaceMenu; //< menu for NurbsSurface Plugin

    ui::Label *currentSurfaceLabel = nullptr;
    ui::Action *newSurface = nullptr;
    ui::Action *saveButton_;

    ui::Slider *surfaceSelectionSlider=nullptr;

    ui::Group *selectionParameters = nullptr;

    ui::Slider *orderUSlider=nullptr;
    ui::Slider *orderVSlider=nullptr;

    ui::Slider *numEdgeSectorsSlider=nullptr;
    std::string fileName = "test.obj";
    void updateMessage();
    void initUI();
    void updateUI();
    void createNewSurface();
};

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}



template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}

template <typename T, typename U>
std::vector<T> operator/(const std::vector<T>& a, const U& b)
{
    std::vector<T> result;
    result.reserve(a.size());
    typename std::vector<T>::const_iterator begin;
    begin = a.begin();
    while(begin != a.end())
                   *std::back_inserter(result)= *begin++/b;
    return result;
}

template <typename T, typename U>
std::vector<T> operator*(const std::vector<T>& a, const U& b)
{
    std::vector<T> result;
    result.reserve(a.size());
    typename std::vector<T>::const_iterator begin;
    begin = a.begin();
    while(begin != a.end())
                   *std::back_inserter(result)= *begin++*b;
    return result;
}

template <typename U, typename T>
std::vector<T> operator*(const U& b,const std::vector<T>& a)
{
    std::vector<T> result;
    result.reserve(a.size());
    typename std::vector<T>::const_iterator begin;
    begin = a.begin();
    while(begin != a.end())
                   *std::back_inserter(result)= *begin++*b;
    return result;
}


#endif

