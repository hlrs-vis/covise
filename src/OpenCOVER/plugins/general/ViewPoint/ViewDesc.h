/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VOL_DESC_H_
#define _VOL_DESC_H_

#include <OpenVRUI/osg/mathUtils.h>

//========================================================================================
//========================================================================================
//========================================================================================

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/ClipNode>
#include <osg/ClipNode>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/LineWidth>
#include <osg/Material>

#include <PluginUtil/coVR3DTransInteractor.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coInteractor.h>

#include <cover/coVRMSController.h>
#include <grmsg/coGRActivatedViewpointMsg.h>
#include <cover/ui/Owner.h>

//========================================================================================
//========================================================================================
//========================================================================================
using namespace osg;

namespace opencover {
namespace ui {
class Action;
class Button;
class Menu;
class EditField;
}
}

class FlightPathVisualizer;
class ViewPoints;

    struct clipPlaneEntry
    {
        bool enabled;
        float a, b, c, d;

        clipPlaneEntry()
            : enabled(false)
            , a(0.0f)
            , b(0.0f)
            , c(0.0f)
            , d(0.0f)
        {
        }
    };


class ViewDesc: public opencover::ui::Owner
{
private:
    std::string name; //< name of the point
    int id_;
    float scale_;
    std::string vp_line; //< viewpoint line for covise.config
    bool flightState_; //<  point included in flight?
    opencover::ui::Action *button_; //< button in Viepoints menu
    opencover::ui::Action *changeButton_; //< button for change in Viepoints menu
    opencover::ui::Button *flightButton_; // button in flight menu
    bool hasScale_;
    bool isViewAll_;
    bool hasPosition_;
    bool hasOrientation_;
    bool hasMatrix_;
    bool activated_; // flag if flight is finished and viewpoint is reached
    bool isChangeable_ = true;
    bool isChangeableFromCover_ = false;
    osg::Matrix xformMat_;
    // common functionality of C'tors
    void createButtons(const char *name,
                       opencover::ui::Menu *menu, opencover::ui::Menu *flightMenu, opencover::ui::Menu *editMenu,
                       ViewPoints *master);


    //========================================================================================
    //========================================================================================
    //========================================================================================
    opencover::ui::Menu *editVPMenu_; //< Edit Viewpoint- menu

    opencover::ui::EditField *editNameInput_;
    opencover::ui::Button *showViewpointCheck_;
    opencover::ui::Button *showTangentCheck_;
    opencover::ui::Button *showMoveInteractorsCheck_;
    opencover::ui::Button *showTangentInteractorsCheck_;
    opencover::ui::Action *updateViewButton;

    bool flightPathActivated;
    bool viewpointVisible;
    bool tangentVisible;
    bool editViewpoint;
    bool editTangent;
    bool shiftFlightpathToEyePoint;
    bool hasTangentOut_;
    bool hasTangentIn_;
    bool hasGeometry_;

    Vec3 eyepoint;
    Vec3 tangentIn;
    Vec3 tangentOut;
    Vec3 scaleVec;

    ref_ptr<Geode> tangentgeode;
    ref_ptr<Geode> viewpointgeode;
    ref_ptr<Switch> viewpointSwitchNode;
    ref_ptr<Switch> tangentSwitchNode;

    ref_ptr<osg::Geometry> line1;
    ref_ptr<osg::Geometry> line2;
    ref_ptr<osg::Geometry> line3;
    ref_ptr<osg::Geometry> line4;

    ref_ptr<Vec3Array> lineEyetoLeftDown;
    ref_ptr<Vec3Array> lineEyetoRightDown;
    ref_ptr<Vec3Array> lineEyetoRightUp;
    ref_ptr<Vec3Array> lineEyetoLeftUp;

    ref_ptr<StateSet> line1_state;
    ref_ptr<StateSet> line2_state;
    ref_ptr<StateSet> line3_state;
    ref_ptr<StateSet> line4_state;

    ref_ptr<osg::Geometry> tangentlinesgeoset;
    ref_ptr<osg::Geometry> viewpointPlaneGeoset;
    ref_ptr<osg::Geometry> viewpointPlaneBorderGeoset;
    ref_ptr<osg::Geometry> viewpointGeoset;
    ref_ptr<Vec3Array> viewpointCoords; // coordinates of viewpoint-box(Plane)
    ref_ptr<Vec3Array> viewpointBorderCoords; //// coordinates of viewpoint-box(Border)
    ref_ptr<Vec3Array> tangentlinescoords; // coordinates of tangent-lines

    ref_ptr<StateSet> tangentlinesgeoset_state;
    ref_ptr<StateSet> viewpointPlaneGeoset_state;
    ref_ptr<StateSet> viewpointGeoset_state;
    ref_ptr<StateSet> viewpointPlaneBorderGeoset_state;

    //     coCoord coord;           //< pos + orientation
    float stoptime; //< stoptime of flight in this viewpoint

    opencover::coVR3DTransInteractor *tanOutInteractor;
    opencover::coVR3DTransInteractor *tanInInteractor;
    opencover::coVR3DTransInteractor *scaleInteractor;

    opencover::coVR3DTransRotInteractor *viewpointInteractor; // Angriffspunkt in Mitte

    //      coVRLabel *myLabel;

    void loadUnlightedGeostate(ref_ptr<StateSet> state);
    //========================================================================================
    //========================================================================================
    //========================================================================================

public:
    coCoord coord; //< pos + orientation

    // constructor with covise.config_entry
    ViewDesc(const char *name, int id, const char *line,
             opencover::ui::Menu *menu, opencover::ui::Menu *flightMenu, opencover::ui::Menu *editMenu,
             ViewPoints *master, bool isChangeable = true);
    // S%f=X%f=Y%f=Z%f=H%f=P%f=R%f

    // constructor with matrix
    ViewDesc(const char *name, int id, float scale, osg::Matrix m,
             opencover::ui::Menu *menu, opencover::ui::Menu *flightMenu, opencover::ui::Menu *editMenu,
             ViewPoints *master, bool isChangeable = true);

    // construct empty point with name only
    ViewDesc(const char *name, int id,
             opencover::ui::Menu *menu, opencover::ui::Menu *flightMenu, opencover::ui::Menu *editMenu,
             ViewPoints *master, bool isChangeable = true);

    // constructor with only scale
    ViewDesc(const char *name, int id, float scale,
             opencover::ui::Menu *menu, opencover::ui::Menu *flightMenu, opencover::ui::Menu *editMenu,
             ViewPoints *master, bool isChangeable = true);

    // constructor with only orientation
    ViewDesc(const char *name, int id, osg::Vec3 hpr,
             opencover::ui::Menu *menu, opencover::ui::Menu *flightMenu, opencover::ui::Menu *editMenu,
             ViewPoints *master, bool isChangeable = true);

    virtual ~ViewDesc();

    // change the viewpoints scale and xFormMatrix
    void changeViewDesc(float scale, osg::Matrix m);
    void changeViewDesc();

    void setPosition(const char *posString);
    void setScale(const char *scaleString);
    void setScale(float scale);
    void setEuler(const char *eulerString);
    void addClipPlane(const char *planeString);
    void setXFromMatrix(osg::Matrix m);
    void setFlightState(bool state);
    void alignViewpoint(char alignment);
    void updateToViewAll();

    const char *getName()
    {
        return name.c_str();
    }
    void setName(const char *n);
    osg::Matrix getMatrix()
    {
        return xformMat_;
    }
    const char *getLine()
    {
        return vp_line.c_str();
    }
    const char *getClipPlane(int i);
    bool getFlightState()
    {
        return flightState_;
    }
    int getId()
    {
        return id_;
    }
    float getScale()
    {
        return scale_;
    }
    void activate(bool clipplane);

    bool hasScale()
    {
        return hasScale_;
    };
    bool hasPosition()
    {
        return hasPosition_;
    };
    bool hasOrientation()
    {
        return hasOrientation_;
    };
    bool hasMatrix()
    {
        return hasMatrix_;
    };
    bool isViewAll()
    {
        return isViewAll_;
    };
    bool isActivated()
    {
        return activated_;
    };
    void setActivated(bool b)
    {
        activated_ = b;
    };
    bool isChangeable()
    {
        return isChangeable_;
    };
    void setChangeable(bool c)
    {
        isChangeable_ = c;
    };
    bool isClipPlaneEnabled(int plane);

    static bool string2ViewDesc(const std::string &line,
                                float *scale,
                                float *x, float *y, float *z,
                                float *h, float *p, float *r,
                                float *tanInX, float *tanInY, float *tanInZ,
                                float *tanOutX, float *tanOutY, float *tanOutZ);

    static bool string2ViewDesc(const std::string &line,
                                float *scale,
                                double *m00, double *m01, double *m02, double *m03,
                                double *m10, double *m11, double *m12, double *m13,
                                double *m20, double *m21, double *m22, double *m23,
                                double *m30, double *m31, double *m32, double *m33,
                                float *tanInX, float *tanInY, float *tanInZ,
                                float *tanOutX, float *tanOutY, float *tanOutZ);

    //========================================================================================
    //========================================================================================
    //========================================================================================
    bool hasGeometry()
    {
        return hasGeometry_;
    };
    //   bool hasTangent();
    void createGeometry();
    void updateGeometry();
    void showGeometry(bool);
    void shiftFlightpath(bool state);
    bool equalVP(Matrix m);
    bool nearVP(Matrix m);
    void preFrame(FlightPathVisualizer *vpVis); // parameter: FlightPathVisualizer *vpVis
    Vec3 getTangentIn();
    Vec3 getTangentOut();
    Vec3 getScaledTangentIn();
    Vec3 getScaledTangentOut();
    bool hasTangent();
    void setTangentOut(const char *tangentString);
    void setTangentOut(Vec3 tanOut);
    void setTangentIn(const char *tangentString);
    void setTangentIn(Vec3 tanIn);
    void deleteGeometry();
    void showTangent(bool state);
    void showMoveInteractors(bool state);
    void showTangentInteractors(bool state);
    bool getFlightPathActivated()
    {
        return flightPathActivated;
    }
    void setFlightPathActivated(bool);

    opencover::ui::Action *getButton()
    {
        return button_;
    };
    struct clipPlaneEntry clipPlanes[6];

    ref_ptr<MatrixTransform> localDCS;
    //========================================================================================
    //========================================================================================
    //========================================================================================
};
#endif
