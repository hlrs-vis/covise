/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <string>
#include <alg/MagmaUtils.h>
#include <covise/covise_unstr.h>
#include "extendmesh2rectangle.hpp"

namespace GeneralUtils
{
bool const reachedUnreachableFalse = false;
}

int
main(int argc, char *argv[])
{
    ExtendMesh2Rectangle *application = new ExtendMesh2Rectangle;
    application->start(argc, argv);
    assert(GeneralUtils::reachedUnreachableFalse);
    return 42;
}

ExtendMesh2Rectangle::ExtendMesh2Rectangle()
    : coSimpleModule("Extend the mesh to fit into the specified rectangle.")
    , DEFAULT_MAXIMUM_(1e6)
    , xSize_(42.0)
    , ySize_(42.0)
    , radius_(-1.0)
{
    p_in_geometry_ = addInputPort("in-polygons", "Polygons", "input mesh");
    p_out_geometry_ = addOutputPort("out-polygons", "Polygons", "output mesh");
    para_xSize_ = addFloatSliderParam("x-size", "x-size of the rectangle.");
    para_xSize_->setValue(0.0, DEFAULT_MAXIMUM_, xSize_);
    para_ySize_ = addFloatSliderParam("y-size", "y-size of the rectangle.");
    para_ySize_->setValue(0.0, DEFAULT_MAXIMUM_, ySize_);
    para_radius_ = addFloatSliderParam(
        "radius", "Radius of circle to use for intermediate mesh.  "
                  "If radius < 0.0 hold this feature is disabled.");
    para_radius_->setValue(-1.0, DEFAULT_MAXIMUM_, radius_);
}

int
ExtendMesh2Rectangle::compute()
{
    coDoPolygons *inMesh = dynamic_cast<coDoPolygons *>(p_in_geometry_->getCurrentObject());
    if (!(inMesh && inMesh->objectOk()))
    {
        std::string errorText = "Input object at port '";
        errorText += p_in_geometry_->getName();
        errorText += "not available or not ok.";
        Covise::sendError(errorText.c_str());
        return STOP_PIPELINE;
    }
    xSize_ = para_xSize_->getValue();
    ySize_ = para_ySize_->getValue();
    radius_ = para_radius_->getValue();
    GeometryUtils::Covise2dMeshWrapper aMesh(inMesh);
    if (featureRadiusEnabled())
        GeometryUtils::extendToQuarterCircleOfRadius(radius_, aMesh);
    GeometryUtils::extendToQuarterRectangle(xSize_, ySize_, aMesh);
    coDoPolygons *polygons_out
        = new coDoPolygons(
            p_out_geometry_->getObjName(),
            aMesh.x.size(),
            aMesh.corners.size(),
            aMesh.elements.size());
    if (!(polygons_out and polygons_out->objectOk()))
    {
        std::string errorText = "Object for port '";
        errorText += p_out_geometry_->getName();
        errorText += "' can't be created.";
        Covise::sendError(errorText.c_str());
        return STOP_PIPELINE;
    }
    {
        float *u_out;
        float *v_out;
        float *w_out;
        int *vl;
        int *pl;
        polygons_out->getAddresses(&u_out, &v_out, &w_out, &vl, &pl);
        std::copy(aMesh.x.begin(), aMesh.x.end(), u_out);
        std::copy(aMesh.y.begin(), aMesh.y.end(), v_out);
        std::copy(aMesh.z.begin(), aMesh.z.end(), w_out);
        std::copy(aMesh.corners.begin(), aMesh.corners.end(), vl);
        std::copy(aMesh.elements.begin(), aMesh.elements.end(), pl);
        p_out_geometry_->setCurrentObject(polygons_out);
    }
    return SUCCESS;
}
