/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <string>
#include <alg/MagmaUtils.h>
#include <covise/covise_unstr.h>
#include "BorderEdges.hpp"

class PutIndizesIntoContainer : std::unary_function<void, GeometryUtils::Edge>
{
    std::set<int> &container_;

public:
    PutIndizesIntoContainer(std::set<int> &container)
        : container_(container)
    {
    }

    void operator()(GeometryUtils::Edge edge)
    {
        container_.insert(edge.first);
        container_.insert(edge.second);
    }
};

int
main(int argc, char *argv[])
{
    BorderEdges *application = new BorderEdges;
    application->start(argc, argv);
    return 42;
}

BorderEdges::BorderEdges()
    : coSimpleModule("Extract the border of the input-mesh.")
{
    p_in_geometry_ = addInputPort("in-polygons", "Polygons", "input");
    p_out_geometry_ = addOutputPort("out-border", "Lines", "output");
}

int BorderEdges::compute()
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

    // determine border
    GeometryUtils::Covise2dMeshWrapper aMesh(inMesh);
    vector<int> numberOfVerticesPerElement; // for satisfy the interface
    aMesh.calculateNumberOfVerticesPerElement(numberOfVerticesPerElement);
    vector<GeometryUtils::Edge> borderEdgesRelativeToMesh;
    MagmaUtils::BorderEdges(
        aMesh.elements, numberOfVerticesPerElement, aMesh.corners, aMesh.x.size(),
        borderEdgesRelativeToMesh);

    // compress/reorder the border
    std::set<int, std::less<int> > vertexNumbersOfBorder;
    std::for_each(
        borderEdgesRelativeToMesh.begin(), borderEdgesRelativeToMesh.end(),
        PutIndizesIntoContainer(vertexNumbersOfBorder));

    GeometryUtils::Covise2dMesh lineMesh;
    lineMesh.x.resize(vertexNumbersOfBorder.size());
    lineMesh.y.resize(vertexNumbersOfBorder.size());
    lineMesh.z.resize(vertexNumbersOfBorder.size());
    std::map<int, int> oldVertexNumber2New;

    int newNumber = 0;
    for (std::set<int, std::less<int> >::const_iterator citer
         = vertexNumbersOfBorder.begin();
         citer != vertexNumbersOfBorder.end();
         ++citer, ++newNumber)
    {
        oldVertexNumber2New[*citer] = newNumber;
        lineMesh.x[newNumber] = aMesh.x[*citer];
        lineMesh.y[newNumber] = aMesh.y[*citer];
        lineMesh.z[newNumber] = aMesh.z[*citer];
    }

    lineMesh.elements.resize(borderEdgesRelativeToMesh.size());
    lineMesh.corners.resize(2 * borderEdgesRelativeToMesh.size());
    for (int i = 0; i != lineMesh.elements.size(); ++i)
    {
        lineMesh.elements[i] = 2 * i;
        lineMesh.corners[2 * i] = oldVertexNumber2New[borderEdgesRelativeToMesh[i].first];
        lineMesh.corners[2 * i + 1] = oldVertexNumber2New[borderEdgesRelativeToMesh[i].second];
    }

    coDoLines *lines_out
        = new coDoLines(
            p_out_geometry_->getObjName(),
            lineMesh.x.size(),
            lineMesh.corners.size(),
            lineMesh.elements.size());

    if (!(lines_out and lines_out->objectOk()))
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
        lines_out->getAddresses(&u_out, &v_out, &w_out, &vl, &pl);
        std::copy(lineMesh.x.begin(), lineMesh.x.end(), u_out);
        std::copy(lineMesh.y.begin(), lineMesh.y.end(), v_out);
        std::copy(lineMesh.z.begin(), lineMesh.z.end(), w_out);
        std::copy(lineMesh.corners.begin(), lineMesh.corners.end(), vl);
        std::copy(lineMesh.elements.begin(), lineMesh.elements.end(), pl);
        p_out_geometry_->setCurrentObject(lines_out);
    }
    return SUCCESS;
}
