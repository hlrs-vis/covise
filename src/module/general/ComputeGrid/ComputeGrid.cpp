/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoSpheres.h>
#include <do/coDoPoints.h>
#include <appl/ApplInterface.h>
#include "ComputeGrid.h"
#include <vector>
#include <string>

#include <api/coFeedback.h>

ComputeGrid::ComputeGrid(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Creates a Grid of picked points")
    {
    /*Fed in points*/
    p_pointsIn = addInputPort("GridIn0", "Spheres|Points", "a set of points or spheres");
    
    /*Integer that specifies the particle to be traced in Module Parameter window*/
    p_particle = addStringParam("UnsortedSelection", "ranges of selected point elements");
    p_particle->setValue("");
    //pp_particle = addStringParam("selection", "ranges of selected point elements");
    //pp_particle->setValue("");
    p_maxParticleNumber = addInt32Param("maxParticleNum", "maximum number of particles to trace");
    p_maxParticleNumber->setValue(100);
    
    /*Integer that specifies the Dimension of Grid*/
    g_x_dim = addIntSliderParam("x_dim", "set x-Dimension of Grid");
    g_x_dim->setValue(1, 10, 1);
    g_y_dim = addIntSliderParam("y_dim", "set y-Dimension of Grid");
    g_y_dim->setValue(1, 10, 1);
    g_z_dim = addIntSliderParam("z_dim", "set z-Dimension of Grid");
    g_z_dim->setValue(1, 10, 1);
    
    /* 3 values specifying the x, y and z dimension of the bounding box */
    p_boundingBoxDimensions = addFloatVectorParam("BoundingBoxDimensions", "x, y, z dimensions of the bounding box");
    p_boundingBoxDimensions->setValue(0, 0, 0);
    
    /*Output*/
    p_pointsOut = addOutputPort("GridOut0", "Points", "Selected Points");
    p_GridOut = addOutputPort("GridOut1", "StructuredGrid", "Grid of selected Points");
}  

int ComputeGrid::compute(const char *)
{
    //input port object
    const coDistributedObject *input = p_pointsIn->getCurrentObject();

    if (!input)
    {
        //no input, no output
        sendError("Did not receive object at port '%s'", p_pointsIn->getName());
        return FAIL;
    }



    /* Creating the output object.*/
    if (p_pointsOut->getObjName())
    {
        std::string name;
        // test if input objects are of coDoPoints* type
        if (dynamic_cast<const coDoPoints *>(input))
        {
            const coDoPoints *inputpoints = dynamic_cast<const coDoPoints *>(input);

            // creating a string that contains all names of
            // all coDoPoints objects separated by newline escape sequence
            name += (inputpoints)->getName();
            name += "\n";
        }

        // test if input objects are of coDoSpheres* type
        if (dynamic_cast<const coDoSpheres *>(input))
        {
            const coDoSpheres *inputspheres = dynamic_cast<const coDoSpheres *>(input);

            // creating a string that contains all names of
            // all coDoSpheres objects separated by newline escape sequence
            name += (inputspheres)->getName();
            name += "\n";
        }

        // getting the selected particle string and
        // create the integer vector holding the element indices
        m_particleSelection.clear();
        std::stringstream ss(p_particle->getValue());
        int i;

        while (ss >> i)
        {
            m_particleSelection.push_back(i);

            if (ss.peek() == ',')
                ss.ignore();
        }

        //Getting number of particles
        //and set BoundingBoxDimensions
        int no_of_points = 0;
        float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
        float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        if (const coDoCoordinates *coord = dynamic_cast<const coDoCoordinates *>(input))
        {
            no_of_points = coord->getNumElements();
            for (int i = 0; i < no_of_points; ++i)
            {
                float x, y, z;
                coord->getPointCoordinates(i, &x, &y, &z);
                if (x > max[0])
                    max[0] = x;
                if (y > max[1])
                    max[1] = y;
                if (z > max[2])
                    max[2] = z;
                if (x < min[0])
                    min[0] = x;
                if (y < min[1])
                    min[1] = y;
                if (z < min[2])
                    min[2] = z;
            }
            p_boundingBoxDimensions->setValue(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        }
        else if (dynamic_cast<const coDoTriangleStrips *>(input))
        {
            sendError("Use Manual CPU Billboards instead of Polygons.");
            return FAIL;
        }
        else
        {
            sendError("wrong input type at input port. Type must be either Spheres or Points.");
            return FAIL;
        }

        //test if specified particles smaller than the maximum number of particles
        if (!m_particleSelection.empty() && m_particleSelection.back() >= no_of_points)
        {
            stringstream range;
            range << "0-" << no_of_points - 1;
            sendWarning("The particle index you specified is out of range. Particles out of range will be removed. Valid range is %s", range.str().c_str());
            while (!m_particleSelection.empty() && m_particleSelection.back() >= no_of_points)
            {
                m_particleSelection.pop_back();
            }
            std::string modifiedString = m_particleIndices.getRestraintString(m_particleSelection);
            p_particle->setValue(modifiedString.c_str());
        }

        if (m_particleSelection.size() > p_maxParticleNumber->getValue())
        {
            while (m_particleSelection.size() > p_maxParticleNumber->getValue())
                m_particleSelection.pop_back();
        }   

        //get size of Grid
        int x_nv = g_x_dim->getValue();
        int y_nv = g_y_dim->getValue();
        int z_nv = g_z_dim->getValue();

        // structure that holds the output components
        // creating the output
        // setting GridOut1
        coDoPoints *selectedPoints = getSelectedPointCoordinates(std::string(p_pointsOut->getObjName()), input);

        if (m_particleSelection.size() == x_nv * y_nv * z_nv)
        {
            coDoStructuredGrid *grid = new coDoStructuredGrid(p_GridOut->getObjName(), x_nv, y_nv, z_nv);
            float *x_start, *y_start, *z_start; 

            grid->getAddresses(&x_start, &y_start, &z_start);

            if (const coDoCoordinates *coordinates = dynamic_cast<const coDoCoordinates *>(selectedPoints))
            {
                for (int particle = 0; particle < m_particleSelection.size(); particle++)
                {
                    coordinates->getPointCoordinates(particle, &x_start[particle], &y_start[particle], &z_start[particle]);
                }
            }

            p_GridOut->setCurrentObject(grid);
        }
        if (m_particleSelection.size() > x_nv * y_nv * z_nv)
        {
            sendWarning("GridSize too small\n");
        }

        // creating the feedback object needed for
        // openCOVER<->COVISE interaction
        coFeedback feedback("PickSphere");

        feedback.addPara(p_particle);
        feedback.addPara(g_x_dim);
        feedback.addPara(g_y_dim);
        feedback.addPara(g_z_dim);

        // all coDoSphere object names are attached to the selected points
        selectedPoints->addAttribute("PICKSPHERE", name.c_str());

        // attaching the feedback object
        feedback.apply(selectedPoints);

        // setting the output object
        p_pointsOut->setCurrentObject(selectedPoints);

    }
    else
    {
        fprintf(stderr, "Covise::getObjName failed\n");
        return FAIL;
    }

    return SUCCESS;
}

coDoPoints *ComputeGrid::getSelectedPointCoordinates(const std::string &name, const coDistributedObject *input)
{

    coDoPoints *points = new coDoPoints(name, (int)m_particleSelection.size());
    float *x, *y, *z;

    points->getAddresses(&x, &y, &z);

    if (const coDoCoordinates *coord = dynamic_cast<const coDoCoordinates *>(input))

        for (int particle = 0; particle < m_particleSelection.size(); particle++)
        {
            coord->getPointCoordinates((int)m_particleSelection[particle], &x[particle], &y[particle], &z[particle]);
        }

    return points;
}

MODULE_MAIN(Filter, ComputeGrid)
