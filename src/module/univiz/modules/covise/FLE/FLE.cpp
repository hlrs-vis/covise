/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Finite Lyapunov Exponents
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "FLE.h"

#include "covise_ext.h"
#include "linalg.h"
#include "unstructured.h"
#include "unifield.h"
#include "unisys.h"

#include "FLE_impl.cpp" // ### including .cpp

static Unstructured *unst_in = NULL;

UniSys us = UniSys(NULL);

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::handleParamWidgets(void)
{
    // show all params (because otherwise they can not be enabled/disabled
    origin->show();
    cells->show();
    cellSize.show();
    unsteady->show();
    velocityFile->show();
    startTime.show();
    mode->show();
    ln->show();
    divT->show();
    integrationTime.show();
    integrationLength.show();
    timeIntervals.show();
    sepFactorMin.show();
    integStepsMax.show();
    forward->show();
    smoothingRange.show();
    omitBoundaryCells->show();
    gradNeighDisabled->show();
    execute->show();

    // enable/disable parameters

    // sampling grid
    if (samplingGrid->getCurrentObject())
    {
        origin->disable();
        cells->disable();
        cellSize.disable();
    }
    else
    {
        origin->enable();
        cells->enable();
        cellSize.enable();
    }

    // unsteady
    if (unsteady->getValue())
    {
        velocityFile->enable();
        startTime.enable();
    }
    else
    {
        velocityFile->disable();
        startTime.disable();
    }

    // mode
    if (mode->getValue() == 0)
    { // FTLE
        ln->enable();
        divT->enable();
        integrationTime.enable();
        integrationLength.disable();
        sepFactorMin.disable();
        timeIntervals.disable();
    }
    else if (mode->getValue() == 1)
    { // FLLE
        ln->enable();
        divT->enable();
        integrationTime.disable();
        integrationLength.enable();
        sepFactorMin.disable();
        timeIntervals.disable();
    }
    else if (mode->getValue() == 2)
    { // FSLE
        ln->enable();
        divT->enable();
        integrationTime.enable();
        integrationLength.disable();
        sepFactorMin.enable();
        timeIntervals.enable();
    }
    else if ((mode->getValue() == 3) || (mode->getValue() == 4))
    { // FTLEM or FTLEA
        ln->enable();
        divT->enable();
        integrationTime.enable();
        integrationLength.disable();
        sepFactorMin.disable();
        timeIntervals.enable();
    }
}

void myModule::postInst()
{
    sendInfo("please use Control Panel for parametrization");

    // show/hide parameters
    handleParamWidgets();
}

void myModule::param(const char *, bool)
{
    // force min/max
    adaptMyParams();

    // show/hide parameters
    handleParamWidgets();
}

int myModule::compute(const char *)
{
    // force min/max
    adaptMyParams();

    // show/hide parameters
    handleParamWidgets();

    // system wrapper
    us = UniSys(this);

    if (unsteady->getValue() && (mode->getValue() + 1 == 2))
    {
        us.warning("FLLE makes no sense for transient data (\"nearby trajectories are not nearby\") !!");
    }

    if (integrationTime.getValue() <= 0.0)
    {
        us.error("integration time must be larger than zero");
        return FAIL;
    }

    // create Unstructured wrapper for velocity input
    if (us.inputChanged("ucd", 0))
    {
        if (unst_in)
            delete unst_in;
        std::vector<coDoVec3 *> vvec;
        vvec.push_back((coDoVec3 *)(velocity->getCurrentObject()));
        unst_in = new Unstructured((coDoUnstructuredGrid *)(velocityGrid->getCurrentObject()),
                                   NULL, &vvec);
    }

    // create Unstructured wrapper for sampling grid
    Unstructured *unst_samplingGrid;
    coDoUnstructuredGrid *sgrid;
    if (samplingGrid->getCurrentObject())
    {
        unst_samplingGrid = new Unstructured((coDoUnstructuredGrid *)(samplingGrid->getCurrentObject()),
                                             NULL, NULL);

        sgrid = (coDoUnstructuredGrid *)(samplingGrid->getCurrentObject());
    }
    else
    {
        sgrid = generateUniformUSG(outputGrid->getObjName(),
                                   origin->getValue(0),
                                   origin->getValue(1),
                                   origin->getValue(2),
                                   cells->getValue(0),
                                   cells->getValue(1),
                                   cells->getValue(2),
                                   cellSize.getValue());
        if (!sgrid)
        {
            sendError("failed to create sampling grid");
            return FAIL;
        }

        unst_samplingGrid = new Unstructured(sgrid, NULL, NULL);
    }

    // go
    {

        // Unstructured wrapper for output
        Unstructured *unst_out;
        coDoFloat *FLEData;
        coDoFloat *eigenvalMaxData;
        coDoFloat *eigenvalMedData;
        coDoFloat *eigenvalMinData;
        coDoFloat *integrationSizeData;
        coDoVec3 *mapData;
        {
            // alloc output field (future: TODO: do it inside Unstructured)
            int nCells, nConn, nNodes;
            sgrid->getGridSize(&nCells, &nConn, &nNodes);

            FLEData = new coDoFloat(FLE->getObjName(), nNodes);
            eigenvalMaxData = new coDoFloat(eigenvalMax->getObjName(), nNodes);
            eigenvalMedData = new coDoFloat(eigenvalMed->getObjName(), nNodes);
            eigenvalMinData = new coDoFloat(eigenvalMin->getObjName(), nNodes);
            integrationSizeData = new coDoFloat(integrationSize->getObjName(), nNodes);
            mapData = new coDoVec3(map->getObjName(), nNodes);

            // wrap
            std::vector<coDoFloat *> svec;
            svec.push_back(FLEData);
            svec.push_back(eigenvalMaxData);
            svec.push_back(eigenvalMedData);
            svec.push_back(eigenvalMinData);
            svec.push_back(integrationSizeData);
            std::vector<coDoVec3 *> vvec;
            vvec.push_back(mapData);
            unst_out = new Unstructured(sgrid, &svec, &vvec);
        }

        // setup Unifield for trajectory output
        UniField *unif_traj = NULL;
        {
            std::vector<coOutputPort *> outPortVec;
            outPortVec.push_back(trajectoriesGrid);
            outPortVec.push_back(trajectoriesData);
            unif_traj = new UniField(outPortVec);

            int nCells, nConn, nNodes;
            sgrid->getGridSize(&nCells, &nConn, &nNodes);

            int dims[2];
            dims[0] = integStepsMax.getValue() + 1;
            dims[1] = nNodes;
            int compVecLens[1] = { 2 };
            const char *compNames[1] = { "trajectory data" };
            if ((unif_traj)->allocField(2 /*ndims*/, dims, 3 /*nspace*/,
                                        false /*regular*/, 1, compVecLens, compNames, UniField::DT_FLOAT) == false)
            {
                us.error("out of memory");
            }
        }

        // compute
        if (execute->getValue())
        {
            FLE_impl(&us, unst_in, 0, unsteady->getValue(), velocityFile->getValue(), startTime.getValue(),
                     //crop_ucd, *origin_x, *origin_y, *origin_z, *voxel_size,
                     mode->getValue() + 1,
                     ln->getValue(), divT->getValue(),
                     integrationTime.getValue(), integrationLength.getValue(),
                     timeIntervals.getValue(), sepFactorMin.getValue(), integStepsMax.getValue(), forward->getValue(),
                     unst_out, smoothingRange.getValue(),
                     omitBoundaryCells->getValue(), gradNeighDisabled->getValue(), unif_traj);
        }

        // output data already assigned to ports
        FLE->setCurrentObject(FLEData);
        eigenvalMax->setCurrentObject(eigenvalMaxData);
        eigenvalMed->setCurrentObject(eigenvalMedData);
        eigenvalMin->setCurrentObject(eigenvalMinData);
        integrationSize->setCurrentObject(integrationSizeData);
        map->setCurrentObject(mapData);

        if (unif_traj)
        {

            // assign output data to ports
            coDoStructuredGrid *wgrid;
            coDoVec2 *wdat;
            unif_traj->getField(&wgrid, NULL, &wdat);

            trajectoriesGrid->setCurrentObject(wgrid);
            trajectoriesData->setCurrentObject(wdat);

            // delete field wrapper (but not the field)
            delete unif_traj;
        }

        delete unst_out;
    }

    delete unst_samplingGrid;
    if (!samplingGrid->getCurrentObject())
    {
        outputGrid->setCurrentObject(sgrid);
    }

    return SUCCESS;
}
