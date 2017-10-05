/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <do/coDoGeometry.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include "Probe3D.h"
#include <alg/coComplexModules.h>
#include <api/coFeedback.h>

static void
normalise(float *e1)
{
    float len = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    if (len > 0.0)
        len = 1.0f / len;
    else
        return;
    e1[0] *= len;
    e1[1] *= len;
    e1[2] *= len;
}

static void
vecProd(const float *e1, const float *e2, float *e3)
{
    e3[0] = e1[1] * e2[2] - e1[2] * e2[1];
    e3[1] = e1[2] * e2[0] - e1[0] * e2[2];
    e3[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

Probe3D::Probe3D(int argc, char *argv[])
    : coModule(argc, argv, "Probe 3d module", true)
    , gridIsTimeDependent_(false)
    , polyIsTimeDependent_(false)
{
    point_[0] = 0.;
    point_[1] = 0.;
    point_[2] = 0.;
    normal_[0] = 0.;
    normal_[1] = 0.;
    normal_[2] = 1.;

    // ports
    p_grid_ = addInputPort("meshIn", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid", "input mesh");
    p_grid_->setRequired(0);
    p_gdata_ = addInputPort("gdataIn", "Float", "input grid scalar data.");
    p_gdata_->setRequired(0);
    p_goctree_ = addInputPort("gOcttreesIn", "OctTree", "input grid octtrees");
    p_goctree_->setRequired(0);
    p_poly_ = addInputPort("polyIn", "Polygons", "input polygons");
    p_poly_->setRequired(0);
    p_pdata_ = addInputPort("pdataIn", "Float", "input polygon scalar data");
    p_pdata_->setRequired(0);
    p_poctree_ = addInputPort("pOcttreesIn", "OctTreeP", "input polygon octtrees");
    p_poctree_->setRequired(0);
    p_colorMapIn_ = addInputPort("colorMapIn", "ColorMap", "color map to create geometry");
    p_colorMapIn_->setRequired(0);

    p_gout_ = addOutputPort("ggeometry", "Geometry", "grid Geometry output");

    // parameters
    const char *dimensionChoices[] = { "3d", "poly" };
    p_dimension_ = addChoiceParam("dimensionality", "3d, poly");
    p_dimension_->setValue(2, dimensionChoices, 0); // FIXME
    const char *probeTypeChoices[] = { "point", "square", "cube" };
    p_icon_type_ = addChoiceParam("probe_type", "point, square, cube");
    p_icon_type_->setValue(3, probeTypeChoices, 0); // FIXME

    p_start1_ = addFloatVectorParam("startpoint1", "startpoint1");
    p_start1_->setValue(0.0, 0.0, 0.0);
    p_start2_ = addFloatVectorParam("startpoint2", "startpoint2");
    p_start2_->setValue(0.0, 0.0, 1.0);
    p_direction_ = addFloatVectorParam("direction", "direction");
    p_direction_->setValue(0.0, 0.0, 1.0);

    p_numsidepoints_ = addIntSliderParam("numsidepoints",
                                         "number of side points");
    p_numsidepoints_->setValue(8, 20, 15);
}

Probe3D::~Probe3D()
{
}
void
Probe3D::param(const char *portName, bool /*inMapLoading*/)
{
    if (strcmp(portName, p_start1_->getName()) == 0 || strcmp(portName, p_start2_->getName()) == 0 || strcmp(portName, p_direction_->getName()) == 0)
    {
        float s1[3], s2[3];
        p_start1_->getValue(s1[0], s1[1], s1[2]);
        p_start2_->getValue(s2[0], s2[1], s2[2]);
        // take middle of line between two starting points as base point
        point_[0] = 0.5f * (s1[0] + s2[0]);
        point_[1] = 0.5f * (s1[1] + s2[1]);
        point_[2] = 0.5f * (s1[2] + s2[2]);

        float s[3], d[3];
        s[0] = s2[0] - s1[0];
        s[1] = s2[1] - s1[1];
        s[2] = s2[2] - s1[2];

        side_ = sqrt(0.5f * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]));

        p_direction_->getValue(d[0], d[1], d[2]);
        dir_[0] = d[0];
        dir_[1] = d[1];
        dir_[2] = d[2];

        normal_[0] = s[1] * d[2] - s[2] * d[1];
        normal_[1] = s[2] * d[0] - s[0] * d[2];
        normal_[2] = s[0] * d[1] - s[1] * d[0];
    }
}

int
Probe3D::compute(const char *)
{
    grid_tsteps_.clear();
    gdata_tsteps_.clear();
    prid_tsteps_.clear();
    pdata_tsteps_.clear();

    gridIsTimeDependent_ = false;
    polyIsTimeDependent_ = false;
    // setSurname ensures that created octrees get
    // a unique name
    gBBoxAdmin_.setSurname();
    pBBoxAdmin_.setSurname();

    if (firstComputation())
    {
        bool diagnose = getDiagnose();
        if (diagnose)
        {
            loadNames();
            if (p_grid_->getCurrentObject())
                gBBoxAdmin_.load(p_grid_->getCurrentObject(), p_goctree_->getCurrentObject());
            if (p_poly_->getCurrentObject())
                pBBoxAdmin_.load(p_poly_->getCurrentObject(), p_poctree_->getCurrentObject());
        }
        else
        { // bad data
            eraseNames();
            return FAIL;
        }
    }
    else
    { // not the first computation, data was already
        // successfully checked
        if (p_grid_->getCurrentObject())
            gBBoxAdmin_.reload(p_grid_->getCurrentObject(), p_goctree_->getCurrentObject());
        if (p_poly_->getCurrentObject())
            pBBoxAdmin_.reload(p_poly_->getCurrentObject(), p_poctree_->getCurrentObject());
    }

    if (p_dimension_->getValue() == 0 && p_grid_->getCurrentObject())
    { // not "3d"
        expandSets(p_grid_, grid_tsteps_, &gridIsTimeDependent_);
        expandSets(p_gdata_, gdata_tsteps_);
    }
    if (p_dimension_->getValue() == 1 && p_poly_->getCurrentObject())
    { // "poly"
        expandSets(p_poly_, prid_tsteps_, &polyIsTimeDependent_);
        expandSets(p_pdata_, pdata_tsteps_);
    }
    assignOctTrees(&gBBoxAdmin_, grid_tsteps_);
    assignOctTrees(&pBBoxAdmin_, prid_tsteps_);

    // so we may now proceed...
    // first get the list of points
    vector<float> x, y, z;
    loadPoints(x, y, z);

    // gresults, presults are as long as the number of time steps,
    // and for each time step, the list is as long as x,y,z
    vector<vector<float> > gresults, presults;
    //BEGINN Maximumberechnung
    const coDistributedObject *gridObj = p_grid_->getCurrentObject();
    const coDistributedObject *dobj_in = p_gdata_->getCurrentObject();

    //spaeter   coDoSet *dset_in;
    //spaeter   int dnum_sets;
    float min = FLT_MAX, max = -FLT_MAX;
    float avg = 0.0;
    int num_avg = 0; // number of valid points to build average
    //spaeter   coDistributedObject * const *delem_in;
    if (dobj_in->isType("SETELE"))
    {
        //spaeter       dset_in = (coDoSet *) dobj_in;
        //       if( obj_in->getAttribute("TIMESTEP") != NULL ) {
        //          delem_in = dset_in->getAllElements(&dnum_sets);

        //       }
    }
    else
    {
        //FIXME noch keine Vector-typen
        if (gridObj->isType("STRGRD") || gridObj->isType("UNIGRD") || gridObj->isType("RCTGRD"))
        {
            coDoFloat *s_data = (coDoFloat *)dobj_in;
            int i, npoint;
            float *scalar;
            npoint = s_data->getNumPoints();
            s_data->getAddress(&scalar);
            for (i = 0; i < npoint; i++)
            {
                if (scalar[i] != FLT_MAX)
                {
                    avg += scalar[i];
                    num_avg++;
                    if (scalar[i] > max)
                        max = scalar[i];
                    if (scalar[i] < min)
                        min = scalar[i];
                }
            }
            if (num_avg)
            {
                avg /= (float)num_avg;
            }
            else
            {
                avg = FLT_MAX;
            }
        }
        else if (gridObj->isType("UNSGRD"))
        {
            coDoFloat *unsData = (coDoFloat *)dobj_in;
            int npoint;
            float *scalar;
            int i;
            npoint = unsData->getNumPoints();
            unsData->getAddress(&scalar);
            for (i = 0; i < npoint; i++)
            {
                if (scalar[i] != FLT_MAX)
                {
                    avg += scalar[i];
                    num_avg++;
                    if (scalar[i] > max)
                        max = scalar[i];
                    if (scalar[i] < min)
                        min = scalar[i];
                }
            }
            if (num_avg)
            {
                avg /= (float)num_avg;
            }
            else
            {
                avg = FLT_MAX;
            }
        }
    }

    char minattr[100], maxattr[100];
    sprintf(minattr, "%f", min);
    sprintf(maxattr, "%f", max);
    coDistributedObject *geometry = NULL;
    if (p_dimension_->getValue() == 0 && p_grid_->getCurrentObject())
    { //  "3d"
        gInterpolate(x, y, z, grid_tsteps_, gdata_tsteps_, gresults);
        geometry = gOutput(gresults, min, max, avg);
    }
    if (p_dimension_->getValue() == 1 && p_poly_->getCurrentObject())
    { // "poly"
        gInterpolate(x, y, z, prid_tsteps_, pdata_tsteps_, presults);
        geometry = pOutput(presults, "PROBE3D_POINT", min, max);
    }
    if (NULL != geometry)
    {
        const char **names;
        const char **atts;
        int size = dobj_in->getAllAttributes(&names, &atts);
        int i;
        for (i = 0; i < size; i++)
        {
            if (0 == strcmp(names[i], "SPECIES"))
            {
                geometry->addAttribute("SPECIES", atts[i]);
            }
        }
    };

    //ENDE MAXimumberechnugn
    /*
   if(distObj != NULL){
      string name(p_gout_->getObjName());
      coDistributedObject *setElemList[] = {NULL};
      name += geometry;
      distObj = new coDoSet(name,setElemList);
      name = p_gout_->getObjName();
   }
*/

    return SUCCESS;
}

coDoFloat *
Probe3D::makeField(const char *name, const vector<float> &field) const
{
    return new coDoFloat(name, (int)field.size(), field.size() > 0 ? const_cast<float *>(&field[0]) : NULL);
}

coDoPolygons *
Probe3D::makePolygon(const char *name) const
{
    int num_side_nodes = p_numsidepoints_->getValue() + 1;
    int no_nodes = num_side_nodes * num_side_nodes;
    int no_poly = (num_side_nodes - 1) * (num_side_nodes - 1);
    int no_vertices = 4 * no_poly;
    vector<float> x, y, z;
    loadPoints(x, y, z, true);

    vector<int> vl, pl;
    int i, j;
    for (j = 0; j < num_side_nodes - 1; ++j)
    {
        int base = j * num_side_nodes;
        int basen = base + num_side_nodes;
        for (i = 0; i < num_side_nodes - 1; ++i)
        {
            pl.push_back((int)vl.size());
            vl.push_back(base + i);
            vl.push_back(base + i + 1);
            vl.push_back(basen + i + 1);
            vl.push_back(basen + i);
        }
    }
    coDoPolygons *poly = new coDoPolygons(name, no_nodes,
                                          x.size() > 0 ? &x[0] : NULL,
                                          y.size() > 0 ? &y[0] : NULL,
                                          z.size() > 0 ? &z[0] : NULL,
                                          no_vertices, vl.size() > 0 ? &vl[0] : NULL,
                                          no_poly, pl.size() > 0 ? &pl[0] : NULL);
    poly->addAttribute("vertexOrder", "2");
    return poly;
}

static void
redressColor(coDistributedObject *color, const vector<vector<float> > &gresults)
{
    if (color->isType("SETELE") && color->getAttribute("TIMESTEP"))
    {
        const coDistributedObject *const *setList = NULL;
        int num;
        setList = ((coDoSet *)(color))->getAllElements(&num);
        assert(num == gresults.size());
        int elem;
        for (elem = 0; elem < num; ++elem)
        {
            if (setList[elem]->isType("RGBADT"))
            {
                vector<vector<float> > temp;
                temp.resize(1);
                temp[0] = gresults[elem];
                redressColor(const_cast<coDistributedObject *>(setList[elem]), temp);
            }
        }
    }
    else if (color->isType("RGBADT"))
    {
        coDoRGBA *Color = (coDoRGBA *)color;
        assert(gresults.size() == 1);
        int i;
        for (i = 0; i < gresults[0].size(); ++i)
        {
            if (gresults[0][i] == FLT_MAX)
                Color->setFloatRGBA(i, 1.0, 1.0, 1.0, 1.0);
        }
    }
}

// in this case we may produce a polygon with colours
// or numeric information
coDistributedObject *
Probe3D::gOutput(const vector<vector<float> > &gresults, float min, float max, float avg)
{
    // square probing
    char attrString[1000];
    float smin, smax, savg;

    coDistributedObject *scalar = NULL;
    string name = p_gout_->getObjName();
    string name_color = p_gout_->getObjName();
    string name_field = p_gout_->getObjName();
    name += "_geometry";
    name_color += "_color";
    name_field += "_field";
    coDistributedObject *const nullSetList[] = { NULL };
    coDistributedObject *geometry;
    if (gresults.size() == 0)
    { // no results

        coDoSet *staticSet = new coDoSet(name.c_str(), nullSetList);
        geometry = staticSet;
    }
    else if (gridIsTimeDependent_)
    { // dynamic case
        coDistributedObject **setList = new coDistributedObject *[gresults.size() + 1];
        coDistributedObject **setListField = new coDistributedObject *[gresults.size() + 1];
        setList[gresults.size()] = NULL;
        setListField[gresults.size()] = NULL;
        int i;
        for (i = 0; i < gresults.size(); ++i)
        {
            string name_i(name);
            string name_field_i(name_field);
            char buf[32];
            sprintf(buf, "_%d", i);
            name_i += buf;
            name_field_i += buf;
            setList[i] = makePolygon(name_i.c_str());
            setListField[i] = makeField(name_field_i.c_str(), gresults[i]);
            staticMinMaxCalculation(smin, smax, savg, gresults[i]);
            sprintf(attrString, "%f %f %f", smin, smax, savg);
            setListField[i]->addAttribute("PROBE3D_SQUARE", attrString);
        }
        coDoSet *dynamicSet = new coDoSet(name.c_str(), setList);
        coDoSet *dynamicSetField = new coDoSet(name_field.c_str(), setListField);
        char buf[62];
        sprintf(buf, "1 %d", (int)gresults.size());

        dynamicSet->addAttribute("TIMESTEP", buf);
        dynamicSetField->addAttribute("TIMESTEP", buf);
        geometry = dynamicSet;
        scalar = dynamicSetField;
    }
    else
    { // static case
        if (p_icon_type_->getValue() == 1)
        {
            coDoPolygons *poly = makePolygon(name.c_str());
            coDoFloat *field = makeField(name_field.c_str(), gresults[0]);
            staticMinMaxCalculation(smin, smax, savg, gresults[0]);
            sprintf(attrString, "%f %f %f", smin, smax, savg);
            poly->addAttribute("PROBE3D_SQUARE", attrString);
            geometry = poly;
            scalar = field;
        }
        else
        {
            coDoPoints *points = new coDoPoints(name.c_str(), 1);
            float *xStart, *yStart, *zStart;
            points->getAddresses(&xStart, &yStart, &zStart);
            if (p_icon_type_->getValue() == 0)
            { // point
                *xStart = point_[0];
                *yStart = point_[1];
                *zStart = point_[2];
            }
            else
            { // centre of cube
                float normal1[3];
                normal1[0] = normal_[0];
                normal1[1] = normal_[1];
                normal1[2] = normal_[2];
                normalise(normal1);

                *xStart = point_[0] + normal1[0] * side_ * 0.5f;
                *yStart = point_[1] + normal1[1] * side_ * 0.5f;
                *zStart = point_[2] + normal1[2] * side_ * 0.5f;

                staticMinMaxCalculation(smin, smax, savg, gresults[0]);
                sprintf(attrString, "%f %f %f", smin, smax, savg);
                points->addAttribute("PROBE3D_CUBE", attrString);
            }

            geometry = points;
        }
    }
    coDoGeometry *do_geom = new coDoGeometry(p_gout_->getObjName(), geometry);
    coFeedback feedback("Probe3D");
    feedback.addPara(p_dimension_);
    feedback.addPara(p_icon_type_);
    feedback.addPara(p_start1_);
    feedback.addPara(p_start2_);
    feedback.addPara(p_direction_);
    feedback.addPara(p_numsidepoints_);
    feedback.apply(geometry);

    if (scalar)
    {
        float dataMin = FLT_MAX;
        float dataMax = -FLT_MAX;
        transientMinMaxCalculation(dataMin, dataMax, avg, gresults);

        // min max are calculated from gresults discarding FLT_MAX...
        coDistributedObject *color = NULL;
        if (!p_colorMapIn_->getCurrentObject())
        {
            color = ComplexModules::DataTexture(name_color,
                                                scalar, NULL, false, 1, &dataMin, &dataMax);
        }
        else
        {
            color = ComplexModules::DataTexture(name_color,
                                                scalar,
                                                p_colorMapIn_->getCurrentObject(), false);
        }
        redressColor(color, gresults);
        do_geom->setColors(PER_VERTEX, color);
        scalar->destroy();
    }

    if (p_icon_type_->getValue() == 0)
    {
        //FIXME time dependend not taken into account
        float value = gresults[0][0];
        sprintf(attrString, "%f %f %f", min, max, value);
        geometry->addAttribute("PROBE3D_POINT", attrString);
    }

    p_gout_->setCurrentObject(do_geom);

    return geometry;
}

void
Probe3D::transientMinMaxCalculation(float &min, float &max, float &avg,
                                    const vector<vector<float> > &gresults) const
{
    min = FLT_MAX;
    max = -FLT_MAX;

    int time;
    int point;
    avg = 0.0;
    int timesteps = (int)gresults.size();
    for (time = 0; time < timesteps; ++time)
    {
        float avg2 = 0.0;
        float tsize = 1.0;
        for (point = 0; point < gresults[time].size(); ++point)
        {
            tsize = float(point);
            if (gresults[time][point] == FLT_MAX)
                continue;
            avg2 += gresults[time][point];
            if (gresults[time][point] > max)
                max = gresults[time][point];
            if (gresults[time][point] < min)
                min = gresults[time][point];
        }
        avg += avg2 / (float)tsize;
    }
    avg /= (float)timesteps;
}

void
Probe3D::staticMinMaxCalculation(float &min, float &max, float &avg,
                                 const vector<float> &gresults) const
{
    int i, num_avg = 0;
    avg = 0.;
    min = FLT_MAX;
    max = -FLT_MAX;
    for (i = 0; i < gresults.size(); i++)
    {
        if (gresults[i] != FLT_MAX)
        {
            avg += gresults[i];
            num_avg++;
            if (gresults[i] > max)
                max = gresults[i];
            if (gresults[i] < min)
                min = gresults[i];
        }
    }
    if (num_avg)
    {
        avg /= (float)num_avg;
    }
    else
    {
        avg = FLT_MAX;
    }
}

// in this case only-numeric information is produced
coDistributedObject *
Probe3D::pOutput(const vector<vector<float> > &presults,
                 const char *pointAttribute, float min, float max)
{
    coDistributedObject *geometry;
    // if we are using a square probe pOutput has nothing to do
    // and distObj must be NULL
    if (p_icon_type_->getValue() == 1)
        return NULL;

    string name = p_gout_->getObjName();
    name += "_geometry";

    coDoPoints *points = new coDoPoints(name.c_str(), 1);
    float *xStart, *yStart, *zStart;
    points->getAddresses(&xStart, &yStart, &zStart);
    *xStart = point_[0];
    *yStart = point_[1];
    *zStart = point_[2];

    geometry = points;

    // produce a value for attribute pointAttribute
    // given by a series of all strings with input numerical
    // values
    int i;
    string attributeValue;
    char buf[32];

    //Get rid of warnigs for 32- and 64- bit platforms
    sprintf(buf, "%d", (int)presults.size()); // how many time steps

    for (i = 0; i < presults.size(); ++i)
    {

        attributeValue += ' ';
        char buf[32];
        sprintf(buf, "%g", presults[i][0]);
        attributeValue += buf;
    }
    char attrBuf[1000]; // Min Max value
    //FIXME time dependent not taken into account
    sprintf(attrBuf, "%f %f %s", min, max, attributeValue.c_str());

    geometry->addAttribute(pointAttribute, attrBuf);
    // feedback for the cover plugin
    coFeedback feedback("Probe3D");
    feedback.addPara(p_dimension_);
    feedback.addPara(p_icon_type_);
    feedback.addPara(p_start1_);
    feedback.addPara(p_start2_);
    feedback.addPara(p_direction_);
    feedback.addPara(p_numsidepoints_);
    feedback.apply(geometry);

    coDoGeometry *do_geom = new coDoGeometry(p_gout_->getObjName(), geometry);

    p_gout_->setCurrentObject(do_geom);

    return geometry;
}

void
Probe3D::loadPoints(vector<float> &x, vector<float> &y, vector<float> &z, bool per_cell) const
{
    if (p_icon_type_->getValue() == 1)
    { // point
        x.push_back(point_[0]);
        y.push_back(point_[1]);
        z.push_back(point_[2]);
    }
    else
    { // square or cube
        vector<float> point, normal;

        point.push_back(point_[0]);
        point.push_back(point_[1]);
        point.push_back(point_[2]);

        normal.push_back(normal_[0]);
        normal.push_back(normal_[1]);
        normal.push_back(normal_[2]);

        if (p_icon_type_->getValue() == 1)
        { //square
            loadSquarePoints(p_numsidepoints_->getValue(), point, normal,
                             side_, x, y, z, per_cell);
        }
        else
        { // cube
            loadSquarePoints(p_numsidepoints_->getValue(), point, normal,
                             side_, x, y, z, per_cell);
            float normal1[3];
            normal1[0] = normal[0];
            normal1[1] = normal[1];
            normal1[2] = normal[2];
            normalise(normal1);
            int steps = p_numsidepoints_->getValue();
            for (int i = 1; i <= steps; i++)
            {
                point[0] += normal1[0] * side_ * 1.0f / steps;
                point[1] += normal1[1] * side_ * 1.0f / steps;
                point[2] += normal1[2] * side_ * 1.0f / steps;

                loadSquarePoints(p_numsidepoints_->getValue(), point, normal,
                                 side_, x, y, z, per_cell);
            }
        }
    }
}

void
Probe3D::loadSquarePoints(int num_sidepoints, vector<float> point,
                          vector<float> normal, float side_length,
                          vector<float> &x, vector<float> &y, vector<float> &z, bool per_cell) const
{
    float e1[3], e2[3], e3[3];
    e3[0] = normal[0];
    e3[1] = normal[1];
    e3[2] = normal[2];
    normalise(e3);

    /*
    if(e3[0]*e3[0]<=e3[1]*e3[1] && e3[0]*e3[0]<=e3[2]*e3[2]){
       e1[0] = 0.0;
       e1[1] = e3[2];
       e1[2] = -e3[1];
    }
    else if(e3[1]*e3[1]<=e3[0]*e3[0] && e3[1]*e3[1]<=e3[2]*e3[2]){
       e1[1] = 0.0;
       e1[0] = e3[2];
       e1[2] = -e3[0];
    }
    else {
       e1[2] = 0.0;
       e1[0] = e3[1];
       e1[1] = -e3[0];
    }
*/

    // test
    e1[0] = dir_[0];
    e1[1] = dir_[1];
    e1[2] = dir_[2];

    normalise(e1);

    vecProd(e3, e1, e2);
    // e1, e1 define the probing plane
    int i, j;
    int num = num_sidepoints;
    float side = side_length;
    float delta = side / (num - 1);
    float centre[3];

    if (per_cell)
    {
        centre[0] = point[0] - (side + delta) * 0.5f * (e1[0] + e2[0]);
        centre[1] = point[1] - (side + delta) * 0.5f * (e1[1] + e2[1]);
        centre[2] = point[2] - (side + delta) * 0.5f * (e1[2] + e2[2]);
        num++;
    }
    else
    {
        centre[0] = point[0] - side * 0.5f * (e1[0] + e2[0]);
        centre[1] = point[1] - side * 0.5f * (e1[1] + e2[1]);
        centre[2] = point[2] - side * 0.5f * (e1[2] + e2[2]);
    }
    for (j = 0; j < num; ++j)
    {
        for (i = 0; i < num; ++i)
        {
            x.push_back(centre[0] + delta * i * e1[0] + delta * j * e2[0]);
            y.push_back(centre[1] + delta * i * e1[1] + delta * j * e2[1]);
            z.push_back(centre[2] + delta * i * e1[2] + delta * j * e2[2]);
        }
    }
}

void
Probe3D::interpolateForAGrid(const float *coordinates, float *result,
                             const coDistributedObject *grid,
                             const coDistributedObject *field)
{
    int cell[3] = { -1, -1, -1 };
    float coords[3];
    coords[0] = coordinates[0];
    coords[1] = coordinates[1];
    coords[2] = coordinates[2];
    if (grid->isType("UNIGRD") && field->isType("USTSDT"))
    {
        coDoUniformGrid *p_uni_grid = (coDoUniformGrid *)(grid);
        coDoFloat *p_uni_field = (coDoFloat *)(field);
        float *u;
        int x_s, y_s, z_s;
        p_uni_grid->getGridSize(&x_s, &y_s, &z_s);
        int np = p_uni_field->getNumPoints();
        p_uni_field->getAddress(&u);

        if (np == x_s * y_s * z_s && p_uni_grid->interpolateField(result, coords, cell, 1, 1, &u) == 0)
        {
            return;
        }
        else
        {
            *result = FLT_MAX;
        }
    }
    else if (grid->isType("RCTGRD") && field->isType("USTSDT"))
    {
        coDoRectilinearGrid *p_rct_grid = (coDoRectilinearGrid *)(grid);
        coDoFloat *p_rct_field = (coDoFloat *)(field);
        float *u;
        int x_s, y_s, z_s;
        p_rct_grid->getGridSize(&x_s, &y_s, &z_s);
        p_rct_field->getAddress(&u);
        int np = p_rct_field->getNumPoints();

        if (np == x_s * y_s * z_s && p_rct_grid->interpolateField(result, coords, cell, 1, 1, &u) == 0)
        {
            return;
        }
        else
        {
            *result = FLT_MAX;
        }
    }
    else if (grid->isType("STRGRD") && field->isType("USTSDT"))
    {
        coDoStructuredGrid *p_str_grid = (coDoStructuredGrid *)(grid);
        coDoFloat *p_str_field = (coDoFloat *)(field);
        float *u;
        int x_s, y_s, z_s;
        p_str_grid->getGridSize(&x_s, &y_s, &z_s);
        p_str_field->getAddress(&u);
        int np = p_str_field->getNumPoints();

        if (np == x_s * y_s * z_s && p_str_grid->interpolateField(result, coords, cell, 1, 1, &u) == 0)
        {
            return;
        }
        else
        {
            *result = FLT_MAX;
        }
    }
    else if (grid->isType("UNSGRD") && field->isType("USTSDT"))
    {
        coDoUnstructuredGrid *p_uns_grid = (coDoUnstructuredGrid *)(grid);
        coDoFloat *p_uns_field = (coDoFloat *)(field);

        // get sizes for comparison
        int nume, numv, numc;
        p_uns_grid->getGridSize(&nume, &numv, &numc);
        int numdc;
        numdc = p_uns_field->getNumPoints();
        float *u; // vector field
        p_uns_field->getAddress(&u);
        if (numc == numdc && p_uns_grid->interpolateField(result, coords, cell, 1, 1, 2.5e-3f, &u) == 0)
        {
            return;
        }
        else
        {
            *result = FLT_MAX;
        }
    }
    else if (grid->isType("POLYGN") && field->isType("USTSDT"))
    {
        const coDoPolygons *p_pol_grid = dynamic_cast<const coDoPolygons *>(grid);
        const coDoFloat *p_pol_field = dynamic_cast<const coDoFloat *>(field);
        // get sizes for comparison
        int numc;
        numc = p_pol_grid->getNumPoints();
        int numdc;
        numdc = p_pol_field->getNumPoints();
        float *u; // vector field
        p_pol_field->getAddress(&u);
        //float *y_cheater = const_cast<float *>(y);
        if (numc == numdc
            && p_pol_grid->interpolateField(result, coords, cell, 1, 1,
                                            2.5e-3f, &u, 1) == 0)
        {
            point_[0] = coords[0];
            point_[1] = coords[1];
            point_[2] = coords[2];
            p_start1_->setValue(point_[0] + 1, point_[1] + 1, point_[2]);
            p_start2_->setValue(point_[0] - 1, point_[1] - 1, point_[2]);
            return;
        }
        else
        {
            *result = FLT_MAX;
        }
    }
}

void
Probe3D::gInterpolate(const vector<float> &x,
                      const vector<float> &y,
                      const vector<float> &z,
                      const vector<vector<const coDistributedObject *> > &grid_tsteps,
                      const vector<vector<const coDistributedObject *> > &gdata_tsteps,
                      vector<vector<float> > &gresults)
// gresults is as long as the number of time steps,
// and each element (again a vector) is as long as the number of points
// when an interpolation fails for all grids in a time step,
// then FLT_MAX ist written
{
    int time;
    gresults.clear();
    gresults.resize(grid_tsteps.size());
    for (time = 0; time < grid_tsteps.size(); ++time)
    {
        vector<float> &time_result = gresults[time];
        const vector<const coDistributedObject *> &grids = grid_tsteps[time];
        const vector<const coDistributedObject *> &field = gdata_tsteps[time];
        int point;
        // loop over points
        for (point = 0; point < x.size(); ++point)
        {
            float coordinates[3];
            coordinates[0] = x[point];
            coordinates[1] = y[point];
            coordinates[2] = z[point];
            // loop over grids
            float result = FLT_MAX;
            int grid;
            for (grid = 0; grid < grids.size(); ++grid)
            {
                interpolateForAGrid(coordinates, &result, grids[grid], field[grid]);
                if (result != FLT_MAX)
                    break;
            }
            time_result.push_back(result);
        }
    }
}

bool
Probe3D::firstComputation() const
{
    string gName(p_grid_->getCurrentObject() ? p_grid_->getCurrentObject()->getName() : "");
    string gfName(p_gdata_->getCurrentObject() ? p_gdata_->getCurrentObject()->getName() : "");
    string gtName(p_goctree_->getCurrentObject() ? p_goctree_->getCurrentObject()->getName() : "");
    string pName(p_poly_->getCurrentObject() ? p_poly_->getCurrentObject()->getName() : "");
    string pfName(p_pdata_->getCurrentObject() ? p_pdata_->getCurrentObject()->getName() : "");
    string ptName(p_poctree_->getCurrentObject() ? p_poctree_->getCurrentObject()->getName() : "");

    bool notret = (gName == gridName_ && gfName == gfieldName_ && gtName == gtreeName_ && pName == pridName_ && pfName == pfieldName_ && ptName == ptreeName_);
    return !notret;
}

bool
Probe3D::getDiagnose() const
{
    if ((p_grid_->getCurrentObject() && !p_gdata_->getCurrentObject())
        || (!p_grid_->getCurrentObject() && p_gdata_->getCurrentObject()))
    {
        sendError("Grid has no data or grid data has no no grid");
        return false;
    }
    else if ((p_poly_->getCurrentObject() && !p_pdata_->getCurrentObject())
             || (!p_poly_->getCurrentObject() && p_pdata_->getCurrentObject()))
    {
        sendError("Polygons has no data or polygon data has no no polygon");
        return false;
    }
    return true;
}

void
Probe3D::loadNames()
{
    gridName_ = p_grid_->getCurrentObject() ? p_grid_->getCurrentObject()->getName() : "";
    gfieldName_ = p_gdata_->getCurrentObject() ? p_gdata_->getCurrentObject()->getName() : "";
    gtreeName_ = p_goctree_->getCurrentObject() ? p_goctree_->getCurrentObject()->getName() : "";
    pridName_ = p_poly_->getCurrentObject() ? p_poly_->getCurrentObject()->getName() : "";
    pfieldName_ = p_pdata_->getCurrentObject() ? p_pdata_->getCurrentObject()->getName() : "";
    ptreeName_ = p_poctree_->getCurrentObject() ? p_poctree_->getCurrentObject()->getName() : "";
}

void
Probe3D::eraseNames()
{
    gridName_ = "";
    gfieldName_ = "";
    gtreeName_ = "";
    pridName_ = "";
    pfieldName_ = "";
    ptreeName_ = "";
}

void
Probe3D::assignOctTrees(const BBoxAdmin *bboxAdmin,
                        vector<vector<const coDistributedObject *> > &objects)
{
    bboxAdmin->assignOctTrees(objects);
}

static void
staticExpandSet(const coDistributedObject *obj, vector<const coDistributedObject *> &list)
{
    if (obj->isType("SETELE"))
    {
        int num_elems;
        const coDistributedObject *const *set_list = NULL;
        set_list = ((coDoSet *)(obj))->getAllElements(&num_elems);
        int i;
        for (i = 0; i < num_elems; ++i)
        {
            staticExpandSet(set_list[i], list);
        }
        return;
    }
    list.push_back(obj);
}

void
Probe3D::expandSets(coInputPort *port, vector<vector<const coDistributedObject *> > &tsteps,
                    bool *timeDependent)
{
    if (!port->getCurrentObject())
        return;
    const coDistributedObject *obj = port->getCurrentObject();
    if (obj->isType("SETELE") && obj->getAttribute("TIMESTEP"))
    {
        if (timeDependent)
            *timeDependent = true;
        // get the element list and length
        const coDistributedObject *const *set_list = NULL;
        int num_elems;
        set_list = ((coDoSet *)(obj))->getAllElements(&num_elems);
        // resize tsteps accordingly
        tsteps.resize(num_elems);
        // for each time step call a staticExpandSet function
        int i;
        for (i = 0; i < num_elems; ++i)
        {
            staticExpandSet(set_list[i], tsteps[i]);
        }
        return;
    }
    // there is a unique "static" time step
    // expand it using a static function
    tsteps.resize(1);
    staticExpandSet(obj, tsteps[0]);
}

MODULE_MAIN(Tools, Probe3D)
