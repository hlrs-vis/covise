/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SimplifySurfaceNT.h"
#include "Point.h"
#include "EdgeCollapse.h"
#include "EdgeCollapseSimple.h"
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <alg/coFeatureLines.h>
#include <config/CoviseConfig.h>

#ifdef HAVE_VTK
#include <vtkVersion.h>
#include <vtkPolyData.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkQuadricDecimation.h>
#include <vtkDecimatePro.h>
#include <vtkQuadricClustering.h>

#include <vtkTriangleFilter.h>
#include <vtkCleanPolyData.h>
#include <vtkPointLocator.h>

#include <vtkDataObjectWriter.h>

#if VTK_MAJOR_VERSION < 6
#define SetInputData SetInput
#endif
#endif

//#include <algorithm>

using std::binary_search;

const float SimplifySurface::PERCENT_DEFAULT = 30.0;

SimplifySurface::SimplifySurface(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Surface simplification with optional data")
{
    p_meshIn = addInputPort("meshIn", "Polygons|TriangleStrips", "Input geometry");
    int port;
    for (port = 0; port < NUM_DATA; ++port)
    {
        ostringstream streamNum;
        streamNum << "dataIn_" << port;
        string num = streamNum.str();
        p_dataIn[port] = addInputPort(num.c_str(), "Float|Vec3", "Vertice-attached data");
        p_dataIn[port]->setRequired(0);
    }
    p_normalsIn = addInputPort("normalsIn", "Vec3", "Vertice-attached normals");
    p_normalsIn->setRequired(0);

    p_meshOut = addOutputPort("meshOut", "Polygons", "The reduced geometry");
    for (port = 0; port < NUM_DATA; ++port)
    {
        ostringstream streamNum;
        streamNum << "dataOut_" << port;
        string num = streamNum.str();
        p_dataOut[port] = addOutputPort(num.c_str(), "Float|Vec3", "The interpolated data");
        p_dataOut[port]->setDependencyPort(p_dataIn[port]);
    }
    p_normalsOut = addOutputPort("normalsOut", "Vec3", "The interpolated normals");
    p_normalsOut->setDependencyPort(p_normalsIn);

#ifdef HAVE_VTK
    const char *method_labels[] = { "EdgeCollapse", "QuadricClustering", "DecimatePro", "QuadricDecimation" };
    p_method = addChoiceParam("method", "simplification algorithm");
    p_method->setValue(4, method_labels, 0);
#endif

    param_percent = addFloatParam("percent", "Percentage of triangles to be left after simplification");
    param_percent->setValue(PERCENT_DEFAULT);

    param_normaldeviation = addFloatParam("max_normaldeviation", "maximal normal deviation");
    param_normaldeviation->setValue(3);

    param_domaindeviation = addFloatSliderParam("max_domaindeviation", "maximal domain deviation");
    param_domaindeviation->setValue(0.3f, 4.0f, 2.0f);

    param_datarelativeweight = addFloatParam("data_relative_weight", "data relative weight");
    param_datarelativeweight->setValue(0.05f);

    param_ignoredata = addBooleanParam("ignore_data", "Performcs simplification independent from data values");

#ifdef HAVE_VTK
    param_divisions = addFloatVectorParam("divisions", "divisions in x, y and z direction (only for QuadricClustering)");
    param_divisions->setValue(0.1, 0.1, 0.1);

    param_divisions_absolute = addBooleanParam("divisons_are_absolute", "if true: interpret divisions as lengths (otherwise take it as numbers of division)");
    param_divisions_absolute->setValue(true);

    param_smoothSurface = addBooleanParam("smooth_surface", "smooth surface");
    param_smoothSurface->setValue(false);

    param_preserveTopology = addBooleanParam("preserve_topology", "Turn on/off whether to preserve the topology of the original mesh. If on, mesh splitting and hole elimination will not occur. This may limit the maximum reduction that may be achieved.");
    param_preserveTopology->setValue(true);

    param_meshSplitting = addBooleanParam("mesh_splitting", "Turn on/off the splitting of the mesh at corners, along edges, at non-manifold points, or anywhere else a split is required. Turning splitting off will better preserve the original topology of the mesh, but you may not obtain the requested reduction.");
    param_meshSplitting->setValue(true);

    param_splitAngle = addFloatParam("split_angle", "Specify the mesh split angle. A split line exists when the surface normals between two edges are >= SplitAngle");
    param_splitAngle->setValue(75.);

    param_featureAngle = addFloatParam("feature_angle", "Specify the mesh feature angle. This angle is used to define what an edge is (normals >= FeatureAngle, an edge exists)");
    param_featureAngle->setValue(15.);

    param_boundaryVertexDeletion = addBooleanParam("boundary_vertex_deletion", "allow deletion of boundary vertices?");
    param_boundaryVertexDeletion->setValue(true);

    param_maximumError = addFloatParam("maximum_error", "specified as a fraction of the maximum boundary box length");
    param_maximumError->setValue(0.03);
#endif
    /*
      param_boundaryfactor = addFloatParam("boundary_factor",
         "boundary factor");
      param_boundaryfactor->setValue(1000.0);

      param_valence = addInt32Param("max_valence",
         "Max valence");
      param_valence->setValue(200);

      const char * choice[] = {"complex","simple"};
      param_algo = addChoiceParam("algorithm","algorithm");
   param_algo->setValue(2,choice,1);
   */

    cf_BoundaryFactor = coCoviseConfig::getFloat("Module.SimplifySurface.BoundaryFactor", 1000.0);

    cf_MaxValence = coCoviseConfig::getInt("Module.SimplifySurface.MaxValence", 200);

    cf_Algorithm = coCoviseConfig::getInt("Module.SimplifySurface.Algorithm", 2);
}

float max_cos_2;
float normaldeviation_cos;
float domaindeviation_cos;
float boundary_factor;
bool ignoreData;

#ifdef HAVE_VTK
void SimplifySurface::postInst()
{
    param_percent->enable();
    param_normaldeviation->enable();
    param_domaindeviation->enable();
    param_datarelativeweight->enable();
    param_ignoredata->enable();
    param_divisions->disable();
    param_divisions_absolute->disable();
    param_smoothSurface->disable();
    param_preserveTopology->disable();
    param_meshSplitting->disable();
    param_boundaryVertexDeletion->disable();
    param_maximumError->disable();
    param_splitAngle->disable();
    param_featureAngle->disable();
}

void SimplifySurface::param(const char *paramname, bool /*inMapLoading*/)
{
    if (strcmp(paramname, p_method->getName()) == 0)
    {
        switch (p_method->getValue())
        {
        case EDGECOLLAPSE:
            param_percent->enable();
            param_normaldeviation->enable();
            param_domaindeviation->enable();
            param_datarelativeweight->enable();
            param_ignoredata->enable();
            param_divisions->disable();
            param_divisions_absolute->disable();
            param_smoothSurface->disable();
            param_preserveTopology->disable();
            param_meshSplitting->disable();
            param_boundaryVertexDeletion->disable();
            param_maximumError->disable();
            param_splitAngle->disable();
            param_featureAngle->disable();
            break;

        case QUADRICCLUSTERING:
            param_percent->disable();
            param_normaldeviation->disable();
            param_domaindeviation->disable();
            param_datarelativeweight->disable();
            param_ignoredata->disable();
            param_divisions->enable();
            param_divisions_absolute->enable();
            param_smoothSurface->enable();
            param_preserveTopology->disable();
            param_meshSplitting->disable();
            param_boundaryVertexDeletion->disable();
            param_maximumError->disable();
            param_splitAngle->disable();
            param_featureAngle->disable();
            break;

        case DECIMATEPRO:
            param_percent->enable();
            param_normaldeviation->disable();
            param_domaindeviation->disable();
            param_datarelativeweight->disable();
            param_ignoredata->disable();
            param_divisions->disable();
            param_divisions_absolute->disable();
            param_smoothSurface->enable();
            param_preserveTopology->enable();
            param_meshSplitting->enable();
            param_boundaryVertexDeletion->enable();
            param_maximumError->enable();
            param_splitAngle->enable();
            param_featureAngle->enable();
            break;

        case QUADRICDECIMATION:
            param_percent->enable();
            param_normaldeviation->disable();
            param_domaindeviation->disable();
            param_datarelativeweight->disable();
            param_ignoredata->disable();
            param_divisions->disable();
            param_divisions_absolute->disable();
            param_smoothSurface->disable();
            param_preserveTopology->disable();
            param_meshSplitting->disable();
            param_boundaryVertexDeletion->disable();
            param_maximumError->disable();
            param_splitAngle->disable();
            param_featureAngle->disable();
            break;
        }
    }
}
#endif

int
SimplifySurface::compute(const char *)
{
    const coDistributedObject *inMesh = p_meshIn->getCurrentObject();

    int max_valence = cf_MaxValence;
    if (max_valence < 12)
    {
        sendWarning("setting highest valence to 12");
        max_valence = 12;
    }
    max_cos_2 = (float)cos(2.0 * M_PI / max_valence);
    max_cos_2 *= max_cos_2;

    if (param_normaldeviation->getValue() < 0
        || param_normaldeviation->getValue() > 10.0)
    {
        sendWarning("normal deviation is expected to be between 0 and 10 degrees");
        param_normaldeviation->setValue(3.0);
    }
    normaldeviation_cos = (float)cos(param_normaldeviation->getValue() * M_PI / 180.0);
    domaindeviation_cos = (float)cos(param_domaindeviation->getValue() * M_PI / 180.0);

    float datarelativeweight = param_datarelativeweight->getValue();
    if (datarelativeweight < 0.0)
    {
        sendWarning("Data relative factor is expected to be positive");
        datarelativeweight = -datarelativeweight;
    }

    ignoreData = param_ignoredata->getValue();

    boundary_factor = cf_BoundaryFactor;
    if (boundary_factor < 0.0)
    {
        sendWarning("Boundary factor is expected to be positive");
        boundary_factor = -boundary_factor;
    }

    if (!inMesh || !inMesh->objectOk())
    {
        sendError("No input mesh or not OK");
        return STOP_PIPELINE;
    }
    float percent = param_percent->getValue();
    if ((percent > 100) || (percent <= 0))
    {
        sendWarning("Parameter 'percent' out of range, set to default");
        percent = PERCENT_DEFAULT;
    }

    int n_vert, n_conn, n_poly;
    int *vl_in, *pl_in;
    float *x_in, *y_in, *z_in;
    vector<int> tri_conn_list;
    if (inMesh->isType("POLYGN"))
    {
        coDoPolygons *InMesh = (coDoPolygons *)inMesh;
        n_vert = InMesh->getNumPoints();
        n_conn = InMesh->getNumVertices();
        n_poly = InMesh->getNumPolygons();
        InMesh->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
        vector<int> dummy_codes;
        coFeatureLines::Triangulate(tri_conn_list, dummy_codes,
                                    n_poly, n_conn, pl_in, vl_in,
                                    x_in, y_in, z_in);
    }
    else if (inMesh->isType("TRIANG"))
    {
        coDoTriangleStrips *InMesh = (coDoTriangleStrips *)inMesh;
        n_vert = InMesh->getNumPoints();
        n_conn = InMesh->getNumVertices();
        n_poly = InMesh->getNumStrips();
        InMesh->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
        int strip, triang;
        for (strip = 0; strip < n_poly; ++strip)
        {
            int no_vert;
            int base = pl_in[strip];
            if (strip < n_poly - 1)
            {
                no_vert = pl_in[strip + 1] - base;
            }
            else
            {
                no_vert = n_conn - base;
            }
            for (triang = 0; triang < no_vert - 2; ++triang)
            {
                if (triang % 2)
                {
                    tri_conn_list.push_back(vl_in[base + 1 + triang]);
                    tri_conn_list.push_back(vl_in[base + triang]);
                    tri_conn_list.push_back(vl_in[base + 2 + triang]);
                }
                else
                {
                    tri_conn_list.push_back(vl_in[base + triang]);
                    tri_conn_list.push_back(vl_in[base + 1 + triang]);
                    tri_conn_list.push_back(vl_in[base + 2 + triang]);
                }
            }
        }
    }
    else
    {
        sendError("Wrong type for input mesh");
        return STOP_PIPELINE;
    }

    vector<float> normals_c;
    const coDistributedObject *in_normals = p_normalsIn->getCurrentObject();
    if (in_normals)
    {
        if (in_normals->isType("USTVDT"))
        {
            coDoVec3 *InNormals = (coDoVec3 *)in_normals;
            int no_points = InNormals->getNumPoints();
            if (no_points != n_vert && no_points > 0)
            {
                sendError("Normal data dimension does not coincide with that of the grid");
                return STOP_PIPELINE;
            }
            if (no_points == n_vert)
            {
                normals_c.reserve(3 * n_vert);
                float *x_n;
                float *y_n;
                float *z_n;
                InNormals->getAddresses(&x_n, &y_n, &z_n);
                int i;
                for (i = 0; i < no_points; ++i)
                {
                    float truenorm[3];
                    truenorm[0] = x_n[i];
                    truenorm[1] = y_n[i];
                    truenorm[2] = z_n[i];
                    Normalise(truenorm);
                    normals_c.push_back(truenorm[0]);
                    normals_c.push_back(truenorm[1]);
                    normals_c.push_back(truenorm[2]);
                }
            }
        }
    }

    // assume we have a unique data port...
    const coDistributedObject *in_data = p_dataIn[0]->getCurrentObject();
    vector<float> data_c;
    float data_min[3] = // FIXME
        {
          FLT_MAX, FLT_MAX, FLT_MAX
        };
    float data_max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    float data_diff = 0.0;
    if (in_data && in_data->isType("USTSDT"))
    {
        coDoFloat *InData = (coDoFloat *)in_data;
        int no_points = InData->getNumPoints();
        if (no_points != n_vert && no_points > 0)
        {
            sendError("Data dimension does not coincide with that of the grid");
            return STOP_PIPELINE;
        }
        if (no_points == n_vert)
        {
            data_c.reserve(n_vert);
            float *data;
            InData->getAddress(&data);
            int i;
            for (i = 0; i < n_vert; ++i)
            {
                if (data[i] > data_max[0])
                    data_max[0] = data[i];
                if (data[i] < data_min[0])
                    data_min[0] = data[i];
                data_c.push_back(data[i]);
            }
        }
    }
    else if (in_data && in_data->isType("USTVDT"))
    {
        coDoVec3 *InData = (coDoVec3 *)in_data;
        int no_points = InData->getNumPoints();
        if (no_points != n_vert && no_points > 0)
        {
            sendError("Data dimension does not coincide with that of the grid");
            return STOP_PIPELINE;
        }
        if (no_points == n_vert)
        {
            data_c.reserve(3 * n_vert);
            float *x_data;
            float *y_data;
            float *z_data;
            InData->getAddresses(&x_data, &y_data, &z_data);
            int i;
            for (i = 0; i < n_vert; ++i)
            {
                if (x_data[i] > data_max[0])
                    data_max[0] = x_data[i];
                if (x_data[i] < data_min[0])
                    data_min[0] = x_data[i];
                data_c.push_back(x_data[i]);

                if (y_data[i] > data_max[1])
                    data_max[1] = y_data[i];
                if (y_data[i] < data_min[1])
                    data_min[1] = y_data[i];
                data_c.push_back(y_data[i]);

                if (z_data[i] > data_max[2])
                    data_max[2] = z_data[i];
                if (z_data[i] < data_min[2])
                    data_min[2] = z_data[i];
                data_c.push_back(z_data[i]);
            }
        }
    }
    else if (in_data)
    {
        sendError("Wrong type for input data");
        return STOP_PIPELINE;
    }
    else if (ignoreData)
    {
        // We have no data but are ignoring the data.
        // Unfortunately, the algorithm is different if we have data or not.
        // We create dummy data so we always have the same result.
        data_c.reserve(n_vert);
        for (int i = 0; i < n_vert; ++i)
        {
            data_c.push_back(0.0f);
        }
    }

    if (data_max[0] != -FLT_MAX && data_diff < data_max[0] - data_min[0])
    {
        data_diff = data_max[0] - data_min[0];
    }
    if (data_max[1] != -FLT_MAX && data_diff < data_max[1] - data_min[1])
    {
        data_diff = data_max[1] - data_min[1];
    }
    if (data_max[2] != -FLT_MAX && data_diff < data_max[2] - data_min[2])
    {
        data_diff = data_max[2] - data_min[2];
    }

    //int num_triangles = tri_conn_list.size()/3;
    vector<float> x_c;
    x_c.reserve(n_vert);
    vector<float> y_c;
    y_c.reserve(n_vert);
    vector<float> z_c;
    z_c.reserve(n_vert);
    int vert;
    float x_min = FLT_MAX, x_max = -FLT_MAX,
          y_min = FLT_MAX, y_max = -FLT_MAX,
          z_min = FLT_MAX, z_max = -FLT_MAX;
    for (vert = 0; vert < n_vert; ++vert)
    {
        if (x_in[vert] > x_max)
            x_max = x_in[vert];
        if (x_in[vert] < x_min)
            x_min = x_in[vert];
        x_c.push_back(x_in[vert]);

        if (y_in[vert] > y_max)
            y_max = y_in[vert];
        if (y_in[vert] < y_min)
            y_min = y_in[vert];
        y_c.push_back(y_in[vert]);

        if (z_in[vert] > z_max)
            z_max = z_in[vert];
        if (z_in[vert] < z_min)
            z_min = z_in[vert];
        z_c.push_back(z_in[vert]);
    }

    float grid_diff = 0.0;
    if (x_min != FLT_MAX && grid_diff < x_max - x_min)
    {
        grid_diff = x_max - x_min;
    }
    if (y_min != FLT_MAX && grid_diff < y_max - y_min)
    {
        grid_diff = y_max - y_min;
    }
    if (z_min != FLT_MAX && grid_diff < z_max - z_min)
    {
        grid_diff = z_max - z_min;
    }

    float data_factor = 1.0;
    if (data_diff > 0.0 && grid_diff > 0.0)
    {
        data_factor = datarelativeweight * grid_diff / data_diff;
        unsigned int i;
        for (i = 0; i < data_c.size(); ++i)
        {
            data_c[i] *= data_factor;
        }
    }

    float total_ratio = percent * 0.01f;

#ifdef HAVE_VTK
    if (p_method->getValue() == EDGECOLLAPSE)
    {
#endif
        int stage;
        int num_ini_triangles = (int)(tri_conn_list.size() / 3);
        int ziel_triangles = int(tri_conn_list.size() * total_ratio / 3);
        for (stage = 0; stage < 1; ++stage) // @@@ relict from original version
        {
            float stage_num_ini_triangles = tri_conn_list.size() / 3.0f;
            if (stage_num_ini_triangles <= ziel_triangles)
            {
                break;
            }
            // @@@ relict from original version
            float remaining_reduction = ziel_triangles / stage_num_ini_triangles;
            float stage_ratio = remaining_reduction;
            EdgeCollapseBasis *edgeCollapse = NULL;
            if (cf_Algorithm == 1)
            {
                edgeCollapse = new EdgeCollapse(x_c, y_c, z_c,
                                                tri_conn_list, data_c, normals_c,
                                                VertexContainer::VECTOR,
                                                TriangleContainer::VECTOR,
                                                EdgeContainer::HASHED_SET);
            }
            else
            {
                edgeCollapse = new EdgeCollapseSimple(x_c, y_c, z_c,
                                                      tri_conn_list, data_c, normals_c,
                                                      VertexContainer::VECTOR,
                                                      TriangleContainer::VECTOR,
                                                      EdgeContainer::HASHED_SET);
            }
            // reduction is expected here...
            float num_tri_red = 0;
            string message("Initial number of triangles for ");
            char buf[512];
            sprintf(buf, "is %lu, trying reduction up to %.2f%%...",
                    (unsigned long)tri_conn_list.size() / 3,
                    stage_ratio * 100.0);
            message += buf;
            sendInfo("%s", message.c_str());
            while ((1.0 - (num_tri_red / stage_num_ini_triangles)) > stage_ratio)
            {
                int reduced = edgeCollapse->EdgeContraction(max_valence);
                if (reduced < 0)
                {
                    sendWarning("...could not attain goal at this stage.");
                    break;
                }
                num_tri_red += reduced;
            }
            // get reduced results
            edgeCollapse->LeftEntities(tri_conn_list, x_c, y_c, z_c, data_c, normals_c);

            delete edgeCollapse;
        }
        if (num_ini_triangles > 0)
        {
            sendInfo("Accomplished reduction to %.2f%%.",
                     100.0 * tri_conn_list.size() / (3.0 * num_ini_triangles));
        }

        coDoPolygons *OutTest = new coDoPolygons(p_meshOut->getObjName(),
                                                 (int)x_c.size(), (int)tri_conn_list.size(),
                                                 (int)(tri_conn_list.size() / 3));
        float *x_out, *y_out, *z_out;
        int *vl_out, *pl_out;
        OutTest->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);
        std::copy(x_c.begin(), x_c.end(), x_out);
        std::copy(y_c.begin(), y_c.end(), y_out);
        std::copy(z_c.begin(), z_c.end(), z_out);
        std::copy(tri_conn_list.begin(), tri_conn_list.end(), vl_out);
        unsigned int triangle;
        for (triangle = 0; triangle < tri_conn_list.size() / 3; ++triangle)
        {
            pl_out[triangle] = 3 * triangle;
        }
        OutTest->copyAllAttributes(inMesh);
        p_meshOut->setCurrentObject(OutTest);

        // normals
        if (in_normals)
        {
            coDoVec3 *VData = new coDoVec3(p_normalsOut->getObjName(), (int)(normals_c.size() / 3));
            float *u = NULL, *v = NULL, *w = NULL;
            VData->getAddresses(&u, &v, &w);
            vector<float>::iterator it = normals_c.begin();
            unsigned int i;
            for (i = 0; i < normals_c.size() / 3; ++i)
            {
                u[i] = *it;
                ++it;
                v[i] = *it;
                ++it;
                w[i] = *it;
                ++it;
            }
            VData->copyAllAttributes(in_normals);
            p_normalsOut->setCurrentObject(VData);
        }

        // data
        if ((data_factor != 1.0) && (data_factor != 0.0))
        {
            unsigned int i;
            for (i = 0; i < data_c.size(); ++i)
            {
                data_c[i] /= data_factor;
            }
        }

        if (in_data && in_data->isType("USTSDT"))
        {
            coDoFloat *SData = new coDoFloat(p_dataOut[0]->getObjName(), (int)data_c.size());
            float *data = NULL;
            SData->getAddress(&data);
            std::copy(data_c.begin(), data_c.end(), data);
            SData->copyAllAttributes(in_data);
            p_dataOut[0]->setCurrentObject(SData);
        }
        if (in_data && in_data->isType("USTVDT"))
        {
            coDoVec3 *VData = new coDoVec3(p_dataOut[0]->getObjName(), (int)(data_c.size() / 3));
            float *u = NULL, *v = NULL, *w = NULL;
            VData->getAddresses(&u, &v, &w);
            vector<float>::iterator it = data_c.begin();
            unsigned int i;
            for (i = 0; i < data_c.size() / 3; ++i)
            {
                u[i] = *it;
                ++it;
                v[i] = *it;
                ++it;
                w[i] = *it;
                ++it;
            }
            VData->copyAllAttributes(in_data);
            p_dataOut[0]->setCurrentObject(VData);
        }
#ifdef HAVE_VTK
    }
    else // we use vtk classes here ...
    {
        //cerr << "using QadricClustering polygon reduction" << endl;

        bool scalarData = true;
        float p[3];
        int i, j;

        vtkPolyData *surface = vtkPolyData::New();
        vtkPoints *points = vtkPoints::New();
        vtkCellArray *polys = vtkCellArray::New();

        vtkFloatArray *scalars = NULL;
        vtkFloatArray *vectors = NULL;

        if (in_data && in_data->isType("USTSDT"))
        {
            scalarData = true;
            scalars = vtkFloatArray::New();
            scalars->SetNumberOfComponents(1);
            scalars->SetNumberOfTuples(n_vert);
        }
        else if (in_data && in_data->isType("USTVDT"))
        {
            scalarData = false;
            vectors = vtkFloatArray::New();
            vectors->SetNumberOfComponents(3);
            vectors->SetNumberOfTuples(n_vert);
        }
        else if (in_data)
        {
            sendError("Wrong type for input data");
            return STOP_PIPELINE;
        }

        vtkFloatArray *normals = vtkFloatArray::New();
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(n_vert);

        // add coordinates
        float xmin = FLT_MAX;
        float xmax = -FLT_MAX;
        float ymin = FLT_MAX;
        float ymax = -FLT_MAX;
        float zmin = FLT_MAX;
        float zmax = -FLT_MAX;

        for (i = 0; i < n_vert; i++)
        {
            p[0] = x_in[i];
            p[1] = y_in[i];
            p[2] = z_in[i];
            if (p[0] > xmax)
                xmax = p[0];
            else if (p[0] < xmin)
                xmin = p[0];
            if (p[1] > ymax)
                ymax = p[1];
            else if (p[1] < ymin)
                ymin = p[1];
            if (p[2] > zmax)
                zmax = p[2];
            else if (p[2] < zmin)
                zmin = p[2];

            points->InsertPoint(i, p);
        }

        // add connectivity
        int no_tri = (int)(tri_conn_list.size() / 3);
        for (i = 0; i < no_tri; i++)
        {
            std::vector<vtkIdType> ids(3);
            for (int j = 0; j < 3; ++j)
                ids[j] = tri_conn_list[3 * i + j];
            polys->InsertNextCell(3, &ids[0]);
        }

        // data
        if (in_data && scalarData)
        {
            const coDistributedObject *in_data = p_dataIn[0]->getCurrentObject();
            const coDoFloat *InData = (const coDoFloat *)in_data;
            float *data;
            InData->getAddress(&data);
            for (i = 0; i < n_vert; i++)
                scalars->InsertTuple1(i, data[i]);
        }
        else if (in_data && !scalarData) // vector data
        {
            coDoVec3 *InData = (coDoVec3 *)in_data;
            float *x_data;
            float *y_data;
            float *z_data;
            InData->getAddresses(&x_data, &y_data, &z_data);
            for (i = 0; i < n_vert; i++)
                vectors->InsertTuple3(i, x_data[i], y_data[i], z_data[i]);
        }

        // normals
        if (in_normals)
        {
            float *x_n, *y_n, *z_n;
            const coDoVec3 *InNormals = static_cast<const coDoVec3 *>(in_normals);
            InNormals->getAddresses(&x_n, &y_n, &z_n);

            for (i = 0; i < n_vert; i++)
            {
                normals->InsertTuple3(i, x_n[i], y_n[i], z_n[i]);
            }
        }

        surface->SetPoints(points);
        points->Delete();
        surface->SetPolys(polys);
        polys->Delete();

        if (in_data && scalarData)
        {
            surface->GetPointData()->SetScalars(scalars);
            scalars->Delete();
        }
        else if (in_data && !scalarData) // vector data
        {
            surface->GetPointData()->SetVectors(vectors);
            vectors->Delete();
        }

        vtkPolyDataAlgorithm *decimate = NULL;

        vtkSmoothPolyDataFilter *smoother = NULL;
        vtkCleanPolyData *cleanpolydata;

        if (p_method->getValue() == QUADRICCLUSTERING)
        {
            vtkQuadricClustering *clusterdecimate = vtkQuadricClustering::New();
            decimate = clusterdecimate;

            int n_div[3];
            if (param_divisions_absolute->getValue()) // param_divisions treated as absolute distance
            {
                for (i = 0; i < 3; i++)
                {
                    if (((int)param_divisions->getValue(i)) < 0)
                    {
                        param_divisions->setValue(i, param_divisions->getValue(i) * (-1.));
                    }
                }
                n_div[0] = int((xmax - xmin) / param_divisions->getValue(0));
                n_div[1] = int((ymax - ymin) / param_divisions->getValue(1));
                n_div[2] = int((zmax - zmin) / param_divisions->getValue(2));
                if (n_div[0] < 2)
                {
                    n_div[0] = 2;
                }
                if (n_div[1] < 2)
                {
                    n_div[1] = 2;
                }
                if (n_div[2] < 2)
                {
                    n_div[2] = 2;
                }
            }
            else // param_divisions treated as number of divisions
            {
                for (i = 0; i < 3; i++)
                {
                    if (((int)param_divisions->getValue(i)) < 0)
                    {
                        param_divisions->setValue(i, param_divisions->getValue(i) * (-1.));
                    }
                    if (((int)param_divisions->getValue(i)) < 2.)
                    {
                        param_divisions->setValue(i, 2.);
                    }
                }
                n_div[0] = (int)param_divisions->getValue(0);
                n_div[1] = (int)param_divisions->getValue(1);
                n_div[2] = (int)param_divisions->getValue(2);
            }

            clusterdecimate->SetNumberOfXDivisions(n_div[0]);
            clusterdecimate->SetNumberOfYDivisions(n_div[1]);
            clusterdecimate->SetNumberOfZDivisions(n_div[2]);
            clusterdecimate->SetInputData(surface);

            // smooth surface
            if (param_smoothSurface->getValue())
            {
                smoother = vtkSmoothPolyDataFilter::New();
                smoother->SetInputData(clusterdecimate->GetOutput());
            }
            clusterdecimate->Update();
        }
        else if (p_method->getValue() == DECIMATEPRO)
        {
            cleanpolydata = vtkCleanPolyData::New();
            cleanpolydata->SetTolerance(0.0);
            cleanpolydata->SetInputData(surface);

            vtkDecimatePro *decimatepro = vtkDecimatePro::New();
            decimate = decimatepro;
            decimatepro->SetInputData(cleanpolydata->GetOutput());
            if (param_preserveTopology->getValue())
            {
                decimatepro->PreserveTopologyOn();
            }
            if (param_meshSplitting->getValue())
            {
                decimatepro->SplittingOn();
            }
            if (param_boundaryVertexDeletion->getValue())
            {
                decimatepro->BoundaryVertexDeletionOn();
            }
            // specified as a fraction of the maximum length of the input data bounding box
            decimatepro->SetMaximumError(param_maximumError->getValue());
            decimatepro->SetTargetReduction(1 - param_percent->getValue() / 100.);
            decimatepro->SetSplitAngle(param_splitAngle->getValue());
            decimatepro->SetFeatureAngle(param_featureAngle->getValue());

            // smooth surface
            if (param_smoothSurface->getValue())
            {
                smoother = vtkSmoothPolyDataFilter::New();
                smoother->SetInputData(decimatepro->GetOutput());
            }
            decimatepro->Update();
        }
        else // (p_method->getValue()==QUADRICDECIMATION)
        {
            cleanpolydata = vtkCleanPolyData::New();
            cleanpolydata->SetTolerance(0.0);
            cleanpolydata->SetInputData(surface);

            vtkQuadricDecimation *quaddecimate = vtkQuadricDecimation::New();
            decimate = quaddecimate;
            quaddecimate->SetInputData(cleanpolydata->GetOutput());

            quaddecimate->SetTargetReduction(1 - param_percent->getValue() / 100.);
            quaddecimate->ScalarsAttributeOn(); // should be on by default
            //quaddecimate->SetScalarsWeight(param_datarelativeweight->getValue()); // this doesn't seem to have any effect ...

            // smooth surface
            if (param_smoothSurface->getValue())
            {
                smoother = vtkSmoothPolyDataFilter::New();
                smoother->SetInputData(quaddecimate->GetOutput());
            }
            quaddecimate->Update();
        }

        /*
            // DEBUG display it with vtk renderer
            vtkPolyDataMapper *myMapper = vtkPolyDataMapper::New();
            myMapper->SetInputData( decimate->GetOutput() );
            //myMapper->SetInputData( surface );
            //myMapper->SetInputData( myTriangulator->GetOutput()  );

            vtkActor *myActor = vtkActor::New();
            myActor->SetMapper( myMapper );

            vtkRenderer *ren1= vtkRenderer::New();
            ren1->AddActor( myActor );
            ren1->SetBackground( 0.0, 0.0, 0.0 );

            vtkRenderWindow *renWin = vtkRenderWindow::New();
            renWin->AddRenderer( ren1 );
            renWin->SetSize( 300, 300 );

            vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
            iren->SetRenderWindow(renWin);

            vtkInteractorStyleTrackballCamera *style =
            vtkInteractorStyleTrackballCamera::New();
           iren->SetInteractorStyle(style);

            iren->Initialize();
            iren->Start();
      */

        // transform vtk object back to covise ...
        vtkPolyData *vtkSurface;
        if (smoother)
        {
            vtkSurface = smoother->GetOutput();
        }
        else if (decimate)
        {
            vtkSurface = decimate->GetOutput();
        }
        else
            vtkSurface = NULL;

        coDoPolygons *outputSurface = new coDoPolygons(p_meshOut->getObjName(),
                                                       vtkSurface->GetNumberOfPoints(),
                                                       vtkSurface->GetPolys()->GetNumberOfConnectivityEntries() - vtkSurface->GetNumberOfPolys(),
                                                       vtkSurface->GetNumberOfPolys());

        float *fcoordx, *fcoordy, *fcoordz;
        int *icorner, *ipolygon;
        outputSurface->getAddresses(&(fcoordx), &(fcoordy), &(fcoordz), &(icorner), &(ipolygon));

        // coordinates
        for (i = 0; i < vtkSurface->GetNumberOfPoints(); i++)
        {
            fcoordx[i] = static_cast<float>(vtkSurface->GetPoint(i)[0]);
            fcoordy[i] = static_cast<float>(vtkSurface->GetPoint(i)[1]);
            fcoordz[i] = static_cast<float>(vtkSurface->GetPoint(i)[2]);
        }

        vtkIdType npts = 0, *pts = NULL;
        j = 0;

        // connectivity
        vtkSurface->GetPolys()->InitTraversal();
        for (int k = 0, i = 0; vtkSurface->GetPolys()->GetNextCell(npts, pts); i++)
        {
            if (i == 0)
                ipolygon[0] = 0;
            else
                ipolygon[i] = ipolygon[i - 1] + npts;

            for (j = 0; j < npts; j++, k++)
                icorner[k] = pts[j];

            //outputSurface->setIntRGBA(i,100,100,100,100);
        }

        // data
        if (in_data)
        {
            // unfortunately, vtk doesn't interpolate data on the mesh
            // we need to do that on our own

            //fprintf(stderr,"performing neighbourhood search and data interpolation\n");

            vtkPointLocator *pointLocator = vtkPointLocator::New();
            pointLocator->SetDataSet(surface);
            pointLocator->BuildLocator();
            int numPoints = vtkSurface->GetNumberOfPoints();

            int *nearestPoints = new int[3]; // save three nearest neighbours (from non-reduced surface) for each node on reduced surface
            vtkIdList *result = vtkIdList::New();

            float *outdata;
            if (scalarData)
            {
                outdata = new float[numPoints];
            }
            else
            {
                outdata = new float[3 * numPoints];
            }

            double coord[3][3];
            double origpos[3];

            double vectdata0[3];
            double vectdata1[3];
            double vectdata2[3];

            double k, l, best_k, best_l;

            // number of elements that contain point
	    // TODO: elems_contain_point should be at least as long as the largest node nr in tri_conn_list
	    int *elems_contain_point = (int *)calloc(n_vert + 1, sizeof(int));
            int *elems_at_point_list = (int *)calloc(tri_conn_list.size(), sizeof(int));

            initializeNeighbourhood(elems_contain_point, elems_at_point_list, tri_conn_list, n_vert);

            int *neighbourElems;
            int n_elems = 0;

            //int inside;

            for (i = 0; i < numPoints; i++)
            {
                origpos[0] = fcoordx[i];
                origpos[1] = fcoordy[i];
                origpos[2] = fcoordz[i];

                pointLocator->FindClosestNPoints(3, &origpos[0], result);
                nearestPoints[0] = (int)result->GetId(0);
                nearestPoints[1] = (int)result->GetId(1);
                nearestPoints[2] = (int)result->GetId(2);

                n_elems = getNumNeighbours(elems_contain_point, nearestPoints[0]);
                neighbourElems = new int[n_elems];
                getNeighbourElemsOfVertex(nearestPoints[0], elems_contain_point, elems_at_point_list, neighbourElems, n_elems);

                // choose correct triangle to interpolate in
                int corner[2] = { 0, 0 };
                double *pos = new double[3];
                //inside=0;
                best_k = FLT_MAX;
                best_l = FLT_MAX;

                if (n_elems <= 0)
                {
                    //fprintf(stderr,"error in neighbourhood search. No neighbour elements\n");
                    //return STOP_PIPELINE;
                    if (scalarData)
                    {
                        outdata[i] = surface->GetPointData()->GetScalars()->GetTuple1(nearestPoints[0]);
                    }
                    else // vector data
                    {
                        surface->GetPointData()->GetVectors()->GetTuple(nearestPoints[0], vectdata0);
                        outdata[3 * i + 0] = vectdata0[0];
                        outdata[3 * i + 1] = vectdata0[1];
                        outdata[3 * i + 2] = vectdata0[2];
                    }
                }
                for (j = 0; j < n_elems; j++)
                {
                    // get local coordinates in triangle
                    if (tri_conn_list[3 * neighbourElems[j] + 0] == nearestPoints[0])
                    {
                        corner[0] = tri_conn_list[3 * neighbourElems[j] + 1];
                        corner[1] = tri_conn_list[3 * neighbourElems[j] + 2];
                    }
                    else if (tri_conn_list[3 * neighbourElems[j] + 1] == nearestPoints[0])
                    {
                        corner[0] = tri_conn_list[3 * neighbourElems[j] + 2];
                        corner[1] = tri_conn_list[3 * neighbourElems[j] + 0];
                    }
                    else if (tri_conn_list[3 * neighbourElems[j] + 2] == nearestPoints[0])
                    {
                        corner[0] = tri_conn_list[3 * neighbourElems[j] + 0];
                        corner[1] = tri_conn_list[3 * neighbourElems[j] + 1];
                    }
                    else
                    {
                        //fprintf(stderr,"error in neighbourhood search. Neighbour element does not contain neighbour node\n");
                        //return STOP_PIPELINE;
                        if (scalarData)
                        {
                            outdata[i] = surface->GetPointData()->GetScalars()->GetTuple1(nearestPoints[0]);
                        }
                        else // vector data
                        {
                            surface->GetPointData()->GetVectors()->GetTuple(nearestPoints[0], vectdata0);
                            outdata[3 * i + 0] = vectdata0[0];
                            outdata[3 * i + 1] = vectdata0[1];
                            outdata[3 * i + 2] = vectdata0[2];
                        }
                        break;
                    }
                    // coord are the three triangle corners
                    // the first corner is our nearest point
                    pos = surface->GetPoint(nearestPoints[0]);
                    coord[0][0] = pos[0];
                    coord[0][1] = pos[1];
                    coord[0][2] = pos[2];
                    pos = surface->GetPoint(corner[0]); // first of the two other triangle corners
                    coord[1][0] = pos[0];
                    coord[1][1] = pos[1];
                    coord[1][2] = pos[2];
                    pos = surface->GetPoint(corner[1]); // the remaining triangle corner
                    coord[2][0] = pos[0];
                    coord[2][1] = pos[1];
                    coord[2][2] = pos[2];

                    getLocalCoords(coord, origpos, &k, &l);

                    if ((k >= 0.) && (l >= 0.) && ((k + l) <= 1.))
                    {
                        best_k = k;
                        best_l = l;
                        break;
                    }
                    if (j != 0)
                    {
                        if ((fabs(k) + fabs(l)) < (fabs(best_k) + fabs(best_l)))
                        {
                            best_k = k;
                            best_l = l;
                        }
                    }
                    else
                    {
                        best_k = k;
                        best_l = l;
                    }
                }

                delete[] neighbourElems;

                // data interpolation

                // finally interpolate data
                if (scalarData)
                {
                    outdata[i] = (1 - best_k - best_l) * surface->GetPointData()->GetScalars()->GetTuple1(nearestPoints[0])
                                 + best_k * surface->GetPointData()->GetScalars()->GetTuple1(corner[0])
                                 + best_l * surface->GetPointData()->GetScalars()->GetTuple1(corner[1]);
                }
                else // vector data
                {
                    surface->GetPointData()->GetVectors()->GetTuple(nearestPoints[0], vectdata0);
                    surface->GetPointData()->GetVectors()->GetTuple(corner[0], vectdata1);
                    surface->GetPointData()->GetVectors()->GetTuple(corner[1], vectdata2);

                    outdata[3 * i + 0] = (1 - k - l) * vectdata0[0]
                                         + k * vectdata1[0]
                                         + l * vectdata2[0];
                    outdata[3 * i + 1] = (1 - k - l) * vectdata0[1]
                                         + k * vectdata1[1]
                                         + l * vectdata2[1];
                    outdata[3 * i + 2] = (1 - k - l) * vectdata0[2]
                                         + k * vectdata1[2]
                                         + l * vectdata2[2];
                }
            }

            result->Delete();
            delete[] nearestPoints;

            //fprintf(stderr,"... finished!\n");

            // get data
            if (in_data && in_data->isType("USTSDT"))
            {
                coDoFloat *SData = new coDoFloat(p_dataOut[0]->getObjName(), numPoints);
                float *data = NULL;
                SData->getAddress(&data);
                for (i = 0; i < numPoints; i++)
                {
                    data[i] = outdata[i];
                }
                SData->copyAllAttributes(in_data);
                p_dataOut[0]->setCurrentObject(SData);
            }
            if (in_data && in_data->isType("USTVDT"))
            {
                coDoVec3 *VData = new coDoVec3(p_dataOut[0]->getObjName(), numPoints);
                float *u = NULL, *v = NULL, *w = NULL;
                VData->getAddresses(&u, &v, &w);
                int i;
                for (i = 0; i < numPoints; i++)
                {
                    u[i] = outdata[3 * i + 0];
                    v[i] = outdata[3 * i + 1];
                    w[i] = outdata[3 * i + 2];
                }
                VData->copyAllAttributes(in_data);
                p_dataOut[0]->setCurrentObject(VData);
            }
        }

        //vtkDataArray *outvectdata = (vtkFloatArray*) vtkSurface->GetPointData()->GetVectors();

        double reduction = 100. / surface->GetNumberOfCells() * i;
        sendInfo("Accomplished reduction to %.2f%%.", reduction);

        outputSurface->copyAllAttributes(inMesh);
        p_meshOut->setCurrentObject(outputSurface);
    }
#endif
    return CONTINUE_PIPELINE;
}

#ifdef HAVE_VTK
void SimplifySurface::initializeNeighbourhood(int *elems_contain_point, int *elems_at_point_list, vector<int> &tri_conn_list, int n_vert)
{
    // we need to have an information in which elements each vert takes part
    int point;
    int i, j;

    for (i = 0; i < tri_conn_list.size(); i++)
    {
	if (tri_conn_list[i] > n_vert)
	{
		sendError("tri_conn_list entry exceeds n_vert");
	}
        elems_contain_point[tri_conn_list[i]]++;
    }

    for (i = n_vert - 1; i > 0; i--)
    {
        elems_contain_point[i] = elems_contain_point[i - 1];
    }
    elems_contain_point[0] = 0;
    for (i = 1; i <= n_vert + 1; i++)
    {
        // elems_contain_point[i] gives us the index where we can find the elements containing point i in array elems_with_point_list
        elems_contain_point[i] += elems_contain_point[i - 1];
    }

    int *point_counter = (int *)calloc(n_vert, sizeof(int));

    for (i = 0; i < tri_conn_list.size() / 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            point = tri_conn_list[3 * i + j];
            elems_at_point_list[elems_contain_point[point] + point_counter[point]] = i;
            point_counter[point]++;
        }
    }
    free(point_counter);

    return;
}

void SimplifySurface::getNeighbourElemsOfVertex(int vertex, int *elems_contain_point, int *elems_at_point_list, int *neighbourElems, int n_elems)
{
    int j;

    for (j = 0; j < n_elems; j++)
    {
        neighbourElems[j] = elems_at_point_list[elems_contain_point[vertex] + j];
    }

    return;
}

int SimplifySurface::getNumNeighbours(int *elems_contain_point, int vertex)
{
    return (elems_contain_point[vertex + 1] - elems_contain_point[vertex]);
}

void SimplifySurface::getLocalCoords(double coord[][3], double origpos[3], double *k, double *l)
{

    // get local coordinates in triangle
    // coord[0] are the coords of nearest point
    // coord[1] and coord[2] are the coords of the other two remaining triangle vertices
    double r[3], s[3], n[3], x[3], d, length, dx;

    // normal vector of local triangle
    r[0] = coord[1][0] - coord[0][0];
    r[1] = coord[1][1] - coord[0][1];
    r[2] = coord[1][2] - coord[0][2];
    s[0] = coord[2][0] - coord[0][0];
    s[1] = coord[2][1] - coord[0][1];
    s[2] = coord[2][2] - coord[0][2];
    n[0] = r[1] * s[2] - r[2] * s[1];
    n[1] = r[2] * s[0] - r[0] * s[2];
    n[2] = r[0] * s[1] - r[1] * s[0];

    if ((fabs(n[0]) < 0.01) && (fabs(n[1]) < 0.01) && (fabs(n[2]) < 0.01))
    {
        // seems that we have two points very close to each other
        // I think we can take over the data from nearest neighbour without risk
        *k = 0.0;
        *l = 0.0;
    }
    else
    {
        length = pow((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 0.5);
        length = 1. / length;
        n[0] *= length;
        n[1] *= length;
        n[2] *= length;
        d = n[0] * coord[0][0] + n[1] * coord[0][1] + n[2] * coord[0][2];

        // project origpos (interpolation point) into triangle plane
        dx = n[0] * origpos[0] + n[1] * origpos[1] + n[2] * origpos[2] - d;
        x[0] = origpos[0] - dx * n[0];
        x[1] = origpos[1] - dx * n[1];
        x[2] = origpos[2] - dx * n[2];
        origpos[0] = x[0];
        origpos[1] = x[1];
        origpos[2] = x[2];

        // get local coordinates
        // we are in yz-plane!
        if (((fabs(n[1]) < 0.01)) && ((fabs(n[2]) < 0.01)))
        {
            *k = (origpos[1] * s[2] - origpos[2] * s[1] - coord[0][1] * s[2] + coord[0][2] * s[1]) / (r[1] * s[2] - r[2] * s[1]);
            *l = (origpos[1] * r[2] - origpos[2] * r[1] - coord[0][1] * r[2] + coord[0][2] * r[1]) / (s[1] * r[2] - s[2] * r[1]);
        }
        // we are in xz-plane!
        else if (((fabs(n[0]) < 0.01)) && ((fabs(n[2]) < 0.01)))
        {
            *k = (origpos[2] * s[0] - origpos[0] * s[2] - coord[0][2] * s[0] + coord[0][0] * s[2]) / (r[2] * s[0] - r[0] * s[2]);
            *l = (origpos[2] * r[0] - origpos[0] * r[2] - coord[0][2] * r[0] + coord[0][0] * r[2]) / (s[2] * r[0] - s[0] * r[2]);
        }
        else // xy-plane or anywhere else
        {
            *k = (origpos[0] * s[1] - origpos[1] * s[0] - coord[0][0] * s[1] + coord[0][1] * s[0]) / (r[0] * s[1] - r[1] * s[0]);
            *l = (origpos[0] * r[1] - origpos[1] * r[0] - coord[0][0] * r[1] + coord[0][1] * r[0]) / (s[0] * r[1] - s[1] * r[0]);
        }
    }

    return;
}
#endif

SimplifySurface::~SimplifySurface()
{
}

MODULE_MAIN(Filter, SimplifySurface)
