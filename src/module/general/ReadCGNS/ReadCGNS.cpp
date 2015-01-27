/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                   (C)2010 Stellba Hydro GmbH & Co. KG  **
**                                                                        **
** Description: READ CGNS CFD format                                      **                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include "ReadCGNS.h"
#ifdef WIN32
#define rindex(s, c) strrchr(s, c)
#endif

// compatibilty with ubuntu system CGNS
#ifndef CGNS_ENUMT
#define CGNS_ENUMT(e) e
#endif
#ifndef CGNS_ENUMV
#define CGNS_ENUMV(e) e

typedef int cgsize_t;
#endif

// for sorting
typedef std::pair<int, float> int_float_pair_;
typedef std::vector<int_float_pair_> int_float_vec_;
bool comparator(const int_float_pair_ &l, const int_float_pair_ &r)
{
    return l.second < r.second;
}

ReadCGNS::ReadCGNS(int argc, char *argv[])
    : coModule(argc, argv, "CGNS Reader")
{

    // the output ports
    p_mesh = addOutputPort("GridOut0", "UnstructuredGrid", "the grid");
    p_scalar_3D = addOutputPort("DataOut0", "Float", "scalar data (3D)");
    p_vector_3D = addOutputPort("DataOut1", "Vec3", "vector data (3D)");

    p_boundaries = addOutputPort("GridOut1", "Polygons", "boundary polygons");
    p_scalar_2D = addOutputPort("DataOut2", "Float", "scalar data at boundaries");
    p_vector_2D = addOutputPort("DataOut3", "Vec3", "vector data at boundaries");

    // parameters

    // input file
    char filePath[200];
    sprintf(filePath, "%s", getenv("HOME"));
    p_cgnsMeshFile = addFileBrowserParam("cgnsMeshFile", "mesh file");
    p_cgnsMeshFile->setValue(filePath, "*.cgns");

    p_meshFileBaseNr = addInt32Param("meshFileBaseNr", "number of mesh file CGNS base to read (1..n)");
    p_meshFileBaseNr->setValue(1);

    p_cgnsDataFile = addFileBrowserParam("cgnsDataFile", "mesh file");
    p_cgnsDataFile->setValue(filePath, "*.cgns");

    p_dataFileBaseNr = addInt32Param("dataFileBaseNr", "number of data file CGNS base to read (1..n)");
    p_dataFileBaseNr->setValue(1);

    p_boundary = addStringParam("BoundarySelection", "selection of boundaries, can be numbers or WALL, INLET, OUTLET, ...");
    p_boundary->setValue("WALL");

    const char *initScalar[] = { "No Scalar data" };
    const char *initVector[] = { "No Vector data" };

    p_scalar = addChoiceParam("scalar_data", "Scalar data");
    p_scalar->setValue(1, initScalar, 0);

    p_vector = addChoiceParam("vector_data", "Vector data");
    p_vector->setValue(1, initVector, 0);

    p_boundScalar = addChoiceParam("boundary_scalar_data", "Boundary Scalar data");
    p_boundScalar->setValue(1, initScalar, 0);

    p_boundVector = addChoiceParam("boundary_vector_data", "Boundary Vector data");
    p_boundVector->setValue(1, initVector, 0);
}

ReadCGNS::~ReadCGNS()
{
}

void ReadCGNS::param(const char *paramName, bool inMapLoading)
{
    int readVariables = 0;

    if (inMapLoading)
    {
        if (0 == strcmp(p_dataFileBaseNr->getName(), paramName))
        {
            readVariables = 1;
        }
    }
    else
    {
        if ((0 == strcmp(p_dataFileBaseNr->getName(), paramName)) || (0 == strcmp(p_cgnsDataFile->getName(), paramName)))
        {
            readVariables = 1;
        }
    }

    if (readVariables)
    {
        // check file status
        int ier;

        // open CGNS mesh file (read-only)
        int dataFile;
        ier = 0;
        ier = cg_open(p_cgnsDataFile->getValue(), CG_MODE_READ, &dataFile);
        if (ier)
        {
            const char *error_message = cg_get_error();
            fprintf(stderr, "%s\n", error_message);
            return;
        }

        // read variable names (from first base, first zone, first solution)
        // count number of scalar and number of vector params
        int nDataFields = 0;
        int dataBase = p_dataFileBaseNr->getValue();
        cg_nfields(dataFile, dataBase, 1, 1, &nDataFields);
        fprintf(stderr, "nDataFields = %d\n", nDataFields);

        // count scalar and vector data fields
        int nScalars = 0;
        int nVectors = 0;
        char fieldname[255];
        CGNS_ENUMT(DataType_t) datatype;

        FieldName.clear();
        if (ScalChoiceVal)
        {
            delete[] ScalChoiceVal;
        }
        if (VectChoiceVal)
        {
            delete[] VectChoiceVal;
        }
        if (ScalIndex)
        {
            delete[] ScalIndex;
        }
        if (VectIndex)
        {
            delete[] VectIndex;
        }

        // count them ...
        for (int i = 0; i < nDataFields; i++)
        {
            /*
	    cg_field_info(int fn, int B, int Z, int S, int F,
	    DataType_t *datatype, char *fieldname); 	r - m
	    fn	   	CGNS file index number.
	    B		Base index number, where 1 ≤ B ≤ nbases.
	    Z		Zone index number, where 1 ≤ Z ≤ nzones.
	    S		Flow solution index number, where 1 ≤ S ≤ nsols.
	    F		Solution array index number, where 1 ≤ F ≤ nfields.
	    nfields	Number of data arrays in flow solution S.
	    datatype	Data type in which the solution array is written.
	    fieldname	Name of the solution array.
*/
            cg_field_info(dataFile, dataBase, 1, 1, i + 1, &datatype, fieldname);
            //fprintf(stderr,"dataBase = %d¸\n", dataBase);
            fprintf(stderr, "data field %d: %s\n", i + 1, fieldname);
            FieldName.push_back(strdup(fieldname));
            if (strstr(fieldname, "Velocity"))
            {
                nVectors++;
            }
            else
            {
                nScalars++;
            }
        }
        if ((nVectors % 3) != 0)
        {
            fprintf(stderr, "we can only handle vectors with 3 components. please check your variables!\n");
            return;
        }
        nVectors /= 3;

        fprintf(stderr, "nScalars = %d\n", nScalars);
        fprintf(stderr, "nVectors = %d\n", nVectors);

        ScalChoiceVal = new char *[nScalars + 2];
        VectChoiceVal = new char *[nVectors + 2];
        ScalIndex = new int[nScalars]; // stores field number for each variable
        VectIndex = new int[nVectors]; // stores field number for each variable

        ScalChoiceVal[0] = (char *)"none";
        VectChoiceVal[0] = (char *)"none";

        // ... and store them
        nScalars = 0;
        nVectors = 0;
        for (int i = 0; i < nDataFields; i++)
        {
            cg_field_info(dataFile, dataBase, 1, 1, i + 1, &datatype, fieldname);
            if (strstr(fieldname, "Velocity"))
            {
                if ((nVectors % 3) == 0)
                {
                    if (strrchr(fieldname, '_'))
                    {
                        // VelocityXYZ_x -> VelocityXYZ
                        char *delim;
                        delim = rindex(fieldname, '_');
                        fieldname[rindex(fieldname, '_') - fieldname] = '\0';
                    }
                    else if (strrchr(fieldname, '.'))
                    {
                        // VelocityXYZ.vx -> VelocityXYZ
                        char *delim;
                        delim = rindex(fieldname, '.');
                        fieldname[rindex(fieldname, '.') - fieldname] = '\0';
                    }
                    else
                    {
                        fieldname[strlen(fieldname) - 1] = '\0';
                    }
                    VectChoiceVal[1 + nVectors / 3] = strdup(fieldname);
                    VectIndex[nVectors / 3] = i;
                    fprintf(stderr, "nVectors = %d, VectIndex[%d] = %d\n", nVectors, nVectors / 3, i);
                    fprintf(stderr, "FieldName[%d] = %s\n", i, FieldName[i]);
                }
                nVectors++;
            }
            else
            {
                ScalChoiceVal[1 + nScalars] = strdup(fieldname);
                ScalIndex[nScalars] = i;
                nScalars++;
            }
        }
        nVectors /= 3;

        p_scalar->updateValue(nScalars + 1, ScalChoiceVal, p_scalar->getValue());
        p_vector->updateValue(nVectors + 1, VectChoiceVal, p_vector->getValue());
    }
}

int ReadCGNS::compute(const char *)
{

    //param(p_cgnsDataFile->getName(), false);

    int meshFile;
    cgsize_t size[9];

    char zonename[255];
    CGNS_ENUMT(ZoneType_t) zonetype;

    float version;
    int ier;

    // open CGNS mesh file (read-only)
    ier = 0;
    ier = cg_open(p_cgnsMeshFile->getValue(), CG_MODE_READ, &meshFile);
    if (ier)
    {
        const char *error_message = cg_get_error();
        fprintf(stderr, "%s\n", error_message);
        return STOP_PIPELINE;
    }

    fprintf(stderr, "\nReadCGNS: reading mesh file %s\n", p_cgnsMeshFile->getValue());

    fprintf(stderr, "\nchecking file ...\n");

    cg_version(meshFile, &version);
    fprintf(stderr, "\tfile version is %f\n", version);

    // get number of bases
    int nbases;
    cg_nbases(meshFile, &nbases);

    fprintf(stderr, "\tfile contains %d bases\n", nbases);

    int *nNodes_total = new int[nbases];
    int *nElem_total = new int[nbases];

    // cgns mesh file and data file do not need to store zones in the same order
    // so we need to remember the zone names
    std::vector<char *> meshZoneNames;

    for (int i = 0; i < nbases; i++)
    {
        // base name
        char basename[200];
        int cell_dim, phys_dim;
        cg_base_read(meshFile, i + 1, basename, &cell_dim, &phys_dim);

        // get number of zones
        int nZones;
        cg_nzones(meshFile, i + 1, &nZones);

        fprintf(stderr, "\tBase %d, name: %s, contains %d zones and is %dD\n", i + 1, basename, nZones, cell_dim);

        nNodes_total[i] = 0;
        nElem_total[i] = 0;

        for (int j = 0; j < nZones; j++)
        {
            cg_zone_type(meshFile, i + 1, j + 1, &zonetype);
            switch (zonetype)
            {
            case CGNS_ENUMV(Structured):
            {
                fprintf(stderr, "\t\tzone %d: Structured, ", j + 1);
                cg_zone_read(meshFile, i + 1, j + 1, zonename, size);
                fprintf(stderr, "name %s\n", zonename);
                meshZoneNames.push_back(strdup(zonename));
                fprintf(stderr, "\t\t\tnodes (i, j, k): %ld %ld %ld\n", (long)size[0], (long)size[1], (long)size[2]);
                nNodes_total[i] += size[0] * size[1] * size[2];
                nElem_total[i] += (size[0] - 1) * (size[1] - 1) * (size[2] - 1);
                break;
            }
            case CGNS_ENUMV(Unstructured):
            {
                fprintf(stderr, "\t\tzone %d: Unstructured, ", j + 1);
                cg_zone_read(meshFile, i + 1, j + 1, zonename, size);
                meshZoneNames.push_back(strdup(zonename));
                fprintf(stderr, "name %s, %ld %ld %ld\n", zonename, (long)size[0], (long)size[1], (long)size[2]);
                fprintf(stderr, "\t\t\t%ld nodes, %ld cells\n", (long)size[0], (long)size[1]);
                nNodes_total[i] += size[0];
                nElem_total[i] += size[1];
                break;
            }
            default:
            {
                sendError("Unrecognized zone type - only Structured and Unstructured are accepted\n");
                break;
            }
            }
        }
        fprintf(stderr, "\t\tnNodes     total in Base: %d\n", nNodes_total[i]);
        fprintf(stderr, "\t\tnElementes total in Base: %d\n", nElem_total[i]);
    }

    // read Base p_meshFileBaseNr->getValue()
    // code reads a block-structured mesh
    // TODO: read unstrucured zones as well - I don't need it ;-)
    int meshBase = p_meshFileBaseNr->getValue();

    fprintf(stderr, "\nreading base %d ...\n\n", meshBase);

    // open data file ...
    int dataFile;
    int readData = 0;
    if (strcmp(p_cgnsDataFile->getValue(), "0"))
    {
        ier = cg_open(p_cgnsDataFile->getValue(), CG_MODE_READ, &dataFile);
        if (ier)
        {
            const char *error_message = cg_get_error();
            fprintf(stderr, "%s\n", error_message);
            fprintf(stderr, "not reading data file\n");
        }
        else
        {
            fprintf(stderr, "reading data file %s\n", p_cgnsDataFile->getValue());
            readData = 1;
        }
    }

    // TODO: check whether data file and mesh file match (same number of nodes in each block)
    int dataBase = p_dataFileBaseNr->getValue();

    int *zoneMapping; // contains in which order to read the zone numbers of the data file (to fit to mesh file)
    int nZones;
    cg_nzones(meshFile, meshBase, &nZones);
    zoneMapping = new int[nZones];

    int nZonesData;
    if (readData)
    {
        cg_nzones(dataFile, meshBase, &nZonesData);
        if (nZones != nZonesData)
        {
            fprintf(stderr, "data file and mesh file do not match (different number of zones).\n");
            return STOP_PIPELINE;
        }

        createMeshDataZoneMapping(meshFile, dataFile, meshBase, dataBase, &meshZoneNames, zoneMapping);
    }

    /*
ier = cg_nsols(int fn, int B, int Z, int *nsols);
ier = cg_sol_info(int fn, int B, int Z, int S, char *solname,
      GridLocation_t *location);
ier = cg_nfields(int fn, int B, int Z, int S, int *nfields);
ier = cg_field_info(int fn, int B, int Z, int S, int F,
      DataType_t *datatype, char *fieldname);
ier = cg_field_read(int fn, int B, int Z, int S, char *fieldname,
      DataType_t datatype, int *range_min, int *range_max,
      void *solution_array);
*/

    int nodeCounter = 0;
    int elemCounter = 0;
    int *startNodeNr = new int[nZones]; // zone start node numbers
    float *xCoords; // global coordinate arrays
    float *yCoords;
    float *zCoords;
    float *valScalar;
    float *valVectorX;
    float *valVectorY;
    float *valVectorZ;

    coDoUnstructuredGrid *grid;
    int *elemList, *connList, *typeList;

    // this is only valid for a blockstructured mesh!
    grid = new coDoUnstructuredGrid(p_mesh->getObjName(), nElem_total[0], 8 * nElem_total[0], nNodes_total[0], 1);
    grid->getAddresses(&elemList, &connList, &xCoords, &yCoords, &zCoords);
    grid->getTypeList(&typeList);

    int nCoords;
    int imax, jmax, kmax;

    char GridCoordName[200];

    coDoFloat *scalar;
    coDoVec3 *vector;

    if (readData)
    {
        if (p_scalar->getValue())
        {
            scalar = new coDoFloat(p_scalar_3D->getObjName(), nNodes_total[0]);
            scalar->getAddress(&valScalar);
        }
        if (p_vector->getValue())
        {
            // read vector variable
            vector = new coDoVec3(p_vector_3D->getObjName(), nNodes_total[0]);
            vector->getAddresses(&valVectorX, &valVectorY, &valVectorZ);
        }
    }

    for (int zoneNr = 0; zoneNr < nZones; zoneNr++)
    {
        int nGrids;
        cg_ngrids(meshFile, meshBase, zoneNr + 1, &nGrids);
        cg_zone_type(meshFile, meshBase, zoneNr + 1, &zonetype);
        cg_zone_read(meshFile, meshBase, zoneNr + 1, zonename, size);
        imax = size[0];
        jmax = size[1];
        kmax = size[2];

        int nSolutions;
        int nDataFields;
        if (readData)
        {
            cg_nsols(dataFile, dataBase, zoneNr + 1, &nSolutions);
            // we will read just the first solution in each zone
            cg_nfields(dataFile, dataBase, zoneNr + 1, 1, &nDataFields);
        }

        if (readData)
        {
            if (nGrids == 0)
            {
                fprintf(stderr, "zone %d (%s) contains no grid! Did you select a data file? Please check. Exiting.\n", zoneNr + 1, zonename);
                return STOP_PIPELINE;
            }
            else if (nGrids == 1)
            {
                fprintf(stderr, "zone %d (%s) contains %d grid, %d solution(s) and %d data fields, ", zoneNr + 1, zonename, nGrids, nSolutions, nDataFields);
            }
            else
            {
                fprintf(stderr, "zone %d (%s) contains %d grids, %d solution(s) and %d data fields, ", zoneNr + 1, zonename, nGrids, nSolutions, nDataFields);
            }
        }
        else
        {
            if (nGrids == 0)
            {
                fprintf(stderr, "zone %d (%s) contains no grid! Did you select a data file? Please check. Exiting.\n", zoneNr + 1, zonename);
                return STOP_PIPELINE;
            }
            else if (nGrids == 1)
            {
                fprintf(stderr, "zone %d (%s) contains %d grid, ", zoneNr + 1, zonename, nGrids);
            }
            else
            {
                fprintf(stderr, "zone %d (%s) contains %d grids, ", zoneNr + 1, zonename, nGrids);
            }
        }

        for (int i = 0; i < nGrids; i++)
        {
            cg_grid_read(meshFile, meshBase, zoneNr + 1, i + 1, GridCoordName);
            if ((i == 0) && (strcmp(GridCoordName, "GridCoordinates")))
            {
                fprintf(stderr, "error: first grid should be named \"GridCoordinates\"\n");
            }
            // fprintf(stderr,"\tGridCoordName: %s\n",GridCoordName);
        }

        cg_ncoords(meshFile, meshBase, zoneNr + 1, &nCoords);
        fprintf(stderr, "dimension: %dD, ", nCoords);

        cgsize_t size_min[3] = { 1, 1, 1 };
        cgsize_t size_max[3] = { size[0], size[1], size[2] };
        int nNodes = size[0] * size[1] * size[2];

        float *x = new float[nNodes];
        float *y = new float[nNodes];
        float *z = new float[nNodes];

        // TODO: check for cartesian coordinate system
        // might also be a cylindrical CS or something else
        ier = cg_coord_read(meshFile, meshBase, zoneNr + 1, "CoordinateX", CGNS_ENUMV(RealSingle), size_min, size_max, x);
        if (ier)
        {
            const char *error_message = cg_get_error();
            fprintf(stderr, "\n%s\n", error_message);
            return STOP_PIPELINE;
        }

        cg_coord_read(meshFile, meshBase, zoneNr + 1, "CoordinateY", CGNS_ENUMV(RealSingle), size_min, size_max, y);
        cg_coord_read(meshFile, meshBase, zoneNr + 1, "CoordinateZ", CGNS_ENUMV(RealSingle), size_min, size_max, z);

        fprintf(stderr, "nNodes = %d\n", nNodes);

        // remember start node nr. for each zone
        startNodeNr[zoneNr] = nodeCounter;

        float *scal;
        float *vectX;
        float *vectY;
        float *vectZ;
        /*
ier = cg_field_read(int fn, int B, int Z, int S, char *fieldname,
      DataType_t datatype, int *range_min, int *range_max,
      void *solution_array);
*/
        cgsize_t data_size_max[3];
        cgsize_t data_size_min[3];
        data_size_min[0] = data_size_min[1] = data_size_min[2] = 1;

        if (readData)
        {
            cg_zone_read(dataFile, dataBase, zoneNr + 1, zonename, data_size_max);

            // cg_zone_read reads 6 values (for nodes and elements max), so reset data_size_min to 1
            data_size_min[0] = data_size_min[1] = data_size_min[2] = 1;

            //fprintf(stderr,"data file zone %d: %s\n",zoneNr+1,zonename);

            if (p_scalar->getValue())
            {
                // read scalar variable
                scal = new float[data_size_max[0] * data_size_max[1] * data_size_max[2]];
                fprintf(stderr, "reading scalar variable %s\n", FieldName[ScalIndex[p_scalar->getValue() - 1]]);
                //fprintf(stderr,"n data values = %d\n",data_size_max[0]*data_size_max[1]*data_size_max[2]);
                ier = cg_field_read(dataFile, dataBase, zoneNr + 1, 1, FieldName[ScalIndex[p_scalar->getValue() - 1]], CGNS_ENUMV(RealSingle), data_size_min, data_size_max, scal);
                /*
float max = -FLT_MAX;
float min = FLT_MAX;
for (int i=0; i<data_size_max[0]*data_size_max[1]*data_size_max[2]; i++)
{
    if (scal[i]>max)
	max = scal[i];
    if (scal[i]<min)
	min = scal[i];
}
fprintf(stderr,"min = %8.4lf, max = %8.4lf\n", min, max);
*/
            }
            if (p_vector->getValue())
            {
                // read vector variable
                vectX = new float[data_size_max[0] * data_size_max[1] * data_size_max[2]];
                vectY = new float[data_size_max[0] * data_size_max[1] * data_size_max[2]];
                vectZ = new float[data_size_max[0] * data_size_max[1] * data_size_max[2]];
                fprintf(stderr, "reading vector variables %s %s %s\n", FieldName[VectIndex[p_vector->getValue() - 1]], FieldName[VectIndex[p_vector->getValue() - 1] + 1], FieldName[VectIndex[p_vector->getValue() - 1] + 2]);
                //fprintf(stderr,"n data values = %d\n",data_size_max[0]*data_size_max[1]*data_size_max[2]);
                cg_field_read(dataFile, dataBase, zoneNr + 1, 1, FieldName[VectIndex[p_vector->getValue() - 1] + 0], CGNS_ENUMV(RealSingle), data_size_min, data_size_max, vectX);
                cg_field_read(dataFile, dataBase, zoneNr + 1, 1, FieldName[VectIndex[p_vector->getValue() - 1] + 1], CGNS_ENUMV(RealSingle), data_size_min, data_size_max, vectY);
                cg_field_read(dataFile, dataBase, zoneNr + 1, 1, FieldName[VectIndex[p_vector->getValue() - 1] + 2], CGNS_ENUMV(RealSingle), data_size_min, data_size_max, vectZ);
                /*
float max = -FLT_MAX;
float min = FLT_MAX;
for (int i=0; i<data_size_max[0]*data_size_max[1]*data_size_max[2]; i++)
{
    if (vectX[i]>max)
	max = vectX[i];
    if (vectX[i]<min)
	min = vectX[i];
}
fprintf(stderr,"min = %8.4lf, max = %8.4lf\n", min, max);
*/
            }
        }

        /*
	ier = cg_field_read(int fn, int B, int Z, int S, char *fieldname,
	DataType_t datatype, int *range_min, int *range_max, void *solution_array);
*/

        // use global vector for coordinates
        for (int i = 0; i < nNodes; i++)
        {
            xCoords[nodeCounter] = x[i];
            yCoords[nodeCounter] = y[i];
            zCoords[nodeCounter] = z[i];

            if (readData)
            {
                if (p_scalar->getValue())
                {
                    valScalar[nodeCounter] = scal[i];
                }
                if (p_vector->getValue())
                {
                    valVectorX[nodeCounter] = vectX[i];
                    valVectorY[nodeCounter] = vectY[i];
                    valVectorZ[nodeCounter] = vectZ[i];
                }
            }

            nodeCounter++;
            /*
	    if (i<10)
		fprintf(stderr,"x=%f, y=%f, z=%f\n",x[i],y[i],z[i]);
*/
        }

        // free memory
        delete[] x;
        delete[] y;
        delete[] z;

        if (readData)
        {
            if (p_scalar->getValue())
            {
                delete[] scal;
            }
            if (p_vector->getValue())
            {
                delete[] vectX;
                delete[] vectY;
                delete[] vectZ;
            }
        }

        // read connectivity list
        if (zonetype == CGNS_ENUMV(Unstructured))
        {
            // code not finished yet ... we just have blockstructured hex meshes here at Stellba ...
            // feel free to implement the rest
            int nSections; // number of element sections
            char sectionname[200];
            CGNS_ENUMT(ElementType_t) eType;
            cgsize_t istart, iend;
            int nbndry;
            int iparent_flag;
            cg_nsections(meshFile, meshBase, zoneNr, &nSections);
            printf("\tnumber of sections: %i\n", nSections);

            for (int sectNr = 1; sectNr <= nSections; sectNr++)
            {
                cg_section_read(meshFile, meshBase, zoneNr, sectNr, sectionname,
                                &eType, &istart, &iend, &nbndry, &iparent_flag);
                printf("\tReading section data...\n");
                printf("\t\tsection name=%s\n", sectionname);
                printf("\t\tsection type=%s\n", ElementTypeName[eType]);
                printf("\t\tistart,iend=%li, %li\n", (long)istart, (long)iend);
                /*
		if (itype == HEXA_8)
		{
		    cg_elements_read(fn,1,zoneNr+1,sectNr+1,ielem[0], \
			     &iparentdata);
		}
    */
            }
        }
        else if (zonetype == CGNS_ENUMV(Structured))
        {
            // connectivity is implicit, generating connectivity list
            // we have only 8-node hex elements
            for (int k = 0; k < (kmax - 1); k++)
            {
                for (int j = 0; j < (jmax - 1); j++)
                {
                    for (int i = 0; i < (imax - 1); i++)
                    {
                        elemList[elemCounter] = 8 * elemCounter;
                        typeList[elemCounter] = TYPE_HEXAGON;
                        connList[8 * elemCounter + 0] = startNodeNr[zoneNr] + k * jmax * imax + j * imax + i;
                        connList[8 * elemCounter + 1] = startNodeNr[zoneNr] + k * jmax * imax + j * imax + i + 1;
                        connList[8 * elemCounter + 2] = startNodeNr[zoneNr] + k * jmax * imax + (j + 1) * imax + i + 1;
                        connList[8 * elemCounter + 3] = startNodeNr[zoneNr] + k * jmax * imax + (j + 1) * imax + i;
                        connList[8 * elemCounter + 4] = startNodeNr[zoneNr] + (k + 1) * jmax * imax + j * imax + i;
                        connList[8 * elemCounter + 5] = startNodeNr[zoneNr] + (k + 1) * jmax * imax + j * imax + i + 1;
                        connList[8 * elemCounter + 6] = startNodeNr[zoneNr] + (k + 1) * jmax * imax + (j + 1) * imax + i + 1;
                        connList[8 * elemCounter + 7] = startNodeNr[zoneNr] + (k + 1) * jmax * imax + (j + 1) * imax + i;
                        elemCounter++;
                    }
                }
            }
        }
    }

    cg_close(meshFile);
    cg_close(dataFile);

    //fprintf(stderr,"nNodes total before merging = %d\n",nodeCounter);
    if (nodeCounter != nNodes_total[0])
    {
        fprintf(stderr, "\n\nsomething has gone wrong with number of nodes!\n");
        return STOP_PIPELINE;
    }
    //fprintf(stderr,"nElem total = %d\n",elemCounter);
    if (elemCounter != nElem_total[0])
    {
        fprintf(stderr, "\n\nsomething has gone wrong with number of elements!\n");
        fprintf(stderr, "elemCounter=%d, nElem_total[0]=%d\n", elemCounter, nElem_total[0]);
        return STOP_PIPELINE;
    }

    // create a mapping table for local to global node numbers
    // we need to merge nodes at structured block margins

    // first step: find a reasonable mergeTolerance
    //    default: 1% of smallest element edge
    //    find smalles element edge
    float smallestEdge = findSmallestEdge(xCoords, yCoords, zCoords, nElem_total[0], elemList, connList, typeList);
    fprintf(stderr, "smallest edge length: %f\n", sqrt(smallestEdge));

    int *mapLocalToGlobal = new int[nNodes_total[0]];
    int mergedNodes = 0;
    mergeNodes(xCoords, yCoords, zCoords,
               nNodes_total[0], 0.01 * smallestEdge,
               mapLocalToGlobal, mergedNodes);
    fprintf(stderr, "\nmerged %d nodes at block margins.\n", mergedNodes);

    int *removedNode = new int[nNodes_total[0]];
    memset(removedNode, 0, nNodes_total[0] * sizeof(int));

    for (int i = 0; i < nNodes_total[0]; i++)
    {
        if (mapLocalToGlobal[i] == -1)
        {
            mapLocalToGlobal[i] = i;

            //fprintf(stderr," mapLocalToGlobal[%d]:%d, %d\n",i,mapLocalToGlobal[i],mapLocalToGlobal[mapLocalToGlobal[i]]);
            //fprintf(stderr,"x: %f %f\n",xCoords[i],xCoords[mapLocalToGlobal[i]]);
            //fprintf(stderr,"y: %f %f\n",yCoords[i],yCoords[mapLocalToGlobal[i]]);
            //fprintf(stderr,"z: %f %f\n",zCoords[i],zCoords[mapLocalToGlobal[i]]);
        }
        else
        {
            removedNode[i] = 1;
        }
    }

    // mapLocalToGlobal now contains the new nodes

    // change connectivity list
    for (int i = 0; i < 8 * nElem_total[0]; i++)
    {
        connList[i] = mapLocalToGlobal[connList[i]];
    }

    // Purge coordinates (remove unused coordinates)

    int *usedPoints = new int[nNodes_total[0]];
    memset(usedPoints, 0, nNodes_total[0] * sizeof(int));

    for (int i = 0; i < 8 * nElem_total[0]; i++)
    {
        usedPoints[connList[i]] = 1;
    }

    int n_usedPoints = 0;

    int *mapping = new int[nNodes_total[0]];
    memset(mapping, -1, nNodes_total[0] * sizeof(int));

    for (int i = 0; i < nNodes_total[0]; i++)
    {
        if (usedPoints[i] == 1)
        {
            mapping[i] = n_usedPoints;
            n_usedPoints++;
        }
    }

    float *xout, *yout, *zout;
    xout = new float[n_usedPoints];
    yout = new float[n_usedPoints];
    zout = new float[n_usedPoints];

    int *connlistout = new int[8 * nElem_total[0]];

    float *scalout;
    float *vectxout, *vectyout, *vectzout;

    if (readData)
    {
        if (p_scalar->getValue())
        {
            scalout = new float[n_usedPoints];
        }
        if (p_vector->getValue())
        {
            vectxout = new float[n_usedPoints];
            vectyout = new float[n_usedPoints];
            vectzout = new float[n_usedPoints];
        }
    }

    for (int i = 0; i < nNodes_total[0]; i++)
    {

        if (mapping[i] >= n_usedPoints)
        {
            fprintf(stderr, "mapping[i] >= n_usedPoints! this should never happen!\n");
        }
        if (mapping[i] != -1)
        {
            xout[mapping[i]] = xCoords[i];
            yout[mapping[i]] = yCoords[i];
            zout[mapping[i]] = zCoords[i];
        }
        if (readData)
        {
            if (p_scalar->getValue())
            {
                scalout[mapping[i]] = valScalar[i];
            }
            if (p_vector->getValue())
            {
                vectxout[mapping[i]] = valVectorX[i];
                vectyout[mapping[i]] = valVectorY[i];
                vectzout[mapping[i]] = valVectorZ[i];
            }
        }
    }

    for (int i = 0; i < 8 * nElem_total[0]; i++)
    {
        connlistout[i] = mapping[connList[i]];
    }

    delete[] usedPoints;

    memcpy(xCoords, xout, n_usedPoints * sizeof(float));
    memcpy(yCoords, yout, n_usedPoints * sizeof(float));
    memcpy(zCoords, zout, n_usedPoints * sizeof(float));

    delete[] xout;
    delete[] yout;
    delete[] zout;

    memcpy(connList, connlistout, 8 * nElem_total[0] * sizeof(int));

    delete[] connlistout;

    if (readData)
    {
        if (p_scalar->getValue())
        {
            memcpy(valScalar, scalout, n_usedPoints * sizeof(float));
            delete[] scalout;

            scalar->setSize(nNodes_total[0] - mergedNodes);

            scalar->addAttribute("SPECIES", ScalChoiceVal[p_scalar->getValue()]);
            p_scalar_3D->setCurrentObject(scalar);
        }
        if (p_vector->getValue())
        {
            memcpy(valVectorX, vectxout, n_usedPoints * sizeof(float));
            memcpy(valVectorY, vectyout, n_usedPoints * sizeof(float));
            memcpy(valVectorZ, vectzout, n_usedPoints * sizeof(float));
            delete[] vectxout;
            delete[] vectyout;
            delete[] vectzout;

            vector->setSize(nNodes_total[0] - mergedNodes);

            vector->addAttribute("SPECIES", VectChoiceVal[p_vector->getValue()]);
            p_vector_3D->setCurrentObject(vector);
        }
    }

    fprintf(stderr, "nElem total = %d\n", nElem_total[0]);
    fprintf(stderr, "nNodes total = %d\n", n_usedPoints);

    grid->setSizes(nElem_total[0], 8 * nElem_total[0], nNodes_total[0] - mergedNodes);
    p_mesh->setCurrentObject(grid);
    delete[] nNodes_total;
    delete[] nElem_total;
    return SUCCESS;
}

float ReadCGNS::findSmallestEdge(float *xCoords, float *yCoords, float *zCoords,
                                 int nElems,
                                 int *elemList, int *connList, int *typeList)
{
    // loop over elements, look for smallest edge
    // just one square root at the end saves a lot of cpu time

    float minEdge = FLT_MAX;

    int j;
    float edge[12];

    for (int i = 0; i < nElems; i++)
    {
        switch (typeList[i])
        {
        case TYPE_HEXAGON:
            edge[0] = sqrdist(xCoords[connList[elemList[i]] + 1], xCoords[connList[elemList[i]] + 0], yCoords[connList[elemList[i]] + 1], yCoords[connList[elemList[i]] + 0], zCoords[connList[elemList[i]] + 1], zCoords[connList[elemList[i]] + 0]);
            edge[1] = sqrdist(xCoords[connList[elemList[i]] + 2], xCoords[connList[elemList[i]] + 1], yCoords[connList[elemList[i]] + 2], yCoords[connList[elemList[i]] + 1], zCoords[connList[elemList[i]] + 2], zCoords[connList[elemList[i]] + 1]);
            edge[2] = sqrdist(xCoords[connList[elemList[i]] + 3], xCoords[connList[elemList[i]] + 2], yCoords[connList[elemList[i]] + 3], yCoords[connList[elemList[i]] + 2], zCoords[connList[elemList[i]] + 3], zCoords[connList[elemList[i]] + 2]);
            edge[3] = sqrdist(xCoords[connList[elemList[i]] + 0], xCoords[connList[elemList[i]] + 3], yCoords[connList[elemList[i]] + 0], yCoords[connList[elemList[i]] + 3], zCoords[connList[elemList[i]] + 0], zCoords[connList[elemList[i]] + 3]);
            edge[4] = sqrdist(xCoords[connList[elemList[i]] + 4], xCoords[connList[elemList[i]] + 0], yCoords[connList[elemList[i]] + 4], yCoords[connList[elemList[i]] + 0], zCoords[connList[elemList[i]] + 4], zCoords[connList[elemList[i]] + 0]);
            edge[5] = sqrdist(xCoords[connList[elemList[i]] + 5], xCoords[connList[elemList[i]] + 1], yCoords[connList[elemList[i]] + 5], yCoords[connList[elemList[i]] + 1], zCoords[connList[elemList[i]] + 5], zCoords[connList[elemList[i]] + 1]);
            edge[6] = sqrdist(xCoords[connList[elemList[i]] + 6], xCoords[connList[elemList[i]] + 2], yCoords[connList[elemList[i]] + 6], yCoords[connList[elemList[i]] + 2], zCoords[connList[elemList[i]] + 6], zCoords[connList[elemList[i]] + 2]);
            edge[7] = sqrdist(xCoords[connList[elemList[i]] + 7], xCoords[connList[elemList[i]] + 3], yCoords[connList[elemList[i]] + 7], yCoords[connList[elemList[i]] + 3], zCoords[connList[elemList[i]] + 7], zCoords[connList[elemList[i]] + 3]);
            edge[8] = sqrdist(xCoords[connList[elemList[i]] + 5], xCoords[connList[elemList[i]] + 4], yCoords[connList[elemList[i]] + 5], yCoords[connList[elemList[i]] + 4], zCoords[connList[elemList[i]] + 5], zCoords[connList[elemList[i]] + 4]);
            edge[9] = sqrdist(xCoords[connList[elemList[i]] + 6], xCoords[connList[elemList[i]] + 5], yCoords[connList[elemList[i]] + 6], yCoords[connList[elemList[i]] + 5], zCoords[connList[elemList[i]] + 6], zCoords[connList[elemList[i]] + 5]);
            edge[10] = sqrdist(xCoords[connList[elemList[i]] + 7], xCoords[connList[elemList[i]] + 6], yCoords[connList[elemList[i]] + 7], yCoords[connList[elemList[i]] + 6], zCoords[connList[elemList[i]] + 7], zCoords[connList[elemList[i]] + 6]);
            edge[11] = sqrdist(xCoords[connList[elemList[i]] + 4], xCoords[connList[elemList[i]] + 7], yCoords[connList[elemList[i]] + 4], yCoords[connList[elemList[i]] + 7], zCoords[connList[elemList[i]] + 4], zCoords[connList[elemList[i]] + 7]);

            for (j = 0; j < 12; j++)
            {
                if (edge[j] < minEdge)
                {
                    minEdge = edge[j];
                }
            }
            break;

        default:
            fprintf(stderr, "unsupported element type in findSmallestEdge. Please implement me!\n");
            return STOP_PIPELINE;
        }
    }

    return minEdge; // smallest edge squared
}

int ReadCGNS::mergeNodes(float *x, float *y, float *z, int numNodes, float mergeTolerance, int *localToGlobal, int &mergedNodes)
{
    // two points p1 and p2 are identical if:
    // necessary condition:      |Op1 - Op2| < tolerance
    // sufficient condition:     |xp1 - xp2|

    // sort list with distance to origin, but remember original node nrs!

    int_float_vec_ myvec;

    float value;
    for (int i = 0; i < numNodes; i++)
    {
        value = x[i] * x[i] + y[i] * y[i] + z[i] * z[i]; // we only consider square distances
        myvec.push_back(std::make_pair(i, value));
    }

    /*
    for (int i=0; i<10; i++)
    {
	fprintf(stderr,"%3d: d=%f, x=%f y=%f z=%f \n",myvec[i].first,myvec[i].second,x[myvec[i].first],y[myvec[i].first],z[myvec[i].first]);
    }
*/

    fprintf(stderr, "start sorting\n");
    sort(myvec.begin(), myvec.end(), comparator);
    fprintf(stderr, "end sorting\n");

    /*
    for (int i=0; i<10; i++)
    {
	fprintf(stderr,"%3d: d=%f, x=%f y=%f z=%f \n",myvec[i].first,myvec[i].second,x[myvec[i].first],y[myvec[i].first],z[myvec[i].first]);
    }
*/

    // loop over all the nodes in order of distance to O
    // if distance of two consecutive nodes is smaller than mergeTolerance,
    // check sufficient condition and merge if necessary
    int nodenr;
    memset(localToGlobal, -1, numNodes * sizeof(int)); // -1 means "node not merged"

    int tolRange;
    float dist;

    for (int i = 0; i < numNodes; i++)
    {
        nodenr = myvec[i].first;
        dist = myvec[i].second;

        tolRange = 1;
        // check following nodes (dist within mergeTolerance) for congruence
        // merge only nodes that have not already been merged
        while ((fabs(myvec[i + tolRange].second - dist) < mergeTolerance) && (localToGlobal[myvec[i].first] == -1))
        {
            if ((fabs(x[myvec[i].first] - x[myvec[i + tolRange].first]) < mergeTolerance) && (fabs(y[myvec[i].first] - y[myvec[i + tolRange].first]) < mergeTolerance) && (fabs(z[myvec[i].first] - z[myvec[i + tolRange].first]) < mergeTolerance))
            {
                //fprintf(stderr,"%d = %d\n",myvec[i+tolRange].first,myvec[i].first);
                localToGlobal[myvec[i + tolRange].first] = myvec[i].first;
                mergedNodes++;
                //if (tolRange>1)
                //    fprintf(stderr,"merged %d nodes\n", tolRange);
            }
            tolRange++;
        }
    }

    myvec.clear();

    return SUCCESS;
}

int ReadCGNS::createMeshDataZoneMapping(int meshFile, int dataFile, int meshBase, int dataBase, std::vector<char *> *meshZoneNames, int *zoneMapping)
{
    // int *zoneMapping;   // contains in which order to read the zone numbers of the data file (to fit to mesh file)

    CGNS_ENUMT(ZoneType_t) zonetype;
    char zonename[255];
    cgsize_t size[9];

    // get zoneNames of data file
    std::vector<char *> dataZoneNames;

    // get number of zones
    int nDataZones;
    cg_nzones(dataFile, dataBase, &nDataZones);

    int nMeshZones;
    cg_nzones(meshFile, meshBase, &nMeshZones);

    if (nMeshZones != nDataZones)
    {
        fprintf(stderr, "Error! mesh file and data file do not match - different number of zones.\n");
        return 0;
    }

    for (int i = 0; i < nMeshZones; i++)
    {
        cg_zone_read(meshFile, meshBase, i + 1, zonename, size);
        //fprintf(stderr,"mesh zone %d: name %s, size %d %d %d\n",i+1,zonename,size[0],size[1],size[2]);
    }

    for (int i = 0; i < nDataZones; i++)
    {
        cg_zone_type(dataFile, dataBase, i + 1, &zonetype);
        switch (zonetype)
        {
        case CGNS_ENUMV(Structured):
        {
            cg_zone_read(dataFile, dataBase, i + 1, zonename, size);
            //fprintf(stderr,"data zone %d: name %s, size %d %d %d\n",i+1,zonename,size[0],size[1],size[2]);
            dataZoneNames.push_back(strdup(zonename));
            break;
        }
        default:
        {
            sendError("Unrecognized zone type - only Structured is accepted up to now\n");
            break;
        }
        }
    }

    // now step through mesh zones and compare their name to the zones in data file
    for (int i = 0; i < nMeshZones; i++)
    {
        for (int j = 0; j < nDataZones; j++)
        {
            if (strcmp((*meshZoneNames)[i], dataZoneNames[j]) == 0)
            {
                zoneMapping[i] = j;
            }
        }
    }

    return SUCCESS;
}

MODULE_MAIN(IO, ReadCGNS)
