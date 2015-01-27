/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <ctype.h>
#include <iostream.h>
#include <fstream.h>
#include "FC_StdIO.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include "coviseFiles.h"

static const int ILL = 8;
static const int HEX = 7;
static const int PRI = 6;
static const int PYR = 5;
static const int TET = 4;

FC_String mapOldFireNames(const FC_String &nameIn);

char *reduceName(const char *name, const char *basename)
{
    char *res = new char[strlen(name) + strlen(basename) + 20];
    strcpy(res, basename);
    strcat(res, "/");
    strcat(res, name);
    char *end = res;
    while (*end && (isalpha(*end) || isdigit(*end) || *end == '/'
                    || *end == '.' || *end == ':' || *end == '_'))
        end++;
    *end = '\0';

    char *dblpoint = strchr(res, ':');
    if (dblpoint)
    {
        *dblpoint = '\0';
        mkdir(res, 0777);
        *dblpoint = '/';
    }

    strcat(res, ".covise");
    //cout << name << " + " << basename << " ---> " << res << endl;
    return res;
}

void writeAllDataSets(FC_StdIO &file, covOutFile **outfiles, char **outnames,
                      int stepNo, int numSteps, int &geoIdx, int &lnkIdx,
                      const char *basename)
{
    cout << "Read step " << stepNo << ":  " << flush;

    int version, numData;
    FC_String name, unit;
    FC_StdIO::eDataType type;

    FC_Array<int> intData;
    FC_Array<float> floatData;

    file.GotoDataSet(stepNo);

    //float refPressure;

    // loop until end of dataset to find all headers
    while (file.InDataSet(stepNo) == FC_StdIO::ok)
    {

        file.ReadDataHeader(version, name, type, numData);
        if (!file)
            return;

        // convert old names version < 7.X
        name.DeleteBlanksAndTabs();
        name = mapOldFireNames(name);

        ///////////////////////// READ DATA /////
        // data is int
        if (type == FC_StdIO::data_int)
        {
            file.Read(numData, intData);
            if (!file)
                return;
        }
        // data is float
        else if (type == FC_StdIO::data_float)
        {
            file.Read(numData, floatData);
            if (!file)
                return;
        }
        //////////////////////////////////////////

        // read step
        if (name.IsSubStr("DataSet:Geo"))
            geoIdx = (int)floatData[0];
        else if (name.IsSubStr("DataSet:Lnk"))
            lnkIdx = (int)floatData[0];
        else if (name.IsSubStr("ReferencePressure"))
        {
            //refPressure = floatData[0];
        }

        // write fields - only floats
        else if (type == FC_StdIO::data_float)
        {
            char *nameBase = reduceName(name, basename);

            cout << "." << flush;

            // find data file
            char **dName = outnames;
            covOutFile **oFile = outfiles;
            while (*dName && strcmp(*dName, nameBase))
            {
                dName++;
                oFile++;
            }

            // not yet created -> create new and begin set
            if (!*oFile)
            {
                *oFile = new covOutFile(nameBase);
                *dName = strcpy(new char[strlen(nameBase) + 1], nameBase);
                (*oFile)->writeSetHeader(numSteps);
            }

            // attributes
            const char *atNam[] = { "SPECIES", NULL };
            const char *atVal[] = { name, NULL };

            (*oFile)->writeS3D(numData, floatData, atNam, atVal);
        }
    }
    cout << endl;
}

void writeGeometry(covOutFile &gridfile, FC_StdIO &geoFile, int geoIdx,
                   FC_StdIO &lnkFile, int lnkIdx)
{
    FC_Array<float> xcoor, ycoor, zcoor;
    FC_String name;
    FC_StdIO::eDataType type;
    FC_Array<int> lcvDum[8];

    int version, numVert, numConn, numCell, i, nd, nc;

    geoFile.GotoDataSet(geoIdx);
    if (!geoFile)
        return;

    // read x-coordinates
    geoFile.ReadDataHeader(version, name, type, numVert);
    if (!geoFile)
        return;
    geoFile.Read(numVert, xcoor);
    cout << "read X-coord [" << numVert << "]" << endl;
    if (!geoFile)
        return;

    // read y-coordinates
    geoFile.ReadDataHeader(version, name, type, numVert);
    if (!geoFile)
        return;
    geoFile.Read(numVert, ycoor);
    cout << "read Y-coord [" << numVert << "]" << endl;
    if (!geoFile)
        return;

    // read z-coordinates
    geoFile.ReadDataHeader(version, name, type, numVert);
    if (!geoFile)
        return;
    geoFile.Read(numVert, zcoor);
    cout << "read Z-coord [" << numVert << "]" << endl;
    if (!geoFile)
        return;

    // -----------------------------------------
    // read lnk Data
    // -----------------------------------------
    if (!lnkFile)
        return;

    lnkFile.GotoDataSet(lnkIdx);
    if (!lnkFile)
        return;

    // loop until end of dataset to find all headers
    while (lnkFile.InDataSet(lnkIdx) == FC_StdIO::ok)
    {
        // read data header
        lnkFile.ReadDataHeader(version, name, type, nc);
        if (!lnkFile)
            return;

        // if lcv found get out direction and read data
        name.DeleteBlanksAndTabs();
        if (name.Find("LCVINDIRECTION").Length())
        {
            // secure value: other fields have other size: do not overwrite
            numCell = nc;

            cout << "reading " << name << " [" << numCell << "]" << endl;
            FC_String dir = name.Extract(14);
            dir.DeleteBlanksAndTabsAtBorder();
            nd = dir.ToInt();

            // error in direction
            if (nd <= 0)
                return;

            lnkFile.Read(numCell, lcvDum[nd - 1]);
            if (!lnkFile)
                return;
        }
        else
        {
            cout << "skipping " << name << endl;
            lnkFile.GotoNextHeader();
            if (!lnkFile)
                return;
        }
    }

    //// Find out element types and count connectivities
    int *typeList = new int[numCell];
    int *elemList = new int[numCell + 1]; //last one not used, but easier to build
    elemList[0] = 0;

    numConn = 0;
    for (i = 0; i < numCell; i++)
    {
        int &a = lcvDum[0][i];
        int &b = lcvDum[1][i];
        int &c = lcvDum[2][i];
        int &d = lcvDum[3][i];
        int &e = lcvDum[4][i];
        int &f = lcvDum[5][i];
        int &g = lcvDum[6][i];
        int &h = lcvDum[7][i];

        typeList[i] = ILL;

        /// all ups != all downs
        if (a != e && a != f && a != g && a != h
            && b != e && b != f && b != g && b != h
            && c != e && c != f && c != g && c != h
            && d != e && d != f && d != g && d != h)
        {
            /// Quad base
            if (d != c && d != b && d != a && c != b && c != a && b != a)
            {
                // Quad top -> HEX
                if (h != g && h != f && h != e && g != f && g != e && f != e)
                {
                    typeList[i] = HEX;
                    numConn += 8;
                }
                // Point top -> Pyr
                else if (h == g && g == f && f == e)
                {
                    typeList[i] = PYR;
                    numConn += 5;
                }
            }

            // Triangle base
            else if (d == c && c != b && c != a && b != a)
            {
                // Triangle top -> PRI
                if (h == g && g != f && g != e && f != e)
                {
                    typeList[i] = PRI;
                    numConn += 6;
                }
                // Point top -> TET
                else if (h == g && g == f && f == e)
                {
                    typeList[i] = TET;
                    numConn += 4;
                }
            }
        }

        if (typeList[i] == ILL)
        {
            numConn += 8;
            typeList[i] = HEX;
        }

        elemList[i + 1] = numConn;
    }

    // build connlist

    int *connList = new int[numCell * 8];
    int *connPtr = connList;

    for (i = 0; i < numCell; i++)
    {
        switch (typeList[i])
        {
        case TET:
            *connPtr = lcvDum[0][i] - 1;
            connPtr++;
            *connPtr = lcvDum[1][i] - 1;
            connPtr++;
            *connPtr = lcvDum[2][i] - 1;
            connPtr++;
            *connPtr = lcvDum[4][i] - 1;
            connPtr++;
            break;

        case PYR:
            *connPtr = lcvDum[0][i] - 1;
            connPtr++;
            *connPtr = lcvDum[1][i] - 1;
            connPtr++;
            *connPtr = lcvDum[2][i] - 1;
            connPtr++;
            *connPtr = lcvDum[3][i] - 1;
            connPtr++;
            *connPtr = lcvDum[4][i] - 1;
            connPtr++;
            break;

        case PRI:
            *connPtr = lcvDum[0][i] - 1;
            connPtr++;
            *connPtr = lcvDum[1][i] - 1;
            connPtr++;
            *connPtr = lcvDum[2][i] - 1;
            connPtr++;
            *connPtr = lcvDum[4][i] - 1;
            connPtr++;
            *connPtr = lcvDum[5][i] - 1;
            connPtr++;
            *connPtr = lcvDum[6][i] - 1;
            connPtr++;
            break;

        case HEX:
        case ILL:
            *connPtr = lcvDum[0][i] - 1;
            connPtr++;
            *connPtr = lcvDum[1][i] - 1;
            connPtr++;
            *connPtr = lcvDum[2][i] - 1;
            connPtr++;
            *connPtr = lcvDum[3][i] - 1;
            connPtr++;
            *connPtr = lcvDum[4][i] - 1;
            connPtr++;
            *connPtr = lcvDum[5][i] - 1;
            connPtr++;
            *connPtr = lcvDum[6][i] - 1;
            connPtr++;
            *connPtr = lcvDum[7][i] - 1;
            connPtr++;
            break;

        default:
            cout << "Unknown CEll type: " << typeList[i] << endl;
            break;
        }
    }

    // attributes
    const char *atNam[] = { "CREATOR", NULL };
    const char *atVal[] = { "fire2covise", NULL };

    gridfile.writeUSG(numCell, numConn, numVert, elemList, connList, typeList,
                      xcoor.GetPtr(), ycoor.GetPtr(), zcoor.GetPtr(), atNam, atVal);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    FC_String floFileName, geoFileName, lnkFileName, filename;
    int geoIdx, linkIdx;
    int i;

    if (argc < 2)
    {
        cout << "File: " << flush;
        cin >> filename;
    }
    else
        filename = argv[1];

    // create, stop on all errors except 'already exist'
    if (mkdir(filename, 0777) < 0 && errno != EEXIST)
    {
        cerr << "cannot create output dir '" << filename << "'" << endl;
        return -1;
    }

    floFileName = filename + ".flo";
    geoFileName = filename + ".geo";
    lnkFileName = filename + ".lnk";

    FC_StdIO floFile, geoFile, lnkFile;

    floFile.DisplayErrorMessages(1);
    floFile.Open(floFileName, FC_StdIO::read);
    if (!floFile)
        return -1;

    geoFile.DisplayErrorMessages(1);
    geoFile.Open(geoFileName, FC_StdIO::read);
    if (!geoFile)
        return -1;

    lnkFile.DisplayErrorMessages(1);
    lnkFile.Open(lnkFileName, FC_StdIO::read);
    if (!lnkFile)
        return -1;

    // find number of steps
    int maxIdx = floFile.NumberOfDataSets();
    int *stepIdx = new int[maxIdx]; // more than really needed...
    int numSteps = 0;
    for (i = 0; i <= maxIdx; i++)
    {
        if (floFile.DataSetIsAvailable(i) == FC_StdIO::ok)
        {
            stepIdx[numSteps] = i;
            numSteps++;
        }
    }

    cout << "Found " << numSteps << " Steps, maxIdx=" << maxIdx << endl;
    if (numSteps == 0)
        return -1;

    /// open geometry file
    FC_String gridFileName = filename + "/grid.covise";
    cout << "Creating file '" << gridFileName << "'" << endl;
    covOutFile gridfile(gridFileName);
    gridfile.writeSetHeader(numSteps);

    /// split output data into separate files
    enum
    {
        MAX_SPEC = 256
    };
    covOutFile *outfiles[MAX_SPEC];
    char *outnames[MAX_SPEC];
    for (i = 0; i < MAX_SPEC; i++)
        outfiles[i] = NULL;

    for (i = 0; i < numSteps; i++)
    {
        int &stepNo = stepIdx[i];
        floFile.GotoDataSet(stepNo);

        if (floFile.InDataSet(stepNo) != FC_StdIO::ok)
        {
            cerr << "Step " << stepNo << " not found, terminating" << endl;
            return -1;
        }

        // writes all data sets, converts if necessary
        writeAllDataSets(floFile, outfiles, outnames, stepNo, numSteps, geoIdx, linkIdx, filename);

        writeGeometry(gridfile, geoFile, geoIdx, lnkFile, linkIdx);
    }
    // finalize sets
    char buffer[23];
    sprintf(buffer, "0 %d", numSteps - 1);
    char *names[] = { "CREATOR", "TIMESTEP", NULL };
    char *values[] = { "fire2asc v1.0", buffer, NULL };
    covOutFile **str = outfiles;
    while (*str)
    {
        (*str)->writeattrib(names, values);
        delete (*str);
        str++;
    }
    gridfile.writeattrib(names, values);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////

FC_String mapOldFireNames(const FC_String &nameIn)
{
    if (nameIn == "U-VELOCITIES")
        return "Flow:Velocity.U[m/s]";
    else if (nameIn == "V-VELOCITIES")
        return "Flow:Velocity.V[m/s]";
    else if (nameIn == "W-VELOCITIES")
        return "Flow:Velocity.W[m/s]";

    else if (nameIn == "Flow:U-Velocity")
        return "Flow:Velocity.U[m/s]";
    else if (nameIn == "Flow:V-Velocity")
        return "Flow:Velocity.V[m/s]";
    else if (nameIn == "Flow:W-Velocity")
        return "Flow:Velocity.W[m/s]";

    else if (nameIn == "SYSTEMU-VELOCITIES")
        return "Flow:SysVelocity.U[m/s]";
    else if (nameIn == "SYSTEMV-VELOCITIES")
        return "Flow:SysVelocity.V[m/s]";
    else if (nameIn == "SYSTEMW-VELOCITIES")
        return "Flow:SysVelocity.W[m/s]";

    else if (nameIn == "PRESSURE")
        return "Flow:Pressure[Pa]";
    else if (nameIn == "TURBULENTKINETICENERGY")
        return "Flow:T.K.Energy[m^2/s^2]";
    else if (nameIn == "DISSIPATIONRATE")
        return "Flow:Dissipation Rate[m^2/s^3]";
    else if (nameIn == "PASSIVESCALAR")
        return "Flow:Passive Scalar[-]";
    else if (nameIn == "DENSITY")
        return "Flow:Density[kg/m^3]";
    else if (nameIn == "TEMPERATURE")
        return "Flow:Temperature[K]";
    else if (nameIn == "VISCOSITY")
        return "Flow:Viscosity[kg/ms]";
    else if (nameIn == "GENERATIONRATE")
        return "Flow:Generationrate[1/s^2]";
    else if (nameIn == "HEATFLUX")
        return "Flow:Heatflux[W/m^2]";
    else if (nameIn == "YPLUSWALL")
        return "Flow:Ypluswall[-]";
    else if (nameIn == "ERRORMOMENTUM")
        return "Flow:ErrorMomentum[-]";

    else if (nameIn.IsSubStr("MAIN:AIR"))
        return "MainAir[-]";
    else if (nameIn.IsSubStr("SPRAY:NUMBOFDROPLET"))
        return "SprayDroplet:NumberOfDroplets[-]";
    else if (nameIn == "SPRAY:ACCESSNUMBOFDROPLET")
        return "SprayDroplet:AccessNumber[-]";
    else if (nameIn == "SPRAY:CELLNUMBOFDROPLET")
        return "SprayDroplet:CellNumber[-]";
    else if (nameIn == "SPRAY:DROPLETX")
        return "SprayDroplet:Coord.x[m]";
    else if (nameIn == "SPRAY:DROPLETY")
        return "SprayDroplet:Coord.y[m]";
    else if (nameIn == "SPRAY:DROPLETZ")
        return "SprayDroplet:Coord.z[m]";
    else if (nameIn == "SPRAY:DROPLETLIFETIME")
        return "SprayDroplet:LifeTime[s]";
    else if (nameIn == "SPRAY:DROPLETUVEL")
        return "SprayDroplet:Velocity.U[m/s]";
    else if (nameIn == "SPRAY:DROPLETVVEL")
        return "SprayDroplet:Velocity.V[m/s]";
    else if (nameIn == "SPRAY:DROPLETWVEL")
        return "SprayDroplet:Velocity.W[m/s]";
    else if (nameIn == "SPRAY:DROPLETUSYSVEL")
        return "SprayDroplet:SysVelocity.U[m/s]";
    else if (nameIn == "SPRAY:DROPLETVSYSVEL")
        return "SprayDroplet:SysVelocity.V[m/s]";
    else if (nameIn == "SPRAY:DROPLETWSYSVEL")
        return "SprayDroplet:SysVelocity.W[m/s]";
    else if (nameIn == "SPRAY:DROPLETDIAM")
        return "SprayDroplet:Diameter[m]";
    else if (nameIn == "SPRAY:DROPLETDENSITY")
        return "SprayDroplet:Density[kg/m^3]";
    else if (nameIn == "SPRAY:DROPLETTEMP")
        return "SprayDroplet:Temperature[K]";
    else if (nameIn == "SPRAY:DROPLETIMPING")
        return "SprayDroplet:Impinged[-]";
    else if (nameIn == "SPRAY:DROPLETPARCELS")
        return "SprayDroplet:Parcels[-]";
    else if (nameIn == "SPRAY:BORENUMBOFDROPLET")
        return "SprayDroplet:BoreID[-]";
    else if (nameIn == "SPRAY:NOZZLENUMBOFDROPLET")
        return "SprayDroplet:NozzleID[-]";
    else if (nameIn == "SPRAY:WEBERNUMBOFDROPLET")
        return "SprayDroplet:WeberNumber[-]";
    else if (nameIn == "SPRAY:TURBLENTSTOKESNUMB")
        return "SprayDroplet:TurbulentStokesNumber[-]";
    else if (nameIn == "SPRAY:PARTICELRELAXATIONTIME")
        return "SprayDroplet:RelaxationTime[s]";
    else if (nameIn == "SPRAY:VAPOURMASSFRACTION")
        return "Spray:VapourMassFraction[-]";
    else if (nameIn == "SPRAY:VOIDFRACTION")
        return "Spray:VoidFraction[-]";
    else if (nameIn == "SPRAY:LIQUIDDENSITY")
        return "Spray:LiquidDensity[kg/m^3]";
    else if (nameIn == "SPRAY:MEANDIAMETERD10")
        return "Spray:MeanDiameter-d10[m]";
    else if (nameIn == "SPRAY:MEANVOLUMEDIAMETERD30")
        return "Spray:MeanVolumeDiameter-d30[m]";
    else if (nameIn == "SPRAY:SAUTERDIAMETERD32")
        return "Spray:SauterDiameter-d32[m]";
    else if (nameIn == "SPRAY:NUMBERDENSITY")
        return "Spray:NumberDensity[-]";

    else if (nameIn == "COMBFUELFRAC")
        return "Comb:FuelFraction[-]";
    else if (nameIn == "COMBMIXTUREFRAC")
        return "Comb:MixtureFraction[-]";
    else if (nameIn == "COMBPROGVAR")
        return "Comb:ProgressVariable[-]";
    else if (nameIn == "COMBFLUCTINT")
        return "Comb:FluctuationIntensity[-]";
    else if (nameIn == "COMBEGRRATE")
        return "Comb:EGRRate[-]";
    else if (nameIn == "COMBRADICALRMASS")
        return "Comb:RadicalRMass[-]";
    else if (nameIn == "COMBAGENTBMASS")
        return "Comb:AgentBMass[-]";
    else if (nameIn == "COMBINTERQMASS")
        return "Comb:InterQMass[-]";
    else if (nameIn == "COMBREACTIONRATE")
        return "Comb:ReactionRate[-]";
    else if (nameIn == "PASSIVESCALAR1")
        return "Flow:PassiveScalar1[-]";
    else if (nameIn == "PASSIVESCALAR2")
        return "Flow:PassiveScalar2[-]";
    else if (nameIn == "PASSIVESCALAR3")
        return "Flow:PassiveScalar3[-]";

    else if (nameIn == "WFILM:FILMTHICKNESS")
        return "Wallfilm:FilmThickness[m]";
    else if (nameIn == "WFILM:FILMVELOCITYU")
        return "Wallfilm:FilmVelocity.U[m/s]";
    else if (nameIn == "WFILM:FILMVELOCITYV")
        return "Wallfilm:FilmVelocity.V[m/s]";
    else if (nameIn == "WFILM:FILMVELOCITYW")
        return "Wallfilm:FilmVelocity.W[m/s]";
    else if (nameIn == "WFILM:WALLSHEARSTRESS")
        return "Wallfilm:WallShearstress[N/m2]";
    else if (nameIn == "WFILM:FILMTEMPERATURE")
        return "Wallfilm:FilmTemperature[K]";
    else if (nameIn == "WFILM:FILMEVAPOURATIONRATE")
        return "Wallfilm:FilmEvapourationRate[kg/sm2]";
    else if (nameIn == "WFILM:FILMENTRAINMENTRATE")
        return "Wallfilm:FilmEntrainmentRate[kg/sm2]";
    else if (nameIn == "WFILM:SPRAY-FILMIMPINGEMENT")
        return "Wallfilm:SprayFilmImpingement[kg/sm2]";
    else if (nameIn == "WFILM:FILMREYNOLDSNUMBER")
        return "Wallfilm:FilmReynoldsNumber[-]";
    else if (nameIn == "WFILM:FILMWEBERNUMBER")
        return "Wallfilm:FilmWeberNumber[-]";
    else if (nameIn == "WFILM:FILMUSEROUTPUT")
        return "Wallfilm:FilmUserOutput[-]";

    else if (nameIn == "REYNOLDSSTRESSUU")
        return "Rsm:Stress.UU[m^2/s^2]";
    else if (nameIn == "REYNOLDSSTRESSVV")
        return "Rsm:Stress.VV[m^2/s^2]";
    else if (nameIn == "REYNOLDSSTRESSWW")
        return "Rsm:Stress.WW[m^2/s^2]";
    else if (nameIn == "REYNOLDSSTRESSUV")
        return "Rsm:Stress.uv[m^2/s^2]";
    else if (nameIn == "REYNOLDSSTRESSUW")
        return "Rsm:Stress.uw[m^2/s^2]";
    else if (nameIn == "REYNOLDSSTRESSVW")
        return "Rsm:Stress.vw[m^2/s^2]";

    else if (nameIn == "PHASE2U-VELOCITIES")
        return "Ph_2:Velocity.U[m/s]";
    else if (nameIn == "PHASE2V-VELOCITIES")
        return "Ph_2:Velocity.V[m/s]";
    else if (nameIn == "PHASE2W-VELOCITIES")
        return "Ph_2:Velocity.W[m/s]";
    else if (nameIn == "PHASE2PRESSURE")
        return "Ph_2:Pressure[Pa]";
    else if (nameIn == "PHASE2TURBULENTKINETICENE")
        return "Ph_2:T.K.Energy[m^2/s^2]";
    else if (nameIn == "PHASE2DISSIPATIONRATE")
        return "Ph_2:Diss.Rate[m^2/s^3]";
    else if (nameIn == "PHASE2DENSITY")
        return "Ph_2:Density[kg/m^3]";
    else if (nameIn == "PHASE2TEMPERATURE")
        return "Ph_2:Temperature[K]";
    else if (nameIn == "PHASE2VISCOSITY")
        return "Ph_2:Viscosity[kg/ms]";
    else if (nameIn == "PHASE2VOLUMEFRACTION")
        return "Ph_2:VolumeFrac[-]";
    else if (nameIn == "PHASE2MASSEXCHRATE")
        return "Ph_2:MassExchRate[kg/s]";
    else if (nameIn == "PHASE2INTERFACEAREA")
        return "Ph_2:InterFaceArea[m^2]";

    else if (nameIn == "ITIME,CELLNUMBERS")
        return "Itime,CellNumbers";
    else if (nameIn == "USEDTIME")
        return "UsedTime";
    else if (nameIn == "REFERENCEPRESSURE")
        return "ReferencePressure";
    else if (nameIn == "BLOCKINTERPOLFACTORS")
        return "DataSet:Geo";
    else if (nameIn == "LNKVALUE")
        return "DataSet:Lnk";

    if (nameIn.IsSubStr("SPRAY:DROPLET"))
        return "SprayDroplet:" + (nameIn - "Spray:Droplet");
    if (nameIn.IsSubStr("Spray:Droplet"))
        return "SprayDroplet:" + (nameIn - "Spray:Droplet");

    FC_String newName = nameIn;
    newName.DeleteBlanksAndTabs();
    newName.Replace('-', '_');
    return newName;
}
