/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FC_StdIO.h"

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

void ReadResultFire7x(const FC_String &fileName, const int &dataset1, int &geoData, int &lnkData)
{
    FC_StdIO file;
    int version, numData;
    FC_String name, unit;
    FC_StdIO::eDataType type;

    // -----------------------------------------
    // open file and enable displaying of error messages
    // -----------------------------------------
    file.DisplayErrorMessages(1);
    file.Open(fileName, FC_StdIO::read);
    if (!file)
        return;

    // check if dataset is avaiable
    if (file.DataSetIsAvailable(dataset1) == FC_StdIO::ok)
    {
        file.GotoDataSet(dataset1);
        if (!file)
            return;
    }
    else
        return;

    FC_Array<int> intData;
    FC_Array<float> floatData;

    // loop until end of dataset to find all headers
    while (file.InDataSet(dataset1) == FC_StdIO::ok)
    {
        file.ReadDataHeader(version, name, type, numData);
        if (!file)
            return;

        // convert old names version < 7.X
        name.DeleteBlanksAndTabs();
        name = mapOldFireNames(name);
        cout << "reading " << name << endl;

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

        if (name.IsSubStr("ReferencePressure"))
            cout << "ReferencePressure = " << floatData[0] << endl;
        else if (name.IsSubStr("Itime,CellNumbers"))
            cout << "TimeStep = " << intData[0] << endl;
        else if (name.IsSubStr("DataSet:Geo"))
        {
            cout << "GeoFlag = " << (int)floatData[0] << endl;
            geoData = (int)floatData[0];
        }
        else if (name.IsSubStr("DataSet:Lnk"))
        {
            cout << "LnkFlag = " << (int)floatData[0] << endl;
            lnkData = (int)floatData[0];
        }
        else if (name.IsSubStr("UsedTime"))
            cout << "Actual Time = " << floatData[0] << endl;
        else if (name.IsSubStr("SprayDroplet:"))
        {
            cout << "Droplet data" << endl;
        }
        else
        {
            cout << "Cell data" << endl;
        }
    }
}

void ReadGeoFire7x(const FC_String &filename, int geoFlagIn)
{
    FC_StdIO file;

    FC_Array<float> xcoor, ycoor, zcoor;

    int version, numData;
    FC_String name;
    FC_StdIO::eDataType type;

    // -----------------------------------------
    // read Geo Data
    // -----------------------------------------
    file.DisplayErrorMessages(1);
    file.Open(filename, FC_StdIO::read);
    if (!file)
        return;

    file.GotoDataSet(geoFlagIn);
    if (!file)
        return;

    // read x-coordinates
    file.ReadDataHeader(version, name, type, numData);
    cout << "reading " << name << endl;
    if (!file)
        return;
    file.Read(numData, xcoor);
    if (!file)
        return;

    // read y-coordinates
    file.ReadDataHeader(version, name, type, numData);
    cout << "reading " << name << endl;
    if (!file)
        return;
    file.Read(numData, ycoor);
    if (!file)
        return;

    // read z-coordinates
    file.ReadDataHeader(version, name, type, numData);
    cout << "reading " << name << endl;
    if (!file)
        return;
    file.Read(numData, zcoor);
    if (!file)
        return;
}

void ReadLnkFire7x(const FC_String &filename, int lnkFlagIn)
{
    FC_StdIO file;

    int version, numData, nd;
    FC_String name;
    FC_StdIO::eDataType type;

    FC_Array<int> lcvDum[8];

    // -----------------------------------------
    // read lnk Data
    // -----------------------------------------
    file.DisplayErrorMessages(1);
    file.Open(filename, FC_StdIO::read);
    if (!file)
        return;

    file.GotoDataSet(lnkFlagIn);
    if (!file)
        return;

    // loop until end of dataset to find all headers
    while (file.InDataSet(lnkFlagIn) == FC_StdIO::ok)
    {
        // read data header
        file.ReadDataHeader(version, name, type, numData);
        if (!file)
            return;

        // if lcv found get out direction and read data
        name.DeleteBlanksAndTabs();
        if (name.Find("LCVINDIRECTION").Length())
        {
            cout << "reading " << name << endl;
            FC_String dir = name.Extract(14);
            dir.DeleteBlanksAndTabsAtBorder();
            nd = dir.ToInt();

            // error in direction
            if (nd <= 0)
                return;

            file.Read(numData, lcvDum[nd - 1]);
            if (!file)
                return;
        }
        else
        {
            cout << "skipping " << name << endl;
            file.GotoNextHeader();
            if (!file)
                return;
        }
    }
}

int main(int narg, char **argv)
{
    FC_String floFile, geoFile, lnkFile;
    int floData, geoData, lnkData;

    cout << "floFile: ";
    cin >> floFile;

    cout << "dataSet: ";
    cin >> floData;

    cout << "geoFile: ";
    cin >> geoFile;

    cout << "lnkFile: ";
    cin >> lnkFile;

    ReadResultFire7x(floFile, floData, geoData, lnkData);
    ReadGeoFire7x(geoFile, geoData);
    ReadLnkFire7x(lnkFile, lnkData);

    return 0;
}

double ceil(double x)
{
}
