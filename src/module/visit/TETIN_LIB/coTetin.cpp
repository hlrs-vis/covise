/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin.h"
#include "coTetin__thinCut.h"
#include "coTetin__affix.h"
#include "coTetin__defCurve.h"
#include "coTetin__apprxCurve.h"
#include "coTetin__getprescPnt.h"
#include "coTetin__transGeom.h"
#include "coTetin__defDensPoly.h"
#include "coTetin__defModel.h"
#include "coTetin__defSurf.h"
#include "coTetin__matPoint.h"
#include "coTetin__mergeSurf.h"
#include "coTetin__period.h"
#include "coTetin__transl.h"
#include "coTetin__rdFamBoco.h"
#include "coTetin__bocoFile.h"
#include "coTetin__prescPnt.h"
#include "coTetin__trianTol.h"
#include "coTetin__tetinFile.h"
#include "coTetin__replayFile.h"
#include "coTetin__configDir.h"
#include "coTetin__Hexa.h"
#include "coTetin__OutputInterf.h"
#include "coTetin__trianFam.h"
#include "coTetin__delFam.h"
#include "coTetin__delSurf.h"
#include "coTetin__delCurve.h"
#include "coTetin__Proj.h"

#define VERBOSE

coTetin::coTetin(istream &str, int binary, int twoPass)
{
    if (binary)
    {
        cerr << "Binary file format not yet implemented" << endl;
        d_commands = NULL;
    }
    if (twoPass)
    {
        cerr << "Two-pass reading not yet implemented" << endl;
        d_commands = NULL;
    }
    d_commands = readFile(str, binary, twoPass);
}

// read a Tetin file ointo a string of commands

coTetinCommand *coTetin::readFile(istream &str, int binary, int twoPass)
{
    // we first produce a dummy, so we can use obj->append on this in the loop.
    // when we are ready, we kill this object. 'affix' is very simple...
    coTetinCommand *actComm, *first;
    first = actComm = new coTetin__affix(0);

    char buffer[2048];
    while (!str.eof())
    {
        str >> buffer;
        char *comment = strstr(buffer, "//");
        if (comment)
        {
            *comment = '\0';
            int len = strlen(buffer) + 1;
            str.getline(buffer + len, 2048 - len); // skip rest of line
        }

        if (!str.fail() && *buffer) // if there is anything left of this line
        {
            coTetinCommand *newComm = NULL;

            // try to read one command
            if (strcmp("affix", buffer) == 0)
                newComm = new coTetin__affix(str, 0);

            else if (strcmp("define_thin_cut", buffer) == 0)
                newComm = new coTetin__thinCut(str, 0);

            else if (strcmp("define_curve", buffer) == 0)
                newComm = new coTetin__defCurve(str, 0);

            else if (strcmp("define_density_poly", buffer) == 0)
                newComm = new coTetin__defDensPoly(str, 0);

            else if (strcmp("define_model", buffer) == 0)
                newComm = new coTetin__defModel(str, 0);

            else if (strcmp("define_surface", buffer) == 0)
                newComm = new coTetin__defSurf(str, 0);

            else if (strcmp("merge_surface", buffer) == 0)
                newComm = new coTetin__mergeSurf(str, 0);

            else if (strcmp("material_point", buffer) == 0)
                newComm = new coTetin__matPoint(str, 0);

            else if (strcmp("periodic", buffer) == 0)
                newComm = new coTetin__period(str, 0);

            else if (strcmp("prescribed_point", buffer) == 0)
                newComm = new coTetin__prescPnt(str, 0);

            else if (strcmp("translational", buffer) == 0)
                newComm = new coTetin__transl(str, 0);

            else if (strcmp("read_family_boco", buffer) == 0)
                newComm = new coTetin__rdFamBoco(str, 0);

            else if (strcmp("boco_file", buffer) == 0)
                newComm = new coTetin__transl(str, 0);

            else if (strcmp("set_triangulation_tolerance", buffer) == 0)
                newComm = new coTetin__trianTol(str, 0);

            else if (strcmp("tetin_filename", buffer) == 0)
                newComm = new coTetin__tetinFile(str, 0);

            else if (strcmp("replay_filename", buffer) == 0)
                newComm = new coTetin__replayFile(str, 0);

            else if (strcmp("config_dir_name", buffer) == 0)
                newComm = new coTetin__configDir(str, 0);

            // Not in the manual, but we assume this means to end everything
            else if (strcmp("return", buffer) == 0)
                break;

            // Inlude another file
            else if (strcmp("include", buffer) == 0)
            {
                str >> buffer;
                ifstream include(buffer);
                if (include.good())
                    newComm = readFile(include, binary, twoPass);
            }

            // we might have created more than one command (e.g. include)
            if (newComm && newComm->isValid())
            {
                actComm->append(newComm);
                while ((newComm = actComm->getNext()) != NULL)
                    actComm = newComm;
            }
        }
    }

    // remove first (dummy) element and return everything behind
    actComm = first->getNext();
    delete first;
    return actComm;
}

/// make a Tetin object from a covise_binary object

coTetin::coTetin(DO_BinData *binObj)
{
    // get the fields
    int *intData = binObj->getIntArr();
    float *floatData = binObj->getFloatArr();
    char *charData = binObj->getCharArr();

    // first is the number of commands
    int numCommands = *intData++;

    // we first produce a dummy, so we can use obj->append on this in the loop.
    // when we are ready, we kill this object. 'affix' is very simple...
    coTetinCommand *actComm;
    d_commands = actComm = new coTetin__affix(0);

    // loop along the commands
    int i;
    for (i = 0; i < numCommands; i++)
    {
        int comm = *intData++;
        coTetinCommand *newComm = NULL;

        switch (comm)
        {
        case coTetin::BOCO_FILE:
            coTetin__bocoFile(intData, floatData, charData);
            break;

        case coTetin::READ_FAMILY_BOCO:
            coTetin__rdFamBoco(intData, floatData, charData);
            break;

        case coTetin::AFFIX:
            newComm = new coTetin__affix(intData, floatData, charData);
            break;

        case coTetin::DEFINE_CURVE:
            newComm = new coTetin__defCurve(intData, floatData, charData);
            break;

        case coTetin::DEFINE_DENSITY_POLYGON:
            newComm = new coTetin__defDensPoly(intData, floatData, charData);
            break;

        case coTetin::DEFINE_MODEL:
            newComm = new coTetin__defModel(intData, floatData, charData);
            break;

        case coTetin::DEFINE_SURFACE:
            newComm = new coTetin__defSurf(intData, floatData, charData);
            break;

        case coTetin::DEFINE_THIN_CUT:
            newComm = new coTetin__thinCut(intData, floatData, charData);
            break;

        case coTetin::MATERIAL_POINT:
            newComm = new coTetin__matPoint(intData, floatData, charData);
            break;

        case coTetin::MERGE_SURFACE:
            newComm = new coTetin__mergeSurf(intData, floatData, charData);
            break;

        case coTetin::PERIODIC:
            newComm = new coTetin__period(intData, floatData, charData);
            break;

        case coTetin::PRESCRIBED_POINT:
            newComm = new coTetin__prescPnt(intData, floatData, charData);
            break;

        case coTetin::SET_TRIANGULATION_TOLERANCE:
            newComm = new coTetin__trianTol(intData, floatData, charData);
            break;

        case coTetin::TETIN_FILENAME:
            newComm = new coTetin__tetinFile(intData, floatData, charData);
            break;

        case coTetin::REPLAY_FILENAME:
            newComm = new coTetin__replayFile(intData, floatData, charData);
            break;

        case coTetin::CONFIGDIR_NAME:
            newComm = new coTetin__configDir(intData, floatData, charData);
            break;

        case coTetin::START_HEXA:
            newComm = new coTetin__Hexa(intData, floatData, charData);
            break;

        case coTetin::OUTPUT_INTERF:
            newComm = new coTetin__OutputInterf(intData, floatData, charData);
            break;

        case coTetin::TRIANGULATE_FAMILY:
            newComm = new coTetin__trianFam(intData, floatData, charData);
            break;

        case coTetin::APPROXIMATE_CURVE:
            newComm = new coTetin__apprxCurve(intData, floatData, charData);
            break;

        case coTetin::GET_PRESCPNT:
            newComm = new coTetin__getprescPnt(intData, floatData, charData);
            break;

        case coTetin::TRANSLATE_GEOM:
            newComm = new coTetin__transGeom(intData, floatData, charData);
            break;

        case coTetin::DELETE_FAMILY:
            newComm = new coTetin__delFam(intData, floatData, charData);
            break;

        case coTetin::DELETE_SURFACE:
            newComm = new coTetin__delSurf(intData, floatData, charData);
            break;

        case coTetin::DELETE_CURVE:
            newComm = new coTetin__delCurve(intData, floatData, charData);
            break;

        case coTetin::PROJECT_POINT:
            newComm = new coTetin__Proj(intData, floatData, charData);
            break;

        case coTetin::TRANSLATIONAL:
            newComm = new coTetin__transl(intData, floatData, charData);
            break;

        default:
            cerr << "unknown command ID in SHM : " << comm << endl;
            break;
        }

        // forward by one command
        if (newComm && newComm->isValid())
        {
            actComm->append(newComm);
            actComm = newComm;
        }
        else
            delete newComm;
    }

    // remove dummy
    actComm = d_commands->getNext();
    delete d_commands;
    d_commands = actComm;
}

coTetin::~coTetin()
{
}

const char *coTetin::getType() const
{
    return "TETIN";
}

void coTetin::getSize(int &numInt, int &numFloat, int &numChar) const
{
    // set empty
    numInt = 1; // number of command in chain
    numFloat = 0;
    numChar = 0;

    // now run along the chain and sum up
    coTetinCommand *comm = d_commands;
    while (comm)
    {

        comm->addSizes(numInt, numFloat, numChar);
        comm = comm->getNext();
    }
}

void coTetin::getBinary(int *intDat, float *floatDat, char *charDat) const
{
    // now run along the chain and put everything in
    coTetinCommand *comm = d_commands;

    // count number of commands in 1st integer field
    int &numComm = *intDat++;
    numComm = 0;

    // now run along the chain
    while (comm)
    {
        if (comm->isValid())
        {
            comm->getBinary(intDat, floatDat, charDat);
            comm = comm->getNext();
            numComm++;
        }
    }
}

void coTetin::append(coTetinCommand *newComm)
{
    if (d_commands)
    {
        d_commands->append(newComm);
    }
    else
    {
        d_commands = newComm;
    }
}

void coTetin::print(ostream &str)
{
    long flags = str.flags();
    str.precision(9);
    str.setf(ios::fixed);

    coTetinCommand *comm = d_commands;
    while (comm)
    {
        comm->print(str);
        comm = comm->getNext();
    }
    str.flags(flags);
}
