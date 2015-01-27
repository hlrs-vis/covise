/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                     (C)2002 VirCinity  **
 **                                                                        **
 ** Description: Read Star Geometry for DC Simulation Coupling             **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** History:                                                               **
 **      07/2002    A. Werner      Initial version                         **
 *\**************************************************************************/

#include "VisitPro.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <strstream.h>

#include "MultiFileParam.h"
#include <api/coFeedback.h>
#include "AddGeom.h"
#include "Attachable.h"
#include "ReadObj.h"

void main(int argc, char *argv[])
{
    VisitPro *application = new VisitPro();
    application->start(argc, argv);
}

///      ####   #  #####   ####   #####
//     #    #   #    #    #    #  #    #
//     #       #     #    #    #  #    #
//     #             #    #    #  #####
//     #    #        #    #    #  #   #
//      ####         #     ####   #    #

VisitPro::VisitPro()
    : coModule("StarCD VISiT module")
{
    // create the output ports
    p_baseOut = addOutputPort("baseOut", "coDoPolygons", "Base Geometry");
    p_attachOut = addOutputPort("attachOut", "coDoPolygons", "Base Geometry");
    p_starOut = addOutputPort("starOut", "coDoText", "StarCD steering");

    // the input files are multi-file selectors
    d_baseGeo = new MultiFileParam("BaseGeo", "Base Geometry", "data/none.obj",
                                   "*.obj;*.OBJ;*.Obj", this);
    d_addPart = new MultiFileParam("AddGeo", "Add Geometry", "data/none.obj",
                                   "*.obj;*.OBJ;*.Obj", this);

    d_objUnit = addChoiceParam("objUnit", "Unit in Obj files");
    static const char *units[] = { "mm", "cm", "ft", "m" };
    d_objUnit->setValue(4, units, 3);

    // switch whether creating a new grid or just a view
    p_createGrid = addBooleanParam("CreateGrid", "select the action");
    p_createGrid->setValue(0);

    // the parameters for the add. Parts are in ther classes - pass 'this'
    int i;
    paraSwitch("Vent", "Select a vent");

    // Case switching group start -
    char buf[64];

    for (i = 0; i < MAX_ADD_GEOM; i++)
    {
        // start new switching group
        sprintf(buf, "Vent_%d", i);
        paraCase(buf);

        const char *label[1] = { "---" };
        sprintf(buf, "Vent_%d:Name", i);
        p_name[i] = addChoiceParam(buf, "Select a vent type substructure");
        p_name[i]->setValue(1, label, 0);

        d_addGeo[i] = new AddGeom(i, this);
        paraEndCase();
    }

    paraEndSwitch();

    // no attachables yet
    for (i = 0; i < MAX_ATTACHABLES; i++)
        d_attachable[i] = NULL;
    d_attachablesInitialized = false;
}

/////////////////////////////////////////////////////////////////////////

VisitPro::~VisitPro()
{
}

/////////////////////////////////////////////////////////////////////////

//      ####    ####   #    #  #####   #    #   #####  ######
//     #    #  #    #  ##  ##  #    #  #    #     #    #
//     #       #    #  # ## #  #    #  #    #     #    #####
//     #       #    #  #    #  #####   #    #     #    #
//     #    #  #    #  #    #  #       #    #     #    #
//      ####    ####   #    #  #        ####      #    ######

int VisitPro::compute()
{
    // create the base geometry objects
    createBaseGeo();

    // map start? update attachable list now!
    if (!d_attachablesInitialized)
        updateAttachables();

    createAttachedGeo();

    // When creating a grid, we make our Star Object
    if (p_createGrid->getValue() == true)
        createStarObj();
    else
        createNoExecObj(); // empty text + NOEXEC attribute = no exec of StarCD module

    return CONTINUE_PIPELINE;
}

/////////////////////////////////////////////////////////////////////////

//     #####    ####    ####   #####  #  #    #   ####    #####
//     #    #  #    #  #         #    #  ##   #  #          #
//     #    #  #    #   ####     #    #  # #  #   ####      #
//     #####   #    #       #    #    #  #  # #       #     #
//     #       #    #  #    #    #    #  #   ##  #    #     #
//     #        ####    ####     #    #  #    #   ####      #

void VisitPro::postInst()
{
    p_createGrid->show();
    d_baseGeo->show();
    d_addPart->show();
    d_objUnit->show();
}

/////////////////////////////////////////////////////////////////////////

//     #####     ##    #####     ##    #    #
//     #    #   #  #   #    #   #  #   ##  ##
//     #    #  #    #  #    #  #    #  # ## #
//     #####   ######  #####   ######  #    #
//     #       #    #  #   #   #    #  #    #
//     #       #    #  #    #  #    #  #    #

void VisitPro::param(const char *pname)
{

    // no actions when just loading a map
    if (in_map_loading() || !strcmp(pname, "SetModuleTitle"))
        return;

    d_baseGeo->param(pname);
    d_addPart->param(pname);

    if (strstr(pname, "AddGeo") == pname)
    {
        updateAttachables();
        return;
    }

    if (strstr(pname, "Vent_") == pname
        && strstr(pname, ":Name") != NULL)
    {
        int number;
        sscanf(pname, "Vent_%d:Name", &number);
        int selection = p_name[number]->getValue();
        if (selection > 1)
            d_addGeo[number]->setAttachable(d_attachable[selection - 2]);
        else
            d_addGeo[number]->clear();
        return;
    }
}

/////////////////////////////////////////////////////////////////////////

/////// Create OBJ carrier object from a single OBJ file
coDistributedObject *VisitPro::readOBJ(const char *filename,
                                       const char *objName,
                                       int id, const char *unit)
{
    char buffer[1024];
    sprintf(buffer, "%s_%d", objName, id);

    return ReadObj::read(filename, buffer, unit);
}

////////////////////////////////////////////////////////////////////////////

void VisitPro::createBaseGeo()
{
    const char *objName = p_baseOut->getObjName();
    int numObj = d_baseGeo->numFiles();

    coDistributedObject **obj = new coDistributedObject *[numObj + 1];
    sendInfo("Transmitting base geometry");
    int i;
    for (i = 0; i < numObj; i++)
    {
        obj[i] = readOBJ(d_baseGeo->fileName(i), objName, i, d_objUnit->getActLabel());
        coModule::sendInfo("OBJ file: %s", d_baseGeo->fileName(i));
    }
    obj[i] = NULL;

    coDoSet *set = new coDoSet(objName, obj);
    p_baseOut->setCurrentObject(set);

    delete[] obj;
}

////////////////////////////////////////////////////////////////////////////

void VisitPro::updateAttachables()
{
    int i;
    for (i = 0; i < MAX_ATTACHABLES; i++)
        delete d_attachable[i];

    int numAttachables = d_addPart->numFiles();
    const char **choiceLabels = new const char *[numAttachables + 1];
    choiceLabels[0] = "---";

    for (i = 0; i < numAttachables; i++)
    {
        d_attachable[i] = new Attachable(d_addPart->fileName(i));
        choiceLabels[i + 1] = d_attachable[i]->getChoiceLabel();
    }

    // update all currently attached geoms, notify if we killed an acive one
    bool killedActive = false;
    for (i = 0; i < MAX_ADD_GEOM; i++)
    {
        // update the choice
        int oldChoice = p_name[i]->getValue();
        p_name[i]->updateValue(numAttachables + 1, choiceLabels, 1);
        int newChoice = p_name[i]->getValue();

        // we've kille a used attached element
        if (newChoice == 1)
        {
            if (oldChoice > 1)
                killedActive = true;
            d_addGeo[i]->clear();
        }
        else
            d_addGeo[i]->setAttachable(d_attachable[newChoice - 2]);
    }

    if (killedActive)
        coModule::sendWarning("Removed active attached element");

    delete[] choiceLabels;

    d_attachablesInitialized = true;
}

////////////////////////////////////////////////////////////////////////////

void VisitPro::createAttachedGeo()
{
    const char *objName = p_attachOut->getObjName();

    coDistributedObject **obj = new coDistributedObject *[MAX_ADD_GEOM + 1];
    sendInfo("Transmitting attached geometry");
    int i;
    int numElem = 0;
    for (i = 0; i < MAX_ADD_GEOM; i++)
    {
        obj[numElem] = d_addGeo[i]->getCurrentObject(objName, d_objUnit->getActLabel());
        if (obj[numElem])
            numElem++;
    }
    obj[numElem] = NULL;

    coDoSet *set = new coDoSet(objName, obj);

    coFeedback feedback("VisitProPlugin");
    feedback.addString("SET");

    int len = 0; // str len
    //int i;
    char *str;
    int numAttachables = 0;
    for (i = 0; i < MAX_ATTACHABLES; i++)
    {
        if (d_attachable[i])
        {
            // " vent_box vent_cyl"
            len += strlen(d_attachable[i]->getChoiceLabel()) + 1;
            numAttachables++;
        }
    }
    len += 100; // "VENTDIRS 2 vent_box vent_cyl"

    str = new char[len + 1];
    sprintf(str, "VENTDIRS %d", numAttachables);
    for (i = 0; i < MAX_ATTACHABLES; i++)
    {
        if (d_attachable[i])
        {
            strcat(str, " ");
            strcat(str, d_attachable[i]->getChoiceLabel());
        }
    }
    ///fprintf(stderr,"str=[%s]\n", str);
    feedback.addString(str);

    for (i = 0; i < MAX_ADD_GEOM; i++)
        feedback.addPara(p_name[i]);

    feedback.apply(set);

    p_attachOut->setCurrentObject(set);

    delete[] str;
    delete[] obj;
}

////////////////////////////////////////////////////////////////////////////

void VisitPro::createStarObj()
{

    // count number of attached elements
    int i;
    int numAttached = 0;
    for (i = 0; i < MAX_ADD_GEOM; i++)
        if (d_addGeo[i]->isAttached())
            numAttached++;

    // prepare the communication object and have a strstream pointing to it
    ostrstream starTextStr;

    // send number of attached
    starTextStr << "NUMADDPART " << numAttached << endl;

    for (i = 0; i < MAX_ADD_GEOM; i++)
    {
        if (d_addGeo[i]->isAttached())
            d_addGeo[i]->printStarObj(starTextStr);
    }
    starTextStr << "CREATOR prostar\n\0";

    // Put the object to the coDoText
    const char *starObjName = p_starOut->getObjName();
    char *StarText = starTextStr.str();
    coDoText *starObj = new coDoText(starObjName, StarText);
    sendInfo(StarText);

    p_starOut->setCurrentObject(starObj);
}

void VisitPro::createNoExecObj()
{
    const char *starObjName = p_starOut->getObjName();
    coDoText *starObj = new coDoText(starObjName, "");
    starObj->addAttribute("NOEXEC", "true");
    p_starOut->setCurrentObject(starObj);
}
