/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
#include "ReadABAQUS.h"
#include "ChoiceState.h"
#include "Node.h"
#include <util/coviseCompat.h>
// +++++++++++++++++++++++++++++++++++++++++++++++++++++
#include <odb_API.h>
//#include <odb_WipReporter.h>
//#include <wip_WIP.h>
#include <wip_WipReporter.h>
#include <nex_Exception.h>

//void stdB_initialize(int&);
//void stdB_finalize(int&);

static int called = 0;

void umkInitialize(int &active)
{
    if (called++)
        return;
    active++;

    //stdB_initialize(active);
    odb_initializeAPI();
}

void umkFinalize(int &active)
{
    if (--called)
        return;
    active--;

    odb_finalizeAPI();
    //stdB_finalize(active);
}

static int active;

void HKSFinalize()
{
    //cerr << "++++++++++++++ HKSFinalize ++++++++++++++"<<endl;
    //cerr <<"Active: "<< active<<", Called: "<<called<<endl;
    if (active)
        umkFinalize(active);
    assert(active == 0);
}

int ABQmain(int argc, char **argv);

int
main(int argc, char **argv)
{
    umkInitialize(active);
    atexit(HKSFinalize);

    int status = 0;
    try
    {
        status = ABQmain(argc, argv);
    }
    catch (const odb_Exception &odb)
    {
        cerr << "odb_Exception: ODB Application exited with error(s)." << endl;
        cerr << odb.UserReport().text() << endl;
        cerr << odb.DeveloperReport().text() << endl;
        cerr << odb.ErrMsg().text() << endl;
    }
    catch (const nex_Exception &nex)
    {
        cerr << "nex_Exception: ODB Application exited with error(s): "
             << nex.UserReport().text() << endl;
        cerr << nex.DeveloperReport().text() << endl;
    }
    catch (...)
    {
        cerr << "ODB Application exited with error(s)." << endl;
    }

    HKSFinalize();

    return status;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++

int
ABQmain(int argc, char *argv[])
{
    ReadABAQUS *application = new ReadABAQUS(argc, argv);

    application->start(argc, argv);

    return 0;
}

ReadABAQUS::ReadABAQUS(int argc, char *argv[])
    : coModule(argc, argv, "Read odb ABAQUS-files")
    , p_choices_(NULL)
    , _callRebuild(false)
{
    // Parameters
    p_odb_ = addFileBrowserParam("ODB_file", "Output database");
    p_odb_->setValue("data/", "*.odb");

    p_steps_ = addInt32VectorParam("timeSteps", "minimum, maximum, interval");
    p_steps_->setValue(1, 1, 1);

    p_frames_ = addInt32Param("frames", "frame interval from last");
    p_frames_->setValue(1);

    p_scale_ = addFloatParam("ScaleGridDisplacement", "scale grid displacement");
    p_scale_->setValue(1.0);

    p_doUpdate_ = addBooleanParam("update_ODB", "name of updated output database");
    p_doUpdate_->setValue(1);

    p_updatedOdb_ = addFileBrowserParam("updated_ODB_file", "updated output database");
    p_updatedOdb_->setValue("/var/tmp/updated_foo.odb", "*.odb");

    int dataset;
    for (dataset = 0; dataset < NUM_OUTPUTS; ++dataset)
    {
        char buf[32];
        sprintf(buf, "%d", dataset);
        string stringdataset(buf);

        string var = "Variable_";
        string var_descr = "Variable ";
        string component = "Component_";
        string component_descr = "Component ";
        string invariant = "Invariant_";
        string invariant_descr = "Invariant ";
        string location = "Location_";
        string location_descr = "Location ";
        string SectPoint = "SectionPoint_";
        string SectPoint_descr = "SectionPoint ";
        string thisSectPoint = "OnlyThisSectionPoint_";
        string thisSectPoint_descr = "OnlyThisSectionPoint ";
        string localCS = "DataInLocalCS_";
        string localCS_descr = "DataInLocalCS ";
        var += stringdataset;
        var_descr += stringdataset;
        component += stringdataset;
        component_descr += stringdataset;
        invariant += stringdataset;
        invariant_descr += stringdataset;
        location += stringdataset;
        location_descr += stringdataset;
        SectPoint += stringdataset;
        SectPoint_descr += stringdataset;
        thisSectPoint += stringdataset;
        thisSectPoint_descr += stringdataset;
        localCS += stringdataset;
        localCS_descr += stringdataset;

        const char *None[] = { "None" };
        p_variable_[dataset] = addChoiceParam(var.c_str(), var_descr.c_str());
        p_variable_[dataset]->setValue(1, None, 0);

        p_components_[dataset] = addChoiceParam(component.c_str(),
                                                component_descr.c_str());
        p_components_[dataset]->setValue(1, None, 0);

        p_invariants_[dataset] = addChoiceParam(invariant.c_str(),
                                                invariant_descr.c_str());
        p_invariants_[dataset]->setValue(1, None, 0);

        p_locations_[dataset] = addChoiceParam(location.c_str(),
                                               location_descr.c_str());
        p_locations_[dataset]->setValue(1, None, 0);

        p_sectionPoints_[dataset] = addChoiceParam(SectPoint.c_str(),
                                                   SectPoint_descr.c_str());
        p_sectionPoints_[dataset]->setValue(1, None, 0);

        p_onlyThisSectionPoint_[dataset] = addBooleanParam(thisSectPoint.c_str(),
                                                           thisSectPoint_descr.c_str());
        p_onlyThisSectionPoint_[dataset]->setValue(0);

        p_LocalCS_[dataset] = addBooleanParam(localCS.c_str(),
                                              localCS_descr.c_str());
        p_LocalCS_[dataset]->setValue(1);
    }

    p_ConjugateData_ = addBooleanParam("ConjugateData",
                                       "Conjugate data in case of frequency frames");
    p_ConjugateData_->setValue(0);

    // Ports
    for (dataset = 0; dataset < NUM_OUTPUTS; ++dataset)
    {
        char buf[32];
        sprintf(buf, "%d", dataset);
        string stringdataset(buf);

        string mesh = "Mesh_";
        string mesh_descr = "Mesh ";
        mesh += stringdataset;
        mesh_descr += stringdataset;
        p_mesh_[dataset] = addOutputPort(mesh.c_str(), "UnstructuredGrid",
                                         mesh_descr.c_str());

        string data = "Data_";
        string data_descr = "Data ";
        data += stringdataset;
        data_descr += stringdataset;
        p_data_[dataset] = addOutputPort(data.c_str(),
                                         "Float|Vec3|Tensor|Mat3",
                                         data_descr.c_str());

        string part_indices = "PartIndices_";
        string part_indices_descr = "PartIndices ";
        part_indices += stringdataset;
        part_indices_descr += stringdataset;
        p_part_indices_[dataset] = addOutputPort(part_indices.c_str(),
                                                 "IntArr",
                                                 part_indices_descr.c_str());
    }
    p_parts_ = addOutputPort("PartsDictionary", "Text", "Dictionary of part names");
    p_displacements = addOutputPort("Displacements", "Vec3", "Displacements");
}

void
ReadABAQUS::postInst()
{
    p_ConjugateData_->disable();
}

ReadABAQUS::~ReadABAQUS()
{
}

void
ReadABAQUS::param(const char *paramName, bool inMapLoading)
{
    // cerr << "p1!" << endl;
    if (strcmp(p_updatedOdb_->getName(), paramName) == 0)
    {
        manualUpdateName_ = p_updatedOdb_->getValue();
    }

    if (strcmp(p_odb_->getName(), paramName) == 0)
    {
        // read possible choices
        sendInfo("Reading info in ODB file. Please wait...");
        // cerr << "pc3!" << endl;

        ChoiceState *p_choices = 0;
        try
        {
            p_choices = new ChoiceState(NUM_OUTPUTS, p_odb_->getValue());
        }
        catch (const nex_Exception &nex)
        {
            // cerr << nex.UserReport() << endl;
            cerr << "nex_Exception: ODB Application exited with error(s): "
                 << nex.UserReport().text() << endl;
            sendError(nex.UserReport().text());
            return;
        }
        catch (...)
        {
            cerr << "ODB Application exited with error(s)" << endl;
            return;
        }
        // cerr << "pc4!" << endl;

        // create name of the updated database in case an update is mandantory
        string fullOdbName(p_odb_->getValue());
        char pat = '/';
        string::iterator it = fullOdbName.begin();
        string::iterator found = fullOdbName.begin();
        while (it != fullOdbName.end())
        {
            if (*it == pat)
            {
                found = it;
                found++;
            }
            it++;
        }
        string odbName(found, fullOdbName.end());

        string proposedUpdateName("/var/tmp/");
        proposedUpdateName += "updated_";
        proposedUpdateName += odbName;

        //p_updatedOdb_->setValue( proposedUpdateName.c_str(), "*.odb" );

        sendInfo("Info read from ODB file.");
        string mssgOut;
        bool diagnose = p_choices->OK(mssgOut);
        // test correctness
        if (diagnose)
        {
            sendInfo(mssgOut.c_str());
        }
        else if (p_choices->updateRequired())
        {
            // in case we find a previous version of the ABAQUS database
            // Covise::sendInfo("! The database is from a previous version of ABAQUS");
            // Covise::sendInfo("! update required - try to do the update...");

            Covise::sendInfo("The output database was created with a previous version of ABAQUS!");
            Covise::sendInfo("ODB file requires to be updated; updating the database...");

            // if the name was already set..
            if (!manualUpdateName_.empty())
                proposedUpdateName = manualUpdateName_;

            // cerr << "pc5!" << endl;
            // ..otherwise we use the name of the database in /tmp and the prefix 'update_'
            if (p_doUpdate_->getValue())
            {
                p_choices->doUpdate(proposedUpdateName);
                Covise::sendInfo("ODB file update finished.");
                string msg("The updated database can be found under: ");
                msg += proposedUpdateName;
                Covise::sendInfo(msg.c_str());
                // now we have to reconstruct the state of p_choices and go ahead
                delete p_choices;
                // cerr << "pc6!" << endl;
                p_choices = new ChoiceState(NUM_OUTPUTS, proposedUpdateName.c_str());
                //cerr << "pc7!" << endl;
                diagnose = p_choices->OK(mssgOut);
            }
            else
            {
                Covise::sendInfo("The output database could not be updated.");
                Covise::sendInfo("Please check the parameter update_ODB.");
                return;
            }
        }
        else
        {
            sendError(mssgOut.c_str());
            // in this case p_choices points to a dummy
            // object and we try to keep previous value
            if (p_choices_ && p_choices_->OK())
            {
                delete p_choices;
                // reset value
                p_odb_->setValue(p_choices_->Path().c_str(), "*.odb");
            }
            else // previous value was anyway not OK
            {
                delete p_choices_;
                p_choices_ = p_choices;
            }
            return;
        }
        if (inMapLoading)
        {
            p_choices_ = p_choices;
            _callRebuild = true;
            return;
        }

        // cerr << "p3!" << endl;
        // the diagnose of p_choices was successful,
        // we try to reproduce the same visualisation
        // as far as possible
        bool exactly_equal = p_choices->ReproduceVisualisation(p_choices_);
        delete p_choices_;
        p_choices_ = p_choices;
        if (p_choices_->harmonic())
        {
            p_ConjugateData_->enable();
        }
        else
        {
            p_ConjugateData_->disable();
        }
        // cerr << "p4!" << endl;
        if (!exactly_equal)
        {
            int dataset;
            for (dataset = 0; dataset < NUM_OUTPUTS; ++dataset)
            {
                p_choices_->AdjustParams(dataset, p_variable_[dataset],
                                         p_components_[dataset],
                                         p_invariants_[dataset],
                                         p_locations_[dataset],
                                         p_sectionPoints_[dataset]);
            }
        }
        p_steps_->setValue(1, p_choices_->no_steps(), 1);
    }
    else if (!inMapLoading && p_choices_ && p_choices_->OK())
    {
        // cerr << "p5!" << endl;
        int dataset = -1;
        // get the param that changed
        ChoiceType choiceType = ParamChanged(paramName, dataset);
        switch (choiceType)
        {
        case DO_NOT_KNOW:
            break;
        case VARIABLE:
            p_choices_->SetVariable(dataset, p_variable_[dataset],
                                    p_components_[dataset],
                                    p_invariants_[dataset],
                                    p_locations_[dataset],
                                    p_sectionPoints_[dataset]);
            break;
        case COMPONENT:
            p_invariants_[dataset]->setValue(0);
            p_choices_->SetComponentOrInvariant(dataset, p_components_[dataset],
                                                p_invariants_[dataset]);
            break;
        case INVARIANT:
            p_components_[dataset]->setValue(0);
            p_choices_->SetComponentOrInvariant(dataset, p_components_[dataset],
                                                p_invariants_[dataset]);
            break;
        case LOCATION:
            p_choices_->SetLocation(dataset, p_locations_[dataset],
                                    p_sectionPoints_[dataset]);
            break;
        case SECTION_POINT:
            p_choices_->SetSectionPoint(dataset, p_sectionPoints_[dataset]);
            break;
        }
    }
}

ReadABAQUS::ChoiceType
ReadABAQUS::ParamChanged(const char *paramName, int &dataset)
{
    int i;
    for (i = 0; i < NUM_OUTPUTS; ++i)
    {
        if (strcmp(p_variable_[i]->getName(), paramName) == 0)
        {
            dataset = i;
            return VARIABLE;
        }
        else if (strcmp(p_components_[i]->getName(), paramName) == 0)
        {
            dataset = i;
            return COMPONENT;
        }
        else if (strcmp(p_invariants_[i]->getName(), paramName) == 0)
        {
            dataset = i;
            return INVARIANT;
        }
        else if (strcmp(p_locations_[i]->getName(), paramName) == 0)
        {
            dataset = i;
            return LOCATION;
        }
        else if (strcmp(p_sectionPoints_[i]->getName(), paramName) == 0)
        {
            dataset = i;
            return SECTION_POINT;
        }
    }
    return DO_NOT_KNOW;
}

int
ReadABAQUS::compute(const char *)
{
    if (p_choices_ == NULL || !p_choices_->OK())
    {
        sendError("Could not read output database");
        return STOP_PIPELINE;
    }
    // cerr << "1!" << endl;

    ///Check if there is a lock file
    string fileName(p_choices_->Path());
    int pos = fileName.find(".odb");
    if (pos == -1)
    {
        fileName.append(".lck");
    }
    else
    {
        fileName.replace(pos, 4, ".lck");
    }
    FILE *lockFile = fopen(fileName.c_str(), "r");
    if (lockFile != NULL)
    {
        fclose(lockFile);
        sendError("Could not open database since there is a lockfile.");
        return STOP_PIPELINE;
    }
    // cerr << "2!" << endl;

    if (_callRebuild)
    {
        // get here the choice lists
        int set;
        for (set = 0; set < NUM_OUTPUTS; ++set)
        {
            p_choices_->RebuildVisualisation(set, p_variable_[set],
                                             p_components_[set],
                                             p_invariants_[set],
                                             p_locations_[set],
                                             p_sectionPoints_[set]);
        }
        string mssgOut;
        bool diagnose = p_choices_->OK(mssgOut);
        if (!diagnose)
        {
            sendError(mssgOut.c_str());
            return STOP_PIPELINE;
        }
        _callRebuild = false;
    }
    // cerr << "3!" << endl;

    Node::DISP_SCALE = p_scale_->getValue();
    p_choices_->PartDictionary(p_parts_);
    int dataset;
    for (dataset = 0; dataset < NUM_OUTPUTS; ++dataset)
    {
        p_choices_->DataSet(dataset,
                            p_steps_->getValue(0),
                            p_steps_->getValue(1),
                            p_steps_->getValue(2),
                            p_frames_->getValue(),
                            p_onlyThisSectionPoint_[dataset]->getValue(),
                            p_LocalCS_[dataset]->getValue(),
                            p_mesh_[dataset],
                            p_data_[dataset],
                            p_part_indices_[dataset],
                            p_displacements,
                            p_ConjugateData_->getValue());
    }
    return CONTINUE_PIPELINE;
}
