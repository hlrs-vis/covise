/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ChoiceState.h"
#include <do/coDoText.h>
#include <do/coDoSet.h>
#include <api/coModule.h>
#include "ResultMesh.h"
#include "VectorData.h"
#include "Data.h"

#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

ChoiceState::ChoiceState(int no_sets, const char *path)
    : _path(path)
    , _no_steps(0)
    , _OK(false)
    , _choice_sets(NULL)
    , _no_sets(no_sets)
    , _harmonic(false)
    , odbUpdateRequired_(false)
{
    struct stat statbuf;
#ifndef _WIN32
    if (stat(path, &statbuf) != 0)
    {
        _mssgOut = strerror(errno);
        perror("ReadABAQUS encountered an error while inspecting the odb path");
        return;
    }
#endif

    if (isUpgradeRequiredForOdb(path))
    {
        _mssgOut = "An update is required for database ";
        _mssgOut += path;
        odbUpdateRequired_ = true;
        return;
    }
    // cerr << "AAAA" << endl;
    odb_Odb &odb = openOdb(path);
    // cerr << "BBBBB" << endl;
    _mssgOut = "ODB: ";
    _mssgOut += odb.name().CStr();
    // cerr << "1BBBBB" << endl;
    _mssgOut += ", analysisTitle: ";
    _mssgOut += odb.analysisTitle().CStr();
    // cerr << "2BBBBB" << endl;
    _mssgOut += ", description: ";
    _mssgOut += odb.description().CStr();
    // cerr << "AAAA" << endl;
    odb_StepRepository sCon = odb.steps();
    // we also accumulate number of steps
    _no_steps = sCon.size();

    // Abaqus version of the ODB file
    Covise::sendInfo(odb.jobData().version().CStr());

    // now we look for available fields in all steps and frames
    odb_StepRepositoryIT sIter(sCon);
    int step_label = 1;
    for (sIter.first(); !sIter.isDone(); sIter.next(), ++step_label)
    {
        AccumulateFieldsInStep(sCon[sIter.currentKey()], step_label);
    }
    _OK = true;
    _choice_sets = new OutputChoice[no_sets];
    int set;
    for (set = 0; set < no_sets; ++set)
    {
        _choice_sets[set]._fieldLabel = 0;
        _choice_sets[set]._component = 0;
        _choice_sets[set]._invariant = 0;
        _choice_sets[set]._location = 0;
        _choice_sets[set]._secPoint = 0;
    }
    // create _instance_dictionary
    odb_InstanceRepository iCon = odb.rootAssembly().instances();
    int no_instances = iCon.size();
    odb_InstanceRepositoryIT iter(iCon);
    int instance_tag = 0;
    for (iter.first(); !iter.isDone(); iter.next(), ++instance_tag)
    {
        // instance
        odb_Instance inst = iter.currentValue();
        _instance_dictionary.insert(pair<string, int>(inst.name().CStr(),
                                                      instance_tag));
        _instance_meshes.push_back(InstanceMesh());
        InstanceMesh &ThisInstanceMesh = *(_instance_meshes.rbegin());
        // nodes
        odb_SequenceNode nC = inst.nodes();
        int numN = nC.size();
        int node;
        for (node = 0; node < numN; node++)
        {
            odb_Node n = nC.node(node);
            odb_SequenceFloat coor = n.coordinate();
            int no_coords = coor.size();
            if (no_coords > 3)
            {
                no_coords = 3;
            }
            int coord;
            vector<float> coordinates;
            for (coord = 0; coord < no_coords; ++coord)
            {
                coordinates.push_back(float(coor.constGet(coord)));
            }
            for (; coord < 3; ++coord)
            {
                coordinates.push_back(0.0);
            }
            ThisInstanceMesh.AddNode(n.label(), coordinates);
        }
        // elements
        odb_SequenceElement elements = inst.elements();
        odb_SequenceElement eC = inst.elements();
        int numE = eC.size();
        int elem;
        for (elem = 0; elem < numE; elem++)
        {
            odb_Element e = eC.element(elem);
            odb_SequenceInt conn = e.connectivity();
            vector<int> connectivity;
            int no_vert = conn.size();
            int vert;
            for (vert = 0; vert < no_vert; ++vert)
            {
                connectivity.push_back(conn.constGet(vert));
            }
            ThisInstanceMesh.AddElement(e.label(), e.type().CStr(), connectivity);
        }
        // surfaces
        /*
            odb_SetRepository& setRepSurf = inst.surfaces();
            odb_SetRepositoryIT setSurfIT(setRepSurf);
            for(setSurfIT.first(); !setSurfIT.isDone(); setSurfIT.next()) {
               odb_Set setSurf = setRepSurf[setSurfIT.currentKey()];
               cerr << "A surface set detected"<<endl;
               const odb_SequenceElement& elements = setSurf.elements();
               const odb_SequenceElementFace& faces = setSurf.faces();
               cerr << "A surface with "<<elements.size() <<" elements"<<endl;
               int elem;
               for (elem=0; elem<elements.size(); elem++) {
      odb_Element e = elements.element(elem);
      cerr <<".... element "<<e.type().CStr()<<endl;
      }
      cerr << "A surface with "<<faces.size() <<" faces"<<endl;
      }
      */
        ThisInstanceMesh.OrderNodesAndElements();
    }
    odb.close();
}

void
ChoiceState::doUpdate(const string &updPath)
{
    if (odbUpdateRequired_)
    {
        upgradeOdb(_path.c_str(), updPath.c_str());
    }
}

bool
ChoiceState::updateRequired() const
{
    return odbUpdateRequired_;
}

void
ChoiceState::PartDictionary(coOutputPort *p_partdict) const
{
    string text;
    map<string, int>::const_iterator it;
    for (it = _instance_dictionary.begin(); it != _instance_dictionary.end(); ++it)
    {
        char buf[32];
        sprintf(buf, "%d: ", it->second);
        text += buf;
        text += it->first.c_str();
        text += '\n';
    }
    coDoText *coText = new coDoText(p_partdict->getObjName(), text.c_str());
    p_partdict->setCurrentObject(coText);
}

ChoiceState::~ChoiceState()
{
    delete[] _choice_sets;
}

void
ChoiceState::AccumulateFieldsInStep(odb_Step &step, int step_label)
{
    odb_SequenceFrame &fCon = step.frames();
    int numFrames = fCon.size();
    int num_modal = 0;
    int num_time = 0;
    int num_harmonic = 0;
    int f;
    for (f = 0; f < numFrames; ++f)
    {
        odb_Frame frame = fCon.get(f);
        switch (frame.domain())
        {
        case odb_Enum::MODAL:
            ++num_modal;
            break;
        case odb_Enum::FREQUENCY:
            ++num_harmonic;
            break;
        case odb_Enum::TIME:
            ++num_time;
            break;
        }
        AccumulateFieldsInFrame(frame);
    }
    // inform about this time step
    string mssg("Step ");
    char buf[256];
    sprintf(buf, "%d has ", step_label);
    mssg += buf;
    if (num_modal > 0)
    {
        sprintf(buf, "%d frames in the modal domain", num_modal);
        mssg += buf;
    }
    if (num_harmonic > 0)
    {
        if (num_modal > 0)
        {
            mssg += ", ";
        }
        else
        {
            mssg += " ";
        }
        sprintf(buf, "%d frames in the harmonic domain", num_harmonic);
        mssg += buf;
        _harmonic = true;
    }
    if (num_time > 0)
    {
        if (num_modal > 0 || num_harmonic > 0)
        {
            mssg += " and ";
        }
        else
        {
            mssg += " ";
        }
        sprintf(buf, "%d frames in the time domain", num_time);
        mssg += buf;
    }
    if (num_time == 0 && num_modal == 0 && num_harmonic == 0)
    {
        mssg += " no frames";
    }
    Covise::sendInfo(mssg.c_str());
}

void
ChoiceState::AccumulateFieldsInFrame(odb_Frame &frame)
{
    odb_FieldOutputRepository &fieldCon = frame.fieldOutputs();
    odb_FieldOutputRepositoryIT fieldConIT(fieldCon);
    for (fieldConIT.first(); !fieldConIT.isDone(); fieldConIT.next())
    {
        odb_FieldOutput field = fieldCon[fieldConIT.currentKey()];
        _fieldContainer.AccumulateField(field);
    }
}

bool
ChoiceState::OK() const
{
    return _OK;
}

bool
ChoiceState::OK(string &mssgOut) const
{
    mssgOut = _mssgOut;
    return _OK;
}

string
ChoiceState::Path() const
{
    return _path;
}

int
ChoiceState::no_steps() const
{
    return _no_steps;
}

void
ChoiceState::SetSectionPoint(int dataset,
                             coChoiceParam *p_sectionPoints)
{
    _choice_sets[dataset]._secPoint = p_sectionPoints->getValue();
}

void
ChoiceState::SetLocation(int dataset,
                         coChoiceParam *p_locations,
                         coChoiceParam *p_sectionPoints)
{
    _choice_sets[dataset]._location = p_locations->getValue();
    // we have to change the choices of p_sectionPoints
    _choice_sets[dataset]._secPoint = 0;
    AdjustParams(dataset, NULL, NULL, NULL, NULL, p_sectionPoints);
}

void
ChoiceState::SetComponentOrInvariant(int dataset,
                                     coChoiceParam *p_components,
                                     coChoiceParam *p_invariants)
{
    _choice_sets[dataset]._component = p_components->getValue();
    _choice_sets[dataset]._invariant = p_invariants->getValue();
}

void
ChoiceState::SetVariable(int dataset,
                         coChoiceParam *p_fields,
                         coChoiceParam *p_components,
                         coChoiceParam *p_invariants,
                         coChoiceParam *p_locations,
                         coChoiceParam *p_sectionPoints)
{
    _choice_sets[dataset]._fieldLabel = p_fields->getValue();
    _choice_sets[dataset]._component = 0;
    _choice_sets[dataset]._invariant = 0;
    _choice_sets[dataset]._location = 0;
    _choice_sets[dataset]._secPoint = 0;
    AdjustParams(dataset,
                 NULL, p_components, p_invariants, p_locations, p_sectionPoints);
}

bool
ChoiceState::ReproduceVisualisation(const ChoiceState *old_state)
{
    if (old_state == NULL || !old_state->OK())
    {
        return false;
    }
    int set;
    for (set = 0; set < _no_sets; ++set)
    {
        // find old choice lists
        const OutputChoice &old_choice = old_state->_choice_sets[set];

        vector<string> fieldLabelsOld;
        vector<string> componentsOld;
        vector<string> invariantsOld;
        vector<string> locationsOld;
        vector<string> sectionPointsOld;

        old_state->_fieldContainer.GetLists(old_choice,
                                            fieldLabelsOld, componentsOld, invariantsOld,
                                            locationsOld, sectionPointsOld);

        _fieldContainer.ReproduceVisualisation(_choice_sets[set],
                                               fieldLabelsOld[old_choice._fieldLabel],
                                               componentsOld[old_choice._component],
                                               invariantsOld[old_choice._invariant],
                                               locationsOld[old_choice._location],
                                               sectionPointsOld[old_choice._secPoint]);
    }
    return false;
}

static string
MapToSpace(const char *title)
{
    string ret;
    while (*title != '\0')
    {
        if (*title != '\177')
        {
            ret += *title;
        }
        else
        {
            ret += ' ';
        }
        ++title;
    }
    return ret;
}

static void Print(const vector<string> &cont)
{
    int i;
    for (i = 0; i < cont.size(); ++i)
    {
        cerr << cont[i].c_str() << endl;
    }
}

void
ChoiceState::RebuildVisualisation(int dataset,
                                  const coChoiceParam *p_fields,
                                  const coChoiceParam *p_components,
                                  const coChoiceParam *p_invariants,
                                  const coChoiceParam *p_locations,
                                  const coChoiceParam *p_sectionPoints)
{
    /*  vector<string> fields;
   vector<string> components;
   vector<string> invariants;
   vector<string> locations;
   vector<string> sectionPoints;*/
    OutputChoice testChoices;
    testChoices._fieldLabel = p_fields->getValue();
    testChoices._component = p_components->getValue();
    testChoices._invariant = p_invariants->getValue();
    testChoices._location = p_locations->getValue();
    testChoices._secPoint = p_sectionPoints->getValue();
    /*  int i;
   for(i=0;i<p_fields->getNumChoices();++i)
   {
      fields.push_back(MapToSpace(p_fields->getLabel(i)));
   }
   for(i=0;i<p_components->getNumChoices();++i)
   {
      components.push_back(MapToSpace(p_components->getLabel(i)));
   }
   for(i=0;i<p_invariants->getNumChoices();++i)
   {
      invariants.push_back(MapToSpace(p_invariants->getLabel(i)));
   }
   for(i=0;i<p_locations->getNumChoices();++i)
   {
      locations.push_back(MapToSpace(p_locations->getLabel(i)));
   }
   for(i=0;i<p_sectionPoints->getNumChoices();++i)
   {
      sectionPoints.push_back(MapToSpace(p_sectionPoints->getLabel(i)));
   }
#ifdef _READ_ABAQUS_VERBOSE_
   cerr << "***********************************"<<endl;
   Print(fields);
   Print(components);
   Print(invariants);
   Print(locations);
   Print(sectionPoints);
   cerr << "***********************************"<<endl;
#endif
   vector<string> fieldsFile;
   vector<string> componentsFile;
   vector<string> invariantsFile;
   vector<string> locationsFile;
   vector<string> sectionPointsFile;
   _fieldContainer.GetLists(testChoices,fieldsFile,componentsFile,
      invariantsFile,locationsFile,sectionPointsFile);
#ifdef _READ_ABAQUS_VERBOSE_
   cerr <<"+++++++++++++++++++++++++++++++++++++"<<endl;
   Print(fieldsFile);
   Print(componentsFile);
   Print(invariantsFile);
   Print(locationsFile);
   Print(sectionPointsFile);
   cerr <<"+++++++++++++++++++++++++++++++++++++"<<endl;
#endif
   if(fields != fieldsFile || components != componentsFile ||
      invariants != invariantsFile || locations != locationsFile ||
      sectionPoints != sectionPointsFile)
   {
      _OK = false;
      _mssgOut = "The set of available fields and variables in the odb file probably changed since net file creation. Please, reload odb file";
      return;
   }*/
    // the choice lables are not set during map load and first execution, only the index of the selected item, thus we can't check and assume it is OK.
    _OK = true;
    _choice_sets[dataset] = testChoices;
}

static const char **
CharArrays(const vector<string> &labels)
{
    const char **ret = new const char *[labels.size()];
    int i;
    for (i = 0; i < labels.size(); ++i)
    {
        ret[i] = labels[i].c_str();
    }
    return ret;
}

void
ChoiceState::AdjustParams(int dataset,
                          coChoiceParam *p_variable_,
                          coChoiceParam *p_components_,
                          coChoiceParam *p_invariants_,
                          coChoiceParam *p_locations_,
                          coChoiceParam *p_sectionPoints_) const
{
    vector<string> fieldLabels;
    vector<string> components;
    vector<string> invariants;
    vector<string> locations;
    vector<string> sectionPoints;

    _fieldContainer.GetLists(_choice_sets[dataset],
                             fieldLabels, components, invariants,
                             locations, sectionPoints);

    // now set params accordingly
    // const char **var_choice = new const char *[fieldLabels.size()];
    if (p_variable_)
    {
        const char **var_choice = CharArrays(fieldLabels);
        p_variable_->setValue(fieldLabels.size(), var_choice,
                              _choice_sets[dataset]._fieldLabel);
        delete[] var_choice;
    }

    if (p_components_)
    {
        const char **comp_choice = CharArrays(components);
        p_components_->setValue(components.size(), comp_choice,
                                _choice_sets[dataset]._component);
        delete[] comp_choice;
    }

    if (p_invariants_)
    {
        const char **inv_choice = CharArrays(invariants);
        p_invariants_->setValue(invariants.size(), inv_choice,
                                _choice_sets[dataset]._invariant);
        delete[] inv_choice;
    }

    if (p_locations_)
    {
        const char **loc_choice = CharArrays(locations);
        p_locations_->setValue(locations.size(), loc_choice,
                               _choice_sets[dataset]._location);
        delete[] loc_choice;
    }

    if (p_sectionPoints_)
    {
        const char **sp_choice = CharArrays(sectionPoints);
        p_sectionPoints_->setValue(sectionPoints.size(), sp_choice,
                                   _choice_sets[dataset]._secPoint);
        delete[] sp_choice;
    }
}

covise::coDoSet *DynamicSet(string name, vector<coDistributedObject *> &objs)
{
    int no_timesteps = objs.size();
    coDistributedObject **setList = new coDistributedObject *[no_timesteps + 1];
    setList[no_timesteps] = NULL;
    int i;
    for (i = 0; i < no_timesteps; ++i)
    {
        setList[i] = objs[i];
    }
    coDoSet *ret = new coDoSet(name.c_str(), setList);
    for (i = 0; i < no_timesteps; ++i)
    {
        delete setList[i];
    }
    delete[] setList;
    char buf[64];
    sprintf(buf, "1_%d", no_timesteps);
    ret->addAttribute("TIMESTEP", buf);
    return ret;
}

void
ChoiceState::DataSet(int dataset, int minStep, int maxStep, int jumpSteps,
                     int jumpFrames,
                     bool onlyThisSection,
                     bool localCS,
                     coOutputPort *p_mesh,
                     coOutputPort *p_data,
                     coOutputPort *p_part_indices,
                     coOutputPort *p_displacements,
                     bool conjugate)
{
    float accumulateTime = 0.0;
    if (getenv("READ_ABAQUS_ACCUMULATE"))
    {
        accumulateTime = 1.0;
    }
    // output buckets
    vector<coDistributedObject *> mesh;
    vector<coDistributedObject *> data;
    vector<coDistributedObject *> part_indices;
    vector<coDistributedObject *> disp;
    // get the task strings
    string fieldLabel;
    string component;
    string invariant;
    string location;
    string sectionPoint;
    {
        vector<string> fieldLabels;
        vector<string> components;
        vector<string> invariants;
        vector<string> locations;
        vector<string> sectionPoints;
        _fieldContainer.GetLists(_choice_sets[dataset], fieldLabels,
                                 components, invariants, locations, sectionPoints);
        fieldLabel = fieldLabels[_choice_sets[dataset]._fieldLabel];
        component = components[_choice_sets[dataset]._component];
        invariant = invariants[_choice_sets[dataset]._invariant];
        location = locations[_choice_sets[dataset]._location];
        sectionPoint = sectionPoints[_choice_sets[dataset]._secPoint];
    }
    Data::SPECIES = fieldLabel;
    odb_Odb &odb = openOdb(_path.c_str());
    odb_StepRepository sCon = odb.steps();
    odb_StepRepositoryIT sIter(sCon);
    int step = 1;
    // loop over steps
    for (sIter.first(); !sIter.isDone() && step <= maxStep; sIter.next(), ++step)
    {
        if (step < minStep || (step - minStep) % jumpSteps != 0)
        {
            continue;
        }
        odb_SequenceFrame &fCon = sCon[sIter.currentKey()].frames();
        int numFrames = fCon.size();
        int f = (numFrames - 1) % jumpFrames;
        // loop over frames
        for (f = 0; f < numFrames; f += jumpFrames)
        {
            odb_Frame frame = fCon.get(f);
            odb_FieldOutputRepository &fieldCon = frame.fieldOutputs();
            odb_FieldOutputRepositoryIT fieldConIT(fieldCon);
            bool try_conjugate(false);
            char buf[256];
            buf[0] = '\0';
            switch (frame.domain())
            {
            case odb_Enum::MODAL:
                if (sCon[sIter.currentKey()].procedure() == "FREQUENCY")
                {
                    sprintf(buf, "%g", frame.frequency());
                }
                else
                {
                    sprintf(buf, "%d", frame.mode());
                }
                break;
            case odb_Enum::FREQUENCY:
                sprintf(buf, "%g", frame.frequency());
                try_conjugate = conjugate;
                break;
            case odb_Enum::TIME:
                sprintf(buf, "%g", frame.frameValue() + accumulateTime * sCon[sIter.currentKey()].totalTime());
                break;
            }
            Data::REALTIME = buf;
            // for each frame we have to copy _instance_meshes, because
            // an InstanceMesh may be altered in the process...
            vector<InstanceMesh> local_instance_meshes(_instance_meshes);
            for (fieldConIT.first(); !fieldConIT.isDone(); fieldConIT.next()) // remove unused elements first before setting node variables
            {
                odb_FieldOutput field = fieldCon[fieldConIT.currentKey()];
                // test whether this is the field we are interested in
                // we have to test with the schema "<name>: <description>"
                string field_name = field.name().CStr();
                field_name += ": ";
                field_name += field.description().CStr();
                // this might be the displacement field...
                if (field.name() == "STATUS")
                {
                    ReadStatus(field, local_instance_meshes);
                }
            }
            for (fieldConIT.first(); !fieldConIT.isDone(); fieldConIT.next())
            {
                odb_FieldOutput field = fieldCon[fieldConIT.currentKey()];
                // test whether this is the field we are interested in
                // we have to test with the schema "<name>: <description>"
                string field_name = field.name().CStr();
                field_name += ": ";
                field_name += field.description().CStr();
                // this might be the displacement field...
                if (field.name() == "U")
                {
                    ReadDisplacements(field, local_instance_meshes);
                }
                if (fieldLabel != field_name)
                {
                    continue; // this is not the field we want to visualise
                }
                ReadField(field, onlyThisSection, localCS,
                          component, invariant, location, sectionPoint,
                          local_instance_meshes, try_conjugate);
            }
            // use local_instance_meshes to produce covise_meshes...
            int instance_num;
            vector<ResultMesh *> results;
            vector<int> instance_labels;
            for (instance_num = 0; instance_num < local_instance_meshes.size();
                 ++instance_num)
            {
                if (!ShowInstance(instance_num))
                {
                    continue;
                }
                results.push_back(local_instance_meshes[instance_num].Result());
                instance_labels.push_back(instance_num);
            }
            // covise objects for this time
            int time = mesh.size();
            string name_mesh = p_mesh->getObjName();
            string data_mesh = p_data->getObjName();
            string name_part_indices = p_part_indices->getObjName();
            string name_disp = p_displacements->getObjName();
            sprintf(buf, "_%d", time);
            name_mesh += buf;
            data_mesh += buf;
            name_part_indices += buf;
            name_disp += buf;
            // gather objects for this time
            ResultMesh::GetObjects(results, mesh, data, part_indices, disp,
                                   name_mesh, data_mesh, name_part_indices, name_disp,
                                   instance_labels);

            int res_mesh;
            for (res_mesh = 0; res_mesh < results.size(); ++res_mesh)
            {
                delete results[res_mesh];
            }
        }
    }
    // use mesh, data and part_indices
    // to produce objects with time steps
    p_mesh->setCurrentObject(DynamicSet(p_mesh->getObjName(), mesh));
    p_data->setCurrentObject(DynamicSet(p_data->getObjName(), data));
    p_part_indices->setCurrentObject(DynamicSet(p_part_indices->getObjName(), part_indices));
    p_displacements->setCurrentObject(DynamicSet(p_displacements->getObjName(), disp));
    odb.close();
}

bool
ChoiceState::CheckLocation(odb_FieldOutput &field,
                           bool onlyThisSection,
                           const string &location) const
{
    odb_SequenceFieldLocation flCon = field.locations();
    int numLoc = flCon.size();
    int loc;
    for (loc = 0; loc < numLoc; ++loc)
    {
        const odb_FieldLocation &LocObj = flCon.constGet(loc);
        if (location == Location::ResultPositionEnumToString(LocObj.position()))
        {
            break;
        }
    }
    if (loc == numLoc)
    {
        return false;
    }
    return true;
}

void
ChoiceState::ReadField(odb_FieldOutput &field,
                       bool onlyThisSection,
                       bool localCS,
                       const string &component,
                       const string &invariant,
                       const string &location,
                       const string &sectionPoint,
                       vector<InstanceMesh> &local_instance_meshes,
                       bool conjugate)
{
    odb_Enum::odb_DataTypeEnum type = field.type();
    Data::TYPE = Data::UNDEFINED_TYPE;
    switch (type)
    {
    case odb_Enum::SCALAR:
        Data::TYPE = Data::SCALAR;
        break;
    case odb_Enum::VECTOR:
        Data::TYPE = Data::VECTOR;
        break;
    case odb_Enum::TENSOR_3D_FULL:
    case odb_Enum::TENSOR_3D_PLANAR:
    case odb_Enum::TENSOR_3D_SURFACE:
    case odb_Enum::TENSOR_2D_PLANAR:
    case odb_Enum::TENSOR_2D_SURFACE:
        Data::TYPE = Data::TENSOR;
        break;
    }
    // first we check the position and section point
    if (!CheckLocation(field, onlyThisSection, location))
    {
        return;
    }

    // trivial are the cases with an invariant or...
    if (invariant != "None"
        && invariant != FieldLabel::InvariantEnumToString(odb_Enum::UNDEFINED_INVARIANT))
    {
        Data::TYPE = Data::SCALAR;
        const odb_SequenceInvariant &invars = field.validInvariants();
        int numInvar = invars.size();
        INV_FUNC inv_func = NULL;
        int inv;
        for (inv = 0; inv < numInvar; ++inv)
        {
            if (invariant == FieldLabel::InvariantEnumToString(invars.constGet(inv)))
            {
                inv_func = FieldLabel::InvariantEnumToINV_FUNC(invars.constGet(inv));
                break;
            }
        }
        assert(invars.isMember(invars.constGet(inv)));
        if (inv_func)
        {
            odb_SequenceFieldValue fvCon = field.values();
            int numVal = fvCon.size();
            int val;
            for (val = 0; val < numVal; ++val)
            {
                const odb_FieldValue &f = fvCon.constGet(val);
                int instance_number = CheckInstanceLocationSectionPoint(f,
                                                                        location, sectionPoint, onlyThisSection);
                if (instance_number >= 0)
                {
                    // all tests have been OK...
                    InstanceMesh &ThisInstance = local_instance_meshes[instance_number];
                    ThisInstance.ReadInvariant(f, inv_func);
                }
            }
        }
    }
    // ...with ((component && localCS) || field is scalar)
    else if (type == odb_Enum::SCALAR
             || (component != "None" && localCS))
    {
        int dataposition = -1;
        Data::TYPE = Data::SCALAR;
        if (type == odb_Enum::SCALAR)
        {
            dataposition = 0;
        }
        else
        {
            int numComp = field.componentLabels().size();
            int j;
            for (j = 0; j < numComp; j++)
            {
                if (component == field.componentLabels().constGet(j).CStr())
                {
                    break;
                }
            }
            if (j == numComp)
            {
                cerr << component.c_str() << " not found, this is probably a bug in ReadABAQUS" << endl;
                return;
            }
            dataposition = j;
        }
        if (dataposition != -1)
        {
            odb_SequenceFieldValue fvCon = field.values();
            int numVal = fvCon.size();
            int val;
            for (val = 0; val < numVal; ++val)
            {
                const odb_FieldValue &f = fvCon.constGet(val);
                int instance_number = CheckInstanceLocationSectionPoint(f,
                                                                        location, sectionPoint, onlyThisSection);
                if (instance_number >= 0)
                {
                    // all tests have been OK...
                    InstanceMesh &ThisInstance = local_instance_meshes[instance_number];
                    ThisInstance.ReadComponent(f, dataposition, conjugate);
                }
            }
        }
    }
    else if (type == odb_Enum::VECTOR && field.componentLabels().size() == 3)
    {
        int dataposition = -1;
        Data::TYPE = Data::VECTOR;
        if (type == odb_Enum::VECTOR)
        {
            dataposition = 0;
            int numComp = field.componentLabels().size();
            int j;
            vector<int> order;
            for (j = 0; j < numComp; j++)
            {
            }
            odb_SequenceFieldValue fvCon = field.values();
            int numVal = fvCon.size();
            int val;
            for (val = 0; val < numVal; ++val)
            {
                const odb_FieldValue &f = fvCon.constGet(val);
                int instance_number = CheckInstanceLocationSectionPoint(f,
                                                                        location, sectionPoint, onlyThisSection);
                if (instance_number >= 0)
                {
                    // all tests have been OK...

                    if (f.precision() == odb_Enum::DOUBLE_PRECISION)
                    {
                        Data *vectorData = new VectorData(f.dataDouble().constGet(0), f.dataDouble().constGet(1), f.dataDouble().constGet(2));
                        local_instance_meshes[instance_number].ReadData(vectorData, f);
                    }
                    else
                    {
                        Data *vectorData = new VectorData(f.data().constGet(0), f.data().constGet(1), f.data().constGet(2));
                        local_instance_meshes[instance_number].ReadData(vectorData, f);
                    }
                }
            }
        }
    }
    // we want to show local reference systems
    else if (type != odb_Enum::SCALAR && localCS)
    {
        Data::TYPE = Data::REFERENCE_SYSTEM;
        odb_SequenceFieldValue fvCon = field.values();
        int numVal = fvCon.size();
        int val;
        for (val = 0; val < numVal; ++val)
        {
            const odb_FieldValue &f = fvCon.constGet(val);
            int instance_number = CheckInstanceLocationSectionPoint(f,
                                                                    location, sectionPoint, onlyThisSection);
            if (instance_number >= 0)
            {
                // all tests have been OK...
                InstanceMesh &ThisInstance = local_instance_meshes[instance_number];
                ThisInstance.ReadLocalReferenceSystem(f);
            }
        }
    }
    // !scalar and !localCS
    // we will have to compute the field in the global ref. system
    // and output the whole thing if !component or
    // a single component otherwise
    else
    {
        int dataposition = -1;
        int numComp = field.componentLabels().size();
        int j;
        for (j = 0; j < numComp; j++)
        {
            if (component == field.componentLabels().constGet(j).CStr())
            {
                dataposition = j;
                break;
            }
        }
        // if dataposition == -1 we will read the whole field
        odb_SequenceFieldValue fvCon = field.values();
        ComponentTranslator ct(field.componentLabels());
        int numVal = fvCon.size();
        int val;
        for (val = 0; val < numVal; ++val)
        {
            const odb_FieldValue &f = fvCon.constGet(val);
            int instance_number = CheckInstanceLocationSectionPoint(f,
                                                                    location, sectionPoint, onlyThisSection);
            if (instance_number >= 0)
            {
                // all tests have been OK...
                InstanceMesh &ThisInstance = local_instance_meshes[instance_number];
                ThisInstance.ReadGlobal(f, dataposition, ct, conjugate);
            }
        }
        if (dataposition != -1)
        {
            Data::TYPE = Data::SCALAR;
        }
    }
}

int
ChoiceState::CheckInstanceLocationSectionPoint(const odb_FieldValue &f,
                                               const string &location,
                                               const string &sectionPoint,
                                               bool onlyThisSection) const
{
    // get the instance
    string instance_name = f.instance().name().CStr();
    map<string, int>::const_iterator it = _instance_dictionary.find(instance_name);
    if (it == _instance_dictionary.end())
    {
        return -1;
    }
    // and test the instance
    int instance_number = it->second;
    if (!ShowInstance(instance_number))
    {
        return -1;
    }
    // we have to check location and section point
    // of this odb_FieldValue
    if (location != Location::ResultPositionEnumToString(f.position()))
    {
        return -1;
    }
    // section point
    odb_SectionPoint sP = f.sectionPoint();
    if (sP.number() <= 0)
    {
        SectionPoint DummySP;
        if (sectionPoint != DummySP.str()
            && onlyThisSection)
        {
            return -1;
        }
    }
    else // here we check for exact equality
    {
        SectionPoint TestSP(sP.number(), sP.description().CStr());
        if (sectionPoint != TestSP.str())
        {
            return -1;
        }
    }
    return instance_number;
}

void
ChoiceState::ReadDisplacements(odb_FieldOutput &disp_field,
                               vector<InstanceMesh> &local_instance_meshes)
{
    // get component labels
    vector<string> compLabels;
    int numComp = disp_field.componentLabels().size();
    int j;
    for (j = 0; j < numComp; j++)
    {
        compLabels.push_back(disp_field.componentLabels().constGet(j).CStr());
    }
    vector<int> order;
    for (j = 0; j < numComp; j++)
    {
        if (compLabels[j] == "U1")
        {
            order.push_back(0);
        }
        else if (compLabels[j] == "U2")
        {
            order.push_back(1);
        }
        else if (compLabels[j] == "U3")
        {
            order.push_back(2);
        }
        else
        {
            order.push_back(-1);
        }
    }
    // read now all field values
    odb_SequenceFieldValue fvCon = disp_field.values();
    int numVal = fvCon.size();
    int val;
    for (val = 0; val < numVal; ++val)
    {
        const odb_FieldValue &f = fvCon.constGet(val);
        if (f.nodeLabel() < 0)
        {
            continue;
        }
        string instance_name = f.instance().name().CStr();
        map<string, int>::iterator it = _instance_dictionary.find(instance_name);
        if (it == _instance_dictionary.end() || !ShowInstance(it->second))
        {
            continue;
        }
        int instance_number = it->second;
        odb_SequenceFloat data;
        if (f.precision() == odb_Enum::DOUBLE_PRECISION)
        {
            data = f.dataDouble();
        }
        else
        {
            data = f.data();
        }
        local_instance_meshes[instance_number].ReadDisplacement(f.nodeLabel(),
                                                                order, data);
    }
}

void
ChoiceState::ReadStatus(odb_FieldOutput &status_field,
                        vector<InstanceMesh> &local_instance_meshes)
{

    odb_Enum::odb_DataTypeEnum type = status_field.type();
    if (type != odb_Enum::SCALAR)
    {
        return;
    }
    // read now all field values
    odb_SequenceFieldValue fvCon = status_field.values();
    int numVal = fvCon.size(); // fvCon contains one entry per element
    int val;
    for (val = 0; val < numVal; ++val)
    {
        const odb_FieldValue &f = fvCon.constGet(val);
        string instance_name = f.instance().name().CStr();
        map<string, int>::iterator it = _instance_dictionary.find(instance_name);
        if (it == _instance_dictionary.end() || !ShowInstance(it->second))
        {
            continue;
        }
        int instance_number = it->second;
        odb_SequenceFloat data;
        if (f.precision() == odb_Enum::DOUBLE_PRECISION)
        {
            data = f.dataDouble();
        }
        else
        {
            data = f.data();
        }
        local_instance_meshes[instance_number].ReadStatus(f.elementLabel(), data);
    }
}

bool
ChoiceState::ShowInstance(int instance_number) const
{
    return true;
}

bool
ChoiceState::harmonic() const
{
    return _harmonic;
}
