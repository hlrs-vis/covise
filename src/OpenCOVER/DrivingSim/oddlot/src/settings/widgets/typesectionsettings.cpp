/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "typesectionsettings.hpp"
#include "typesectionsettingsUI.h"

#include "src/data/commands/typesectioncommands.hpp"
#include "src/gui/projectwidget.hpp"

TypeSectionSettings::TypeSectionSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, TypeSection *typeSection)
    : SettingsElement(projectSettings, parentSettingsElement, typeSection)
    , ui(new TypeSectionSettingsUI)
    , typeSection_(typeSection)
    , init_(false)
{
    ui->setupUI(this);

    // Initial Values //
    //
    updateProperties();

    init_ = true;
}

TypeSectionSettings::~TypeSectionSettings(void)
{
    delete ui;
}

void TypeSectionSettings::updateProperties()
{
    ui->getRoadTypeComboBox()->getComboBox()->setCurrentIndex(typeSection_->getRoadType() - 1);
    if(typeSection_->getSpeedRecord()!=NULL)
    {
        bool before = ui->getSpinBox()->blockSignals(true);
        ui->getSpinBox()->setValue(typeSection_->getSpeedRecord()->maxSpeed);
        ui->getSpinBox()->blockSignals(before);
    }
}

//##################//
// Observer Pattern //
//##################//

void
TypeSectionSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // TypeSection //
    //
    int changes = typeSection_->getTypeSectionChanges();

    if (changes & TypeSection::CDE_SelectionChange)
    {
        updateProperties();
    }
}

//################//
// SLOTS          //
//################//

void TypeSectionSettings::on_maxSpeed_valueChanged(double ms)
{
    QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

        // Macro Command //
        //
        int numberOfSelectedElements = selectedElements.size();
        if (numberOfSelectedElements > 1)
        {
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set MaxSpeed"));
        }

        // Change types of selected items //
        //
        foreach (DataElement *element, selectedElements)
        {
            TypeSection *typeSection = dynamic_cast<TypeSection *>(element);
            if (typeSection)
            {
                SetSpeedTypeSectionCommand *command = new SetSpeedTypeSectionCommand(typeSection, ms, NULL);
                getProjectSettings()->executeCommand(command);
            }
        }

        // Macro Command //
        //
        if (numberOfSelectedElements > 1)
        {
            getProjectData()->getUndoStack()->endMacro();
        }
}
void
TypeSectionSettings::on_roadTypeBox_activated(int id)
{
    if (id != lastIndex_)
    {
        TypeSection::RoadType roadType;

        if (id == 0)
        {
            roadType = TypeSection::RTP_UNKNOWN;
        }
        else if (id == 1)
        {
            roadType = TypeSection::RTP_RURAL;
        }
        else if (id == 2)
        {
            roadType = TypeSection::RTP_MOTORWAY;
        }
        else if (id == 3)
        {
            roadType = TypeSection::RTP_TOWN;
        }
        else if (id == 4)
        {
            roadType = TypeSection::RTP_LOWSPEED;
        }
        else
        {
            roadType = TypeSection::RTP_PEDESTRIAN;
        }

        QList<DataElement *> selectedElements = getProjectData()->getSelectedElements();

        // Macro Command //
        //
        int numberOfSelectedElements = selectedElements.size();
        if (numberOfSelectedElements > 1)
        {
            getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set Road Type"));
        }

        // Change types of selected items //
        //
        foreach (DataElement *element, selectedElements)
        {
            TypeSection *typeSection = dynamic_cast<TypeSection *>(element);
            if (typeSection)
            {
                SetTypeTypeSectionCommand *command = new SetTypeTypeSectionCommand(typeSection, roadType, NULL);
                getProjectSettings()->executeCommand(command);
            }
        }

        // Macro Command //
        //
        if (numberOfSelectedElements > 1)
        {
            getProjectData()->getUndoStack()->endMacro();
        }

        lastIndex_ = id;
    }
}
