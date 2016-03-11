/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/26/2010
**
**************************************************************************/

#include "trackcomponentsettings.hpp"
#include "ui_trackcomponentsettings.h"

// Data //
//
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/commands/trackcommands.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"

// Qt //
//

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

TrackComponentSettings::TrackComponentSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, TrackComponent *trackComponent)
    : SettingsElement(projectSettings, parentSettingsElement, trackComponent)
    , ui(new Ui::TrackComponentSettings)
    , trackComponent_(trackComponent)
    , line_(NULL)
    , arc_(NULL)
    , spiral_(NULL)
    , poly3_(NULL)
    , sparcs_(NULL)
{
    ui->setupUi(this);

    // Casts //
    //
    TrackComponent::DTrackType type = trackComponent_->getTrackType();
    if (type == TrackComponent::DTT_LINE)
    {
        ui->groupBox->setTitle(tr("line"));
        line_ = dynamic_cast<TrackElementLine *>(trackComponent_);
    }
    else if (type == TrackComponent::DTT_ARC)
    {
        ui->groupBox->setTitle(tr("arc"));
        arc_ = dynamic_cast<TrackElementArc *>(trackComponent_);
    }
    else if (type == TrackComponent::DTT_SPIRAL)
    {
        ui->groupBox->setTitle(tr("spiral"));
        spiral_ = dynamic_cast<TrackElementSpiral *>(trackComponent_);
    }
    else if (type == TrackComponent::DTT_POLY3)
    {
        ui->groupBox->setTitle(tr("poly3"));
        poly3_ = dynamic_cast<TrackElementPoly3 *>(trackComponent_);
    }
    else if (type == TrackComponent::DTT_COMPOSITE)
    {
        ui->groupBox->setTitle(tr("composite"));
    }
    else if (type == TrackComponent::DTT_SPARCS)
    {
        ui->groupBox->setTitle(tr("spiral-arc-spiral"));
        sparcs_ = dynamic_cast<TrackSpiralArcSpiral *>(trackComponent_);
    }
    else
    {
        ui->groupBox->setTitle(tr("unknown type"));
    }

    updateTransformation();
    updateS();
    updateLength();
    updateCurvature();
}

TrackComponentSettings::~TrackComponentSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
TrackComponentSettings::updateTransformation()
{
    ui->xBox->setValue(trackComponent_->getGlobalPoint(trackComponent_->getSStart()).x());
    ui->yBox->setValue(trackComponent_->getGlobalPoint(trackComponent_->getSStart()).y());
    ui->headingBox->setValue(trackComponent_->getGlobalHeading(trackComponent_->getSStart()));
}

void
TrackComponentSettings::updateS()
{
    ui->sBox->setValue(trackComponent_->getSStart());
}

void
TrackComponentSettings::updateLength()
{
    ui->lengthBox->setValue(trackComponent_->getLength());
}

void
TrackComponentSettings::updateCurvature()
{
    ui->curv1Label->setVisible(false);
    ui->curv1Box->setVisible(false);
    ui->curv2Label->setVisible(false);
    ui->curv2Box->setVisible(false);
    ui->line_2->setVisible(false);
    ui->aBox->setVisible(false);
    ui->bBox->setVisible(false);
    ui->cBox->setVisible(false);
    ui->dBox->setVisible(false);
    ui->aLabel->setVisible(false);
    ui->bLabel->setVisible(false);
    ui->cLabel->setVisible(false);
    ui->dLabel->setVisible(false);
    ui->factorBox->setVisible(false);
    ui->factorLable->setVisible(false);

    TrackComponent::DTrackType type = trackComponent_->getTrackType();
    if (type == TrackComponent::DTT_ARC)
    {
        ui->curv1Label->setVisible(true);
        ui->curv1Box->setVisible(true);
        ui->line_2->setVisible(true);
        ui->curv1Label->setText(tr("curvature"));
        ui->curv1Box->setValue(trackComponent_->getCurvature(trackComponent_->getSStart()));
    }
    else if (type == TrackComponent::DTT_SPIRAL)
    {
        ui->curv1Label->setVisible(true);
        ui->curv1Box->setVisible(true);
        ui->curv2Label->setVisible(true);
        ui->curv2Box->setVisible(true);
        ui->line_2->setVisible(true);
        ui->curv1Label->setText(tr("curvStart"));
        ui->curv1Box->setValue(trackComponent_->getCurvature(trackComponent_->getSStart()));
        ui->curv2Label->setText(tr("curvEnd"));
        ui->curv2Box->setValue(trackComponent_->getCurvature(trackComponent_->getSEnd()));
    }
    else if (type == TrackComponent::DTT_POLY3)
    {
        ui->aBox->setVisible(true);
        ui->bBox->setVisible(true);
        ui->cBox->setVisible(true);
        ui->dBox->setVisible(true);
        ui->aLabel->setVisible(true);
        ui->bLabel->setVisible(true);
        ui->cLabel->setVisible(true);
        ui->dLabel->setVisible(true);
        ui->line_2->setVisible(true);

        ui->aBox->setValue(poly3_->getA());
        ui->bBox->setValue(poly3_->getB());
        ui->cBox->setValue(poly3_->getC());
        ui->dBox->setValue(poly3_->getD());
    }
    else if (type == TrackComponent::DTT_SPARCS)
    {
        ui->line_2->setVisible(true);
        ui->factorBox->setVisible(true);
        ui->factorLable->setVisible(true);
        ui->factorBox->setValue(sparcs_->getFactor());
    }
}

//################//
// SLOTS          //
//################//

void
TrackComponentSettings::on_factorBox_editingFinished()
{
    if (!sparcs_)
    {
        return;
    }
    double newValue = ui->factorBox->value();
    if (newValue != sparcs_->getFactor())
    {
        SetSpArcSFactorCommand *command = new SetSpArcSFactorCommand(sparcs_, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
TrackComponentSettings::on_lengthBox_editingFinished()
{
    double newValue = ui->lengthBox->value();
    if (newValue != trackComponent_->getLength())
    {
        SetTrackLengthCommand *command = new SetTrackLengthCommand(trackComponent_, newValue, NULL);
        getProjectSettings()->executeCommand(command);
    }
}
//##################//
// Observer Pattern //
//##################//

void
TrackComponentSettings::updateObserver()
{

    // Get change flags //
    //
    int changes = trackComponent_->getTrackComponentChanges();

    // TrackComponent //
    //
    if ((changes & TrackComponent::CTC_ParentChanged)
        || (changes & TrackComponent::CTC_TransformChange)
        || (changes & TrackComponent::CTC_SChange)
        || (changes & TrackComponent::CTC_LengthChange)
        || (changes & TrackComponent::CTC_ShapeChange))
    {
        updateTransformation();
        updateS();
        updateLength();
        updateCurvature();
    }

    // Parent //
    //
    SettingsElement::updateObserver();
}
