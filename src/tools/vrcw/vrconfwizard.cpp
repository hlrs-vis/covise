#include "vrconfwizard.h"

#include <QMessageBox>
#include <QCloseEvent>

#include <iostream>

#include "vrcwstart.h"
#include "vrcwprojectionhw.h"
#include "vrcwprojectiondimpowerwall.h"
#include "vrcwprojectiondimcave.h"
#include "vrcwhost.h"
#include "vrcwhostprojection.h"
#include "vrcwtrackinghw.h"
#include "vrcwtrackingdim.h"
#include "vrcwfinal.h"


/*****
 * constructor - destructor
 *****/

VRConfWizard::VRConfWizard(QWidget* parent) :
   QMainWindow(parent)
{
   ui.setupUi(this);

   //Seite1
   start = new VRCWStart(ui.page1);
   ui.page1VLayout->addWidget(start);
   vrcwList.append(start);
   //Seite2
   projectionHw = new VRCWProjectionHw(ui.page2);
   ui.page2VLayout->addWidget(projectionHw);
   vrcwList.append(projectionHw);
   //Seite3 (Powerwall oder Cave)
   projectionDimPowerwall = new VRCWProjectionDimPowerwall(ui.page3);
   ui.page3VLayout->addWidget(projectionDimPowerwall);
   vrcwList.append(projectionDimPowerwall);
   projectionDimCave = new VRCWProjectionDimCave(ui.page3);
   ui.page3VLayout->addWidget(projectionDimCave);
   projectionDimCave->hide();
   //Seite4
   hostWidget = new VRCWHost(ui.page4);
   ui.page4VLayout->addWidget(hostWidget);
   vrcwList.append(hostWidget);
   //Seite5
   hostProjection = new VRCWHostProjection(ui.page5);
   ui.page5VLayout->addWidget(hostProjection);
   vrcwList.append(hostProjection);
   //Seite6
   trackingHw = new VRCWTrackingHw(ui.page6);
   ui.page6VLayout->addWidget(trackingHw);
   vrcwList.append(trackingHw);
   //Seite7
   trackingDim = new VRCWTrackingDim(ui.page7);
   ui.page7VLayout->addWidget(trackingDim);
   vrcwList.append(trackingDim);
   //Seite8
   final = new VRCWFinal(ui.page8);
   ui.page8VLayout->addWidget(final);
   vrcwList.append(final);

   //VRConfWizard-Klasse immer ans Ende der Liste haengen
   vrcwList.append(this);

   //immer bei Seite 1 beginnen
   ui.mainStackedWidget->setCurrentIndex(0);

   //Empfang des Signal von vrcwfinal
   connect(final, SIGNAL(configIsModified(bool)), this,
         SLOT(vrcwFinalConfigModified(bool)));

   //Verbinden der Slots setExpertMode mit dem Signal expertModeChanged
   connect(this, SIGNAL(expertModeChanged(bool)), start,
            SLOT(setExpertMode(bool)));
   connect(this, SIGNAL(expertModeChanged(bool)), projectionHw,
         SLOT(setExpertMode(bool)));
   connect(this, SIGNAL(expertModeChanged(bool)), hostWidget,
         SLOT(setExpertMode(bool)));
   connect(this, SIGNAL(expertModeChanged(bool)), trackingHw,
         SLOT(setExpertMode(bool)));
}

VRConfWizard::~VRConfWizard()
{

}


/*****
 * public functions
 *****/

//anzeigen der ProjectonConfigDimensions fuer Powerwall oder Cave
//
void VRConfWizard::changeProConfDimPowerCave(const int& index,
      const proKind& kind, const bool& tiled, const typePro& typeP,
      const stType& stereo, const bool& bothEyes, const int& graka)
{
   switch (kind)
   {
      case CAVE:
      {
         //entsprechende Config anzeigen
         projectionDimPowerwall->hide();
         projectionDimCave->show();

         //Index in vrcwList anpassen
         vrcwList.removeAt(index + 1);
         vrcwList.insert(index + 1, projectionDimCave);

         //Liste der zu konfigurierenden Waende besorgen
         QList<cWall> walls = projectionHw->getGuiWalls();

         //Liste an dimCave weiterreichen
         projectionDimCave->setGuiWallsToConfigure(walls);

         //Parameter fuer die Bestimmung der Projection weitergeben
         projectionDimCave->setProjectionHwParams(kind, tiled, typeP, stereo,
               bothEyes, graka);
         break;
      }
      case Powerwall://Powerwall and _3D_TV are the same
      case _3D_TV:
      {
         //entsprechende Config anzeigen
         projectionDimCave->hide();
         projectionDimPowerwall->show();

         //Index in vrcwList anpassen
         vrcwList.removeAt(index + 1);
         vrcwList.insert(index + 1, projectionDimPowerwall);

         //Parameter fuer die Bestimmung der Projection weitergeben
         projectionDimPowerwall->setProjectionHwParams(kind, tiled, typeP,
               stereo, bothEyes, graka);
         break;
      }
   }
}


/*****
 * protected functions
 *****/

//eigene Version des virtuellen Events
//verwendet bei Signals (close()) abhaengig davon,
//ob finish() oder abort_vrcw() ausgefuehrt wird
//
void VRConfWizard::closeEvent(QCloseEvent* event)
{
   if (ui.finishButton->isEnabled())
   {
      if (finish())
      {
         event->accept();
      }
      else
      {
         event->ignore();
      }
   }
   else
   {
      if (abort_vrcw())
      {
         event->accept();
      }
      else
      {
         event->ignore();
      }
   }
}


/*****
 * private functions
 *****/

//generic Input Processing
//
int VRConfWizard::procPageInput(const int& index) const
{
   const int ERROR_1 = 91;
   const int DEF_ERROR = 99;

   int success = DEF_ERROR;

   if (index < vrcwList.size())
   {
      switch (index)
      {
            // fuer index = (
            //               1 == ProjectionHW (Seite 2)
            //          oder 2 == ProjectionDim... (Seite 3)
            //          oder 3 == Host (Seite 4)
            //          oder 5 == TrackingHW (Seite 6) )
            // soll die gleiche Funktion aufgerufen werden
         case 1:
         case 2:
         case 3:
         case 5:
         {
            success = vrcwList[index]->processGuiInput(index, vrcwList);
            break;
         }
            // fuer index = (
            //               0 == Start (Seite 1)
            //               4 == HostProjection (Seite 5)
            //          oder 6 == TrackingDim (Seite 7) )
            // soll die gleiche Funktion aufgerufen werden
         default:
         {
            success = vrcwList[index]->processGuiInput(vrcwList);
            break;
         }
      }
   }
   else
   {
      std::cout << "Fuer Index " << index << " (Seite " << index + 1
            << ") ist kein Eintrag in der WidgetListe vrcwList enthalten!"
            << std::endl;

      success = ERROR_1;
   }

   std::cout << "Success: " << success << std::endl;
   std::cout << std::endl;

   return success;
}

//Button Action for closeEvent
//and for the finish and close button
//
bool VRConfWizard::finish()
{
   if (this->isWindowModified())
   {
      QString message = tr("You have unsaved changes in your config.\n\n"
            "Do you want to save them before closing?");
      int button = QMessageBox::question(this,
            tr("COVISE VR Configuration Wizard"), message,
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);

      if (button == QMessageBox::Save)
      {
         if (final->saveConfig())
         {
            return exitVRConfWizard();
         }
         else
         {
            return false;
         }
      }
      else if (button == QMessageBox::Discard)
      {
         return true;
      }
      else
      {
         return false;
      }
   }
   else
   {
      return exitVRConfWizard();
   }
}

bool VRConfWizard::abort_vrcw()
{
   QString message = tr("All changes will be lost!\n\n"
         "Do you want to abort the configuration?");
   int button = QMessageBox::warning(this,
         tr("COVISE VR Configuration Wizard"), message,
         QMessageBox::Abort | QMessageBox::Cancel, QMessageBox::Cancel);

   if (button == QMessageBox::Abort)
   {
      return true;
   }
   else
   {
      return false;
   }
}

//called from finish when nothing is to save
//
bool VRConfWizard::exitVRConfWizard()
{
   QMessageBox msgBox(this);
   msgBox.setIcon(QMessageBox::Question);
   msgBox.setText("All changes are saved.");
   msgBox.setInformativeText(tr("Do you want to exit the\n"
         "COVISE VR Configuration Wizard?"));
   msgBox.setWindowTitle(tr("COVISE VR Configuration Wizard"));
   QPushButton* exitButton = msgBox.addButton(tr("Exit"), QMessageBox::NoRole);
   QPushButton* cancelButton = msgBox.addButton(QMessageBox::Cancel);
   msgBox.setDefaultButton(exitButton);
   msgBox.exec();

   if (msgBox.clickedButton() == exitButton)
   {
      return true;
   }
   else
   {
      return false;
   }
}


/*****
 * private slots
 *****/

//Button-Action Navigation
//
void VRConfWizard::next() const
{
   const int MAX_OK = 10;
   const int count = ui.mainStackedWidget->count(); //count goes from 1 to n
   const int index = ui.mainStackedWidget->currentIndex(); //index goes from
                                                           //0 to n-1
   int success = procPageInput(index);

   // if (success == 0) {
   if (success <= MAX_OK)
   {
      if ((index + 1) < count)
      {
         ui.mainStackedWidget->setCurrentIndex(index + 1);
      }

      ui.backButton->setEnabled(true);
      ui.nextButton->setEnabled((index + 1) < (count - 1));
      ui.finishButton->setEnabled((index + 1) == (count - 1));
      ui.abortButton->setEnabled((index + 1) < (count - 1));

      ui.action_Back->setEnabled(true);
      ui.action_Next->setEnabled((index + 1) < (count - 1));
      ui.action_Finish->setEnabled((index + 1) == (count - 1));
      ui.action_Abort->setEnabled((index + 1) < (count - 1));
   }
}

void VRConfWizard::back() const
{
   const int index = ui.mainStackedWidget->currentIndex();

   if (index > 0)
   {
      ui.mainStackedWidget->setCurrentIndex(index - 1);
   }

   ui.backButton->setEnabled(index - 1);
   ui.nextButton->setEnabled(true);
   ui.finishButton->setEnabled(false);
   ui.abortButton->setEnabled(true);

   ui.action_Back->setEnabled(index - 1);
   ui.action_Next->setEnabled(true);
   ui.action_Finish->setEnabled(false);
   ui.action_Abort->setEnabled(true);
}

//Entgegennehmen des Modification-Status von textEdit von vrcwfinal
//und Setzen des Modification-Status der Anwendung
//
void VRConfWizard::vrcwFinalConfigModified(const bool& modified)
{
   this->setWindowModified(modified);
}

//Versenden des Signals expertModeChanged
//
void VRConfWizard::emitExpertModeChanged(const bool& changed)
{
   emit expertModeChanged(changed);
}
