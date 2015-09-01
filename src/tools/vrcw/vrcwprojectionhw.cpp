#include "vrcwprojectionhw.h"

#include <QMessageBox>

#include "vrconfwizard.h"
#include "vrcwfinal.h"


/*****
 * constructor - destructor
 *****/

VRCWProjectionHw::VRCWProjectionHw(QWidget* parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //GUI erstellen
   //
   ctrlMonRes = new VRCWProjectionResSize(ui.ctrlMonResWidget);
   ui.ctrlMonResWidgetGLayout->addWidget(ctrlMonRes);
   ctrlMonRes->showCtrlMonConfig();
   ctrlMon_exec();
   lrRB_exec();

   //set stereo mode comboBox
   typePro_exec();

   //displayProjectorWidget nicht anzeigen
   ui.displayProjectorWidget->hide();

   //CaveWidget nicht anzeigen
   ui.caveWidget->hide();

   //GraphicCards per host auf 1 setzen
   ui.grakaSpinBox->setMaximum(1);

   //Variable setzen
   expertMode = false;
}

VRCWProjectionHw::~VRCWProjectionHw()
{

}


/*****
 * public functions
 *****/

//Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
//Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWProjectionHw::processGuiInput(const int& index,
      const QList<VRCWBase*>& vrcwList)
{
   const int ERROR_1 = 21;
   const int OK = 2;

   int success = DEF_ERROR;

   //when CAVE selected check that more than one wall is chosen
   if (getGuiKind() == CAVE && getGuiWalls().size() == 1)
   {
      QString message = tr("You selected a CAVE with only the "
            "front wall to configure.\n"
            "But this is the same as a Powerwall.\n\n"
            "Please choose more walls or choose the Powerwall.");
      QMessageBox::warning(this, tr("Configuration"), message,
            QMessageBox::Ok);

      success = ERROR_1;
   }
   else
   {
      success = OK;
   }

   //sobald keine Fehler in der Eingabe sind:
   //- wird die ProjectionConfiguration abhaengig von kind gesetzt und
   //  die erfragten Parameter zur Weiterverarbeitung an ProjectionDim...
   //  uebergeben
   //- wird die Liste der Variablen an VRCWFinal uebergeben
   //
   if (success == OK)
   {
      int finalIndex = vrcwList.size() - 2;
      VRConfWizard* vrConfWizard =
            dynamic_cast<VRConfWizard*> (vrcwList.last());
      VRCWFinal* final = dynamic_cast<VRCWFinal*> (vrcwList[finalIndex]);

      //set variables
      proKind kind = getGuiKind();
      bool tiled = getGuiTiled();
      typePro typeP = getTypePro();
      stType stereo = getGuiStereo();
      bool bothEyes = getGuiBothEyes();
      int graka = getGuiGraka();
      bool cMon = getCtrlMon();
      QVector<int> cMonRes = ctrlMonRes->getGuiRes();
      loRoOrient cMon3DOrient = getCMon3DOrient();


      vrConfWizard->changeProConfDimPowerCave(index, kind, tiled, typeP,
            stereo, bothEyes, graka);

      final->setProjectionHw(kind, tiled, stereo, bothEyes, graka, cMon, cMonRes, cMon3DOrient);
   }

   return success;
}

//Auslesen des GUI
//
QList<cWall> VRCWProjectionHw::getGuiWalls() const
{
   QList<cWall> w;

   w.append(Front);

   if (ui.caveLeftCheckBox->isChecked())
   {
      w.append(Left);
   }
   if (ui.caveRightCheckBox->isChecked())
   {
      w.append(Right);
   }
   if (ui.caveBottomCheckBox->isChecked())
   {
      w.append(Bottom);
   }
   if (ui.caveTopCheckBox->isChecked())
   {
      w.append(Top);
   }
   if (ui.caveBackCheckBox->isChecked())
   {
      w.append(Back);
   }

   return w;
}


/*****
 * private functions
 *****/

//Auslesen des GUI
//
proKind VRCWProjectionHw::getGuiKind() const
{
   return strToProKind(ui.projectionComboBox->currentText());
}

bool VRCWProjectionHw::getGuiTiled() const
{
   return ui.tiledCheckBox->isChecked();
}

typePro VRCWProjectionHw::getTypePro() const
{
   typePro typeP;

   if (ui.projectorRadioButton->isChecked())
   {
      typeP = Projector;
   }
   else
   {
      typeP = Monitor;
   }

   return typeP;
}

stType VRCWProjectionHw::getGuiStereo() const
{
   return strToStType(ui.stereoComboBox->currentText());
}

bool VRCWProjectionHw::getGuiBothEyes() const
{
   return ui.bothEyesCheckBox->isChecked();
}

int VRCWProjectionHw::getGuiGraka() const
{
   return ui.grakaSpinBox->value();
}

bool VRCWProjectionHw::getCtrlMon() const
{
   return ui.ctrlMonCheckBox->isChecked();
}

loRoOrient VRCWProjectionHw::getCMon3DOrient() const
{
   loRoOrient oMon3D;

   if (ui.leftOfRadioButton->isChecked())
   {
      oMon3D = leftOf;
   }
   else
   {
      oMon3D = rightOf;
   }

   return oMon3D;
}


/*****
 * private slots
 *****/

//show/hide the cave walls widget
//enable/disable tiled checkbox
//verarbeiten der tiled-Checkbox
//
void VRCWProjectionHw::kindPro_exec() const
{
   proKind kind = getGuiKind();

   //show/hide the cave walls widget
   //enable/disable tiled checkbox
   switch (kind)
   {
      case Powerwall:
      {
         ui.tiledCheckBox->setEnabled(true);
         ui.caveWidget->hide();
         break;
      }
      case CAVE:
      {
         ui.tiledCheckBox->setEnabled(true);
         ui.caveWidget->show();
         break;
      }
      case _3D_TV:
      {
         ui.tiledCheckBox->setEnabled(false);
         ui.caveWidget->hide();
         break;
      }
   }

   //uncheck tiled if kind is changed
   ui.tiledCheckBox->setChecked(false);

   tiled_exec();
}

//fuer tiled Powerwall/Cave soll der Typ der Anzeige gewaehlt werden koennen
//entweder Display/Monitor/TV oder Projektor
//set default projection type
//verarbeiten der Anzeigegeraete
//
void VRCWProjectionHw::tiled_exec() const
{
   proKind kind = getGuiKind();
   bool checked = getGuiTiled();

   //default projection type
   //show/hide displayProjectorWidget
   if (checked)
   {
      ui.displayProjectorWidget->show();
   }
   else
   {
      ui.displayProjectorWidget->hide();

      switch (kind)
      {
         case Powerwall:
         case CAVE:
         {
            ui.projectorRadioButton->setChecked(true);
            break;
         }
         case _3D_TV:
         {
            ui.displayRadioButton->setChecked(true);
            break;
         }
      }
   }

   typePro_exec();
}

//mit typePro wird bestimmt, welche Stereoarten das jeweilige Anzeigegeraet unterstuetzt
//set stereo type in stereoComboBox
//
void VRCWProjectionHw::typePro_exec() const
{
   typePro typeP = getTypePro();

   //set stereo type in stereoComboBox and number of graphic cards
   ui.stereoComboBox->clear();

   switch (typeP)
   {
      case Monitor:
      {
         ui.stereoComboBox->addItem(stTypeToStr(topBottom));
         ui.stereoComboBox->addItem(stTypeToStr(sideBySide));
         ui.stereoComboBox->addItem(stTypeToStr(checkerboard));
         ui.stereoComboBox->addItem(stTypeToStr(vInterlaced));
         ui.stereoComboBox->addItem(stTypeToStr(cInterleave));
         ui.stereoComboBox->addItem(stTypeToStr(hInterlaced));
         ui.stereoComboBox->addItem(stTypeToStr(rInterleave));
         ui.stereoComboBox->addItem(stTypeToStr(active));
         //one 1 graphic card should be possible
         ui.grakaSpinBox->setMaximum(1);
         break;
      }
      case Projector:
      {
         ui.stereoComboBox->addItem(stTypeToStr(passive));
         ui.stereoComboBox->addItem(stTypeToStr(topBottom));
         ui.stereoComboBox->addItem(stTypeToStr(sideBySide));
         ui.stereoComboBox->addItem(stTypeToStr(active));
         //at the moment only 1 graphic card is enabled
         ui.grakaSpinBox->setMaximum(1);
         break;
      }
   }
}

//damit die Front CheckBox immer gecheckt ist,
//denn sie soll immer konfiguriert werden
//
void VRCWProjectionHw::frontChecked() const
{
   ui.caveFrontCheckBox->setChecked(true);
}

//back wall auswaehlbar in Abhaengigkeit von expertMode
//
void VRCWProjectionHw::backEnable() const
{
   bool left = ui.caveLeftCheckBox->isChecked();
   bool right = ui.caveRightCheckBox->isChecked();
   bool bottom = ui.caveBottomCheckBox->isChecked();
   bool top = ui.caveTopCheckBox->isChecked();

   if (expertMode)
   {
      //die back wall kann ohne left, right, bottom oder top
      //nicht konfiguriert werden
      if (left || right || bottom || top)
      {
         ui.caveBackCheckBox->setEnabled(true);
      }
      else
      {
         ui.caveBackCheckBox->setEnabled(false);
         ui.caveBackCheckBox->setChecked(false);
      }
   }
   else
   {
      //im nicht-expertMode ist die Rueckwand nur auswaehlbar
      //wenn die rechte und linke Wand ausgewaehlt ist
      if (left && right)
      {
         ui.caveBackCheckBox->setEnabled(true);
      }
      else
      {
         ui.caveBackCheckBox->setEnabled(false);
         ui.caveBackCheckBox->setChecked(false);
      }
   }
}

//im passive mode ist das Rendern eines Auges oder beider Augen
//pro Grafikkarte moeglich,
//bei active macht die Frage keinen Sinn, es werden immer beide Augen
//pro Grafikkarte gerendert
//
void VRCWProjectionHw::stereo_exec() const
{
   switch (getGuiStereo())
   {
      case passive:
      {
         ui.bothEyesCheckBox->setChecked(false);
         ui.bothEyesCheckBox->setEnabled(true);
         break;
      }
      case active:     //active, topBottom, sideBySide, checkerboard, vInterlaced,
      case topBottom:  //cInterleave, hInterlaced and rInterleave are the same
      case sideBySide:
      case checkerboard:
      case vInterlaced:
      case cInterleave:
      case hInterlaced:
      case rInterleave:
      {
         ui.bothEyesCheckBox->setChecked(true);
         ui.bothEyesCheckBox->setEnabled(false);
         break;
      }
   }
}

//show/hide die Konfiguration fuer den Control Monitor
//
void VRCWProjectionHw::ctrlMon_exec() const
{
   if (ui.ctrlMonCheckBox->isChecked())
   {
      ui.ctrlMonResWidget->setEnabled(true);
      ui.ctrlMonOrientWidget->show();
   }
   else
   {
      ui.ctrlMonResWidget->setEnabled(false);
      ui.ctrlMonOrientWidget->hide();
   }
}

//show monitor leftOf/rightOf 3D projection Picture
//
void VRCWProjectionHw::lrRB_exec() const
{
   if (ui.leftOfRadioButton->isChecked())
   {
      ui.loDescPictLabel->show();
      ui.roDescPictLabel->hide();
   }
   else
   {
      ui.loDescPictLabel->hide();
      ui.roDescPictLabel->show();
   }
}

//Setzen der Variable expertMode
//
void VRCWProjectionHw::setExpertMode(const bool& changed)
{
   expertMode = changed;

   //in backEnable() wird expertMode ausgewertet
   backEnable();
}
