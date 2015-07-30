#include <vrcwhsensbindex.h>


/*****
 * constructor - destructor
 *****/

VRCWHSensBIndex::VRCWHSensBIndex(QWidget *parent) : QWidget(parent)
{
   ui.setupUi(this);

   //Variablen setzen
   tSysData = ART;
   tHVrcData = false;
   osData = Linux;

   //setup GUI
   buttonSys_exec();
   trackSysChanged_exec(tSysData);
   trackHandleVrcChecked_exec(tHVrcData);
   hideConstHeaderDesc();
   setBtnDevComboBoxContent();
   setBtnDrvComboBoxContent();
}

VRCWHSensBIndex::~VRCWHSensBIndex()
{

}


/*****
 * public functions
 *****/

// Auslesen des GUI
//
QString VRCWHSensBIndex::getGuiHSensLabel() const
{
   return ui.hSensLabel->text();
}

btnSys VRCWHSensBIndex::getGuiButtonSystem() const
{
   return strToBtnSys(ui.buttonSysComboBox->currentText());
}

QString VRCWHSensBIndex::getGuiButtonDevice() const
{
   QString bDev;

   if (ui.buttonSysDevWidget->isVisible())
   {
      bDev = ui.buttonDevComboBox->currentText();
   }
   else
   {
      bDev = "";
   }

   if (bDev == OTHER)
   {
      QStringList bDevList = ui.buttonDevLineEdit->text().split(
            QRegExp("\\s+"), QString::SkipEmptyParts);

      //we only use the first entry
      if (bDevList.isEmpty())
      {
         bDevList.append("");
      }
      bDev = bDevList[0];
   }

   return bDev;
}

btnDrv VRCWHSensBIndex::getGuiButtonDriver() const
{
   return strToBtnDrv(ui.buttonDrvComboBox->currentText());
}

int VRCWHSensBIndex::getGuiBodyIndex() const
{
   return ui.bodyIndexSpinBox->value();
}

int VRCWHSensBIndex::getGuiVrcButtonAddr() const
{
   return ui.vrcButtonAddrSpinBox->value();
}



//Setzen des GUI
//
void VRCWHSensBIndex::setHSensLabel(const QString& hsLabel)
{
   ui.hSensLabel->setText(hsLabel);
}

void VRCWHSensBIndex::setButtonSystem(const btnSys& bSys)
{
   QString bSysStr = btnSysToStr(bSys);
   int index = ui.buttonSysComboBox->findText(bSysStr);
   ui.buttonSysComboBox->setCurrentIndex(index);
}

void VRCWHSensBIndex::setButtonDevice(const QString& bDev)
{
   int index = ui.buttonDevComboBox->findText(bDev);

   if (index > 0)
   {
      ui.buttonDevComboBox->setCurrentIndex(index);
   }
   else
   {
      ui.buttonDevComboBox->setCurrentIndex(0);
      ui.buttonDevLineEdit->setText(bDev);
   }
}

void VRCWHSensBIndex::setButtonDriver(const btnDrv& bDrv)
{
   QString bDrvStr = btnDrvToStr(bDrv);
   int index = ui.buttonDrvComboBox->findText(bDrvStr);
   ui.buttonDrvComboBox->setCurrentIndex(index);
}

void VRCWHSensBIndex::setBodyIndex(const int& index)
{
   ui.bodyIndexSpinBox->setValue(index);
}

void VRCWHSensBIndex::setVrcButtonAddr(const int& addr)
{
   ui.vrcButtonAddrSpinBox->setValue(addr);
}

//show only body index and hide button sys and dev
//
void VRCWHSensBIndex::bodyIndexOnly() const
{
   ui.bodyIndexWidget->show();
   ui.buttonSysDevVrcWidget->hide();
   hideConstHeaderDesc();
}

//show only const Header Desc
//
void VRCWHSensBIndex::constHeaderDescOnly(const QString& desc) const
{
   ui.constHeadDescLabel->setText(desc);
   ui.constHeadDescLabel->show();
   ui.buttonSysDevVrcWidget->hide();
   ui.bodyIndexWidget->hide();
   ui.constHeadDescLabelVLayout->setAlignment(Qt::AlignVCenter);
}

//Layout hand/head sensor label for head sensor
//
void VRCWHSensBIndex::headSensLayout()const
{
   ui.hSensLabelVLayout->setAlignment(Qt::AlignVCenter);
   //setContentsMargins(left, top, right, bottom)
   ui.bodyIndexWidgetHLayout->setContentsMargins(0, 3, 0, 0);
}


//set variable osData (operating system)
//
void VRCWHSensBIndex::setOS_GUI(const opSys& os)
{
      osData = os;
}


/*****
 * private functions
 *****/

//const header description
//
void VRCWHSensBIndex::hideConstHeaderDesc() const
{
   ui.constHeadDescLabel->hide();
}

//Set content of buttonDevComboBox
//
void VRCWHSensBIndex::setBtnDevComboBoxContent() const
{
   //set the buttonDevice comboBox
   ui.buttonDevComboBox->clear();
   switch (osData)
   {
      case Linux:
      {
         ui.buttonDevComboBox->addItem(OTHER);
         ui.buttonDevComboBox->addItem("/dev/ttyS0");
         ui.buttonDevComboBox->addItem("/dev/ttyS1");
         ui.buttonDevComboBox->addItem("/dev/usb/ttyUSB0");
         ui.buttonDevComboBox->addItem("/dev/usb/ttyUSB1");
         break;
      }
      case Windows:
      {
         ui.buttonDevComboBox->addItem(OTHER);
         ui.buttonDevComboBox->addItem("COM1");
         ui.buttonDevComboBox->addItem("COM2");
         ui.buttonDevComboBox->addItem("COM3");
         break;
      }
   }
   //set default index for comboBox
   ui.buttonDevComboBox->setCurrentIndex(2);

   btnSys guiButtonSys = getGuiButtonSystem();

   //set buttonDevComboBox for Wiimote
   switch (guiButtonSys)
   {
      case Wiimote:
      {
         //remove all but OTHER entries from buttonDevice comboBox
         for (int i = (ui.buttonDevComboBox->count() -1 ); i > 0; --i)
         {
            ui.buttonDevComboBox->removeItem(i);
         }
         break;
      }
      case ART_Fly:   //ART_Fly, Wand, Stylus, FOB_Mouse,
      case o_Optical: //Hornet and Mike are the same
      case Wand:
      case Stylus:
      case FOB_Mouse:
      case Hornet:
      case Mike:
      {
         //do nothing
         break;
      }
   }
}

//Set content of buttonDrvComboBox
//
void VRCWHSensBIndex::setBtnDrvComboBoxContent() const
{
   btnSys bSys = getGuiButtonSystem();

   //set the buttonDriver comboBox
   ui.buttonDrvComboBox->clear();

   switch (bSys)
   {
      case ART_Fly:
      {
         ui.buttonDrvComboBox->addItem(btnDrvToStr(DTrack));
         break;
      }
      case o_Optical:
      {
         ui.buttonDrvComboBox->addItem(btnDrvToStr(Mouse_Buttons));
         break;
      }
      case Wand: //Wand and Stylus are the same and only with Polhemus
      case Stylus:
      {
         ui.buttonDrvComboBox->addItem(btnDrvToStr(PolhemusDrv));
         break;
      }
      case FOB_Mouse:
      {
         if (tSysData == FOB)
         {
            ui.buttonDrvComboBox->addItem(btnDrvToStr(FOB_Drv));
         }
         else
         {
            ui.buttonDrvComboBox->addItem(btnDrvToStr(MotionstarDrv));
         }
         break;
      }
      case Hornet:
      {
         ui.buttonDrvComboBox->addItem(btnDrvToStr(HornetDrv));
         break;
      }
      case Mike:
      {
         ui.buttonDrvComboBox->addItem(btnDrvToStr(MikeDrv));
         break;
      }
      case Wiimote:
      {
         ui.buttonDrvComboBox->addItem(btnDrvToStr(WiimoteDrv));
         break;
      }
   }
}


/*****
 * private slots
 *****/

//set the hand/button device for the tracking system specified
//in vrcwtrackinghw
//executed per signal/slot defined in vrcwtrackinghw
//
void VRCWHSensBIndex::trackSysChanged_exec(const trackSys& tSys)
{
   tSysData = tSys;

   //button system
   switch (tSysData)
   {
      case ART:
      {
         ui.buttonSysComboBox->clear();
         ui.buttonSysComboBox->addItem(btnSysToStr(ART_Fly));
         ui.buttonSysComboBox->addItem(btnSysToStr(o_Optical));
         break;
      }
      case Vicon:
      {
         ui.buttonSysComboBox->clear();
         ui.buttonSysComboBox->addItem(btnSysToStr(o_Optical));
         break;
      }
      case InterSense://InterSense uses the Polhemus Fastrak protocol
      case Polhemus:
      {
         ui.buttonSysComboBox->clear();
         ui.buttonSysComboBox->addItem(btnSysToStr(Wand));
         ui.buttonSysComboBox->addItem(btnSysToStr(Stylus));
         break;
      }
      case FOB://FOB and Motionstar are the same
      case Motionstar:
      {
         ui.buttonSysComboBox->clear();
         ui.buttonSysComboBox->addItem(btnSysToStr(FOB_Mouse));
         ui.buttonSysComboBox->addItem(btnSysToStr(Hornet));
         ui.buttonSysComboBox->addItem(btnSysToStr(Mike));
         ui.buttonSysComboBox->addItem(btnSysToStr(Wiimote));
         break;
      }
      case Wii:
      {
         ui.buttonSysComboBox->clear();
         ui.buttonSysComboBox->addItem(btnSysToStr(Wiimote));
         break;
      }
   }
}

//enable/disable the hints for tracking with VRC
//executed per signal/slot defined in vrcwtrackinghw
//
void VRCWHSensBIndex::trackHandleVrcChecked_exec(const bool& tHVrc)
{
   tHVrcData = tHVrc;

   if (tHVrcData)
   {
      ui.vrcButtonSysHintWidget->show();
   }
   else
   {
      ui.vrcButtonSysHintWidget->hide();
   }
}

//enable input for button device
//depending on button system
//
void VRCWHSensBIndex::buttonSys_exec() const
{
   //ButtonSystem will be set using a signal/slot connection between
   //vrcwtrackinghw::trackSysComboBox and trackSysChanged_exec()
   //and then by a connection between
   //buttonSysCombobox and buttonSys_exec()

   btnSys guiButtonSys = getGuiButtonSystem();

   //enable/disable button address
   //show/hide buttonDevWidget
   switch (guiButtonSys)
   {
      case ART_Fly://ART_Fly, Wand, Stylus and FOB_Mouse are the same
      case Wand:
      case Stylus:
      case FOB_Mouse:
      {
         //button device
         ui.buttonSysDevWidget->hide();
         //button driver
         ui.buttonDrvWidget->hide();
         break;
      }
      case o_Optical:
      {
         //button device
         ui.buttonSysDevWidget->show();
         //button driver
         ui.buttonDrvWidget->show();
         break;
      }
      case Hornet://Hornet and Mike are the same
      case Mike:
      {
         //button device
         ui.buttonSysDevWidget->show();
         //button driver
         ui.buttonDrvWidget->hide();
         break;
      }
      case Wiimote:
      {
         //button device
         ui.buttonSysDevWidget->show();
         //button driver
         ui.buttonDrvWidget->hide();
         break;
      }
   }

   //enable/disable button Address Spinbox with vrc
   //if vrc is checked tHVrcData is true
   if (tHVrcData)
   {
      switch (guiButtonSys)
      {
         case ART_Fly://ART_Fly, Wand, Stylus and FOB_Mouse are the same
         case Wand:
         case Stylus:
         case FOB_Mouse:
         {
            ui.vrcButtonAddrWidget->hide();
            break;
         }
         case o_Optical: //o_Optical, Hornet, Mike and Wiimote are the same
         case Hornet:
         case Mike:
         case Wiimote:
         {
            ui.vrcButtonAddrWidget->show();
            break;
         }
      }
   }
   else
   {
      ui.vrcButtonAddrWidget->hide();
   }

   //Set content of buttonDevComboBox and buttonDrvComboBox
   setBtnDevComboBoxContent();
   setBtnDrvComboBoxContent();


   //set VRC server hint
   switch (guiButtonSys)
   {
      case ART_Fly:
      {
         ui.vrcButtonSysHintLabel->setText("No additional server is "
               "necessary. Buttons are sent with the ART_DTRACKserver.");
         break;
      }
      case o_Optical:
      {
         switch (osData)
         {
            case Linux:
            {
               ui.vrcButtonSysHintLabel->setText("Start the server "
                     "in a console/terminal that fits to your device.\n"
                     "For example the WIIserver, LOGITECHserver "
                     "or PS2server.");

               break;
            }
            case Windows:
            {
               ui.vrcButtonSysHintLabel->setText("Start the server "
                     "in a console/terminal that fits to your device.\n"
                     "For example the WiiButtonserver.");

               break;
            }
         }
         break;
      }
      case Wand://Wand and Stylus have the same hint
      case Stylus:
      {
         ui.vrcButtonSysHintLabel->setText("No additional server is "
               "necessary. Buttons are sent with the POLHEMUSserver.");
         break;
      }
      case FOB_Mouse:
      {
         switch (tSysData)
         {
            case FOB:
            {
               ui.vrcButtonSysHintLabel->setText("No additional server is "
                     "necessary. Buttons are sent with the FOBserver.");
               break;
            }
            case Motionstar:
            {
               ui.vrcButtonSysHintLabel->setText("No additional server is "
                     "necessary. Buttons are sent with the MOTIONSTARserver.");
               break;
            }
            default:
            {
               //do nothing
               break;
            }
         }
         break;
      }
      case Hornet:
      {
         ui.vrcButtonSysHintLabel->setText("Start the HORNETserver "
               "or MIKEserver in a console/terminal.");
         break;
      }
      case Mike:
      {
         ui.vrcButtonSysHintLabel->setText("Start the MIKEserver "
               "in a console/terminal.");
         break;
      }
      case Wiimote:
      {
         switch (osData)
         {
            case Linux:
            {
               ui.vrcButtonSysHintLabel->setText("Start the WIIserver "
                     "in a console/terminal.");
               break;
            }
            case Windows:
            {
               switch (tSysData)
               {
                  case Wii:
                  {
                     ui.vrcButtonSysHintLabel->setText("No additional server "
                           "is necessary. Buttons are sent with the "
                           "WiiAutoConnect or 'WiiMotionPlus Server'.");
                     break;
                  }
                  default:
                  {
                     ui.vrcButtonSysHintLabel->setText("Start the "
                           "WiiButtonserver in a console/terminal.");
                     break;
                  }
               }
               break;
            }
         }
         break;
      }
   }
}

//show/hide ButtonDevice-Connection lineEdit
//
void VRCWHSensBIndex::buttonDev_exec(const QString& btDev) const
{
   if (btDev == OTHER)
   {
      ui.buttonDevLEditWidget->show();
   }
   else
   {
      ui.buttonDevLEditWidget->hide();
   }
}
