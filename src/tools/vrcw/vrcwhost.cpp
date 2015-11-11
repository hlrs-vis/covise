#include "vrcwhost.h"

#include <QMessageBox>
#include <algorithm>
#include <QHostInfo>
#include <QHostAddress>
#include <QItemSelectionModel>

#include "vrcwhostprojection.h"
#include "vrcwtrackinghw.h"
#include "vrcwfinal.h"
#include "datatypes.h"
#include "vrcwutils.h"


/*****
 * constructor - destructor
 *****/

VRCWHost::VRCWHost(QWidget* parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   hostsModel = new QStringListModel();
   //Testing
   //
   hosts << "adventure" << "discovery" << "resolution" << "endeavour"
         << "santa_maria" << "nina" << "pinta"
         << "pamir" << "passat" << "gorch_fock" << "mayflower"
         << "vasa" << "fala";
   //
   hostsModel->setStringList(hosts);
   ui.hostsListView->setModel(hostsModel);

   //Variablen setzen
   calcNumHosts = 0;
   natOs = getNativeOs();
   expertMode = false;

   //native OS in ComboBox setzen und disable oder enable
   //abhaengig von expertMode
   setGuiOs();

   //disable execModeComboBox
   setEnableExecMode(hosts.size());
}

VRCWHost::~VRCWHost()
{

}


/*****
 * public functions
 *****/

//Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
//Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWHost::processGuiInput(const int& index,
      const QList<VRCWBase*>& vrcwList)
{
   const int ERROR_1 = 401;
   const int ERROR_2 = 402;
   const int PART_OK = 409;
   const int OK = 4;

   int success = DEF_ERROR;

   QVector<QString> hHandled = readHostLineEdit();

   //Ueberpruefen, ob noch hosts in der Eingabezeile eingetragen sind
   if (!hHandled.empty())
   {
      QMessageBox msgBox(this);
      msgBox.setIcon(QMessageBox::Question);
      msgBox.setInformativeText(tr("You have entered some hosts in the \"Add\" line\n"
            "but you didn't press the Add button.\n\n"
            "Do you want to add them to the list of hosts?"));
      msgBox.setWindowTitle(tr("Configuration"));
      QPushButton* addButton = msgBox.addButton(tr("Add"), QMessageBox::YesRole);
      QPushButton* noButton = msgBox.addButton(QMessageBox::Cancel);
      msgBox.setDefaultButton(addButton);
      msgBox.exec();

      if (msgBox.clickedButton() == addButton)
      {
         VRCWHost::add();
      }

      success = ERROR_1;
   }
   else
   {
      success = PART_OK;
   }

   if (success == PART_OK)
   {
      //Ueberpruefen, ob Anzahl der eingegebenen hosts der Mindestanzahl
      //fuer die gewaehlte Projektion entspricht
      if (hosts.size() < calcNumHosts)
      {
         QString message = tr("You specified fewer hosts\n"
               "than you need for the\n"
               "selected projection type.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_2;
      }
      else
      {
         success = OK;
      }
   }

   //sobald keine Fehler in der Eingabe sind:
   //- wird die Liste der hostnames an VRCWHostProjection uebergeben
   //- wird os und exec an VRCWFinal uebergeben
   //
   if (success == OK)
   {
      int finalIndex = vrcwList.size() - 2;
      VRCWHostProjection* hostProjection =
            dynamic_cast<VRCWHostProjection*> (vrcwList[index + 1]);
      VRCWTrackingHw* trackingHw =
            dynamic_cast<VRCWTrackingHw*> (vrcwList[index + 2]);
      VRCWFinal* final = dynamic_cast<VRCWFinal*> (vrcwList[finalIndex]);

      hostProjection->setGuiHostData(hosts);

      opSys os = getGuiOs();
      trackingHw->setOS_GUI(os);
      final->setHost(os, getGuiExec());
   }

   return success;
}

//Liste der projection von VRCWProjectionDim... uebernehmen
//und daraus die minimal erforderliche Anzahl an hosts bestimmen
//
void VRCWHost::setCalcNumHosts(const QStringList& projection)
{
   calcNumHosts = projection.size();
}


/*****
 * private functions
 *****/

//Auslesen des GUI
//
opSys VRCWHost::getGuiOs() const
{
   return strToOpSys(ui.osComboBox->currentText());
}

execMode VRCWHost::getGuiExec() const
{
   return strToExecMode(ui.execComboBox->currentText());
}

//read hosts from input line (lineEdit)
//
QVector<QString> VRCWHost::readHostLineEdit() const
{
   return ((ui.hostsLineEdit->text()).split(
         QRegExp("\\s+"), QString::SkipEmptyParts)).toVector();
}

//OS-Version des Programms ausgeben
//
opSys VRCWHost::getNativeOs() const
{
   opSys nOs;

//   #ifdef Q_OS_LINUX
   #ifdef __linux__
   {
      nOs = Linux;
   }
//   #elif Q_OS_WIN32
   #else
   {
      nOs = Windows;
   }
   #endif

   return nOs;
}

//GuiOs setzen abhaengig vom expertMode
//
void VRCWHost::setGuiOs() const
{
   if (expertMode)
   {
      ui.osComboBox->setEnabled(true);
   }
   else
   {
      int index = ui.osComboBox->findText(opSysToStr(natOs));
      ui.osComboBox->setCurrentIndex(index);
      ui.osComboBox->setEnabled(false);
   }
}

//enable/disable Execution Mode abhaengig der definierten hosts
//
void VRCWHost::setEnableExecMode(const int& size)
{
   if (size > 1)
   {
      ui.execComboBox->setEnabled(true);
   }
   else
   {
      ui.execComboBox->setEnabled(false);
   }
}


/*****
 * private slots
 *****/

//set execution mode depending on OS choice
//
void VRCWHost::os_exec() const
{
   switch (getGuiOs())
   {
      case Linux:
      {
         ui.execComboBox->clear();
         ui.execComboBox->addItem(execModeToStr(ssh));
         ui.execComboBox->addItem(execModeToStr(covremote));
         ui.execComboBox->addItem(execModeToStr(CovDaemon));
         ui.execComboBox->addItem(execModeToStr(rsh));
         break;
      }
      case Windows:
      {
         ui.execComboBox->clear();
         ui.execComboBox->addItem(execModeToStr(covremote));
         ui.execComboBox->addItem(execModeToStr(CovDaemon));
         break;
      }
   }
}

//Add Button
//
void VRCWHost::add()
{
   //_Voraussetzung:_
   //jeder host hat eine IP
   //max. einen/keinen FQDN
   //max. einen/keinen shortname/alias/hostname == FQDN ohne Domaenenanteil

   //mit hostname einen lookup machen -> IP-Adresse
   //mit IP-Adresse eine reverseLookup machen -> FQDN, falls vorhanden
   //bei FQDN den Domaenenanteil abschneiden und mit hostname
   // vergleichen
   //den gefundenen Namen mit der gefundenen IP-Adresse vergleichen
   // und damit sicherstellen, dass der hostname auch existiert
   // z.B. unter linux ergibt ein lookup mit 1111
   // ( QHostInfo::fromName(1111) ) eine IP-Adresse
   //den hostname, den FQDN und die IP-Adresse mit hosts vergleichen


   const QString ERRCODE_2 = "hostInList";


   QVector<QString> hostsInput = readHostLineEdit();

   if (!hostsInput.empty())
   {
      for (QVector<QString>::size_type i = 0; i < hostsInput.size(); ++i)
      {
         //hostname, FQDN or IP address for the host name lookup
         QString hostToLookup = hostsInput[i].toLower();

         //wird true gesetzt, sobald entschieden ist, dass der eingegebene
         //hostname in die Liste der zu konfigurierenden Hosts eingetragen
         //werden kann
         bool appendHost = false;

         if (expertMode)
         {
            //im expertMode wird der/die eingegebene Name/IP-Adresse
            //einfach ohne irgendwelche Ueberpruefung uebernommen
            appendHost = true;
         }
         else
         {
            //verification of the input
            hIPLkpVal* hostIPLookupValue = hostIPLookup(hostToLookup);

            //errorCode is used for different error messages or hints
            QString errorCode = hostIPLookupValue->errCode;

            //if no error, check and append to hosts
            if (errorCode.isEmpty())
            {
               //den eingegebenen Namen, den hostname, den FQDN
               //und die IP-Adresse mit hosts vergleichen
               //hostToLookup == hostIPLookupValue->toLookup
               if (!hosts.contains(hostIPLookupValue->toLookup)
                     && !hosts.contains(hostIPLookupValue->hostname_2)
                     && !hosts.contains(hostIPLookupValue->fqdn_2)
                     && !hosts.contains(hostIPLookupValue->ipAddr_1))
               {
                  appendHost = true;
               }
               else
               {
                  errorCode = ERRCODE_2;
               }
            }

            //handle errorCode
            if (errorCode == ERRCODE_2)
            {
               QString message = tr("The hostname\n\n" "%1" "\n\n"
                     "is already in the list.").arg(hostToLookup);
               QMessageBox::warning(this, tr("Configuration"), message,
                     QMessageBox::Ok);
            }
            else if (errorCode == HIPLKP_EC_1)
            {
               QString message = tr("The hostname\n\n" "%1" "\n\n"
                     "doesn't seem to be valid.\n"
                     "Please make sure the hostname is in your hosts file\n"
                     "or can be validated through a DNS lookup.")
                     .arg(hostToLookup);
               QMessageBox::warning(this, tr("Configuration"), message,
                     QMessageBox::Ok);
            }
         }

         if (appendHost)
         {
            hosts.append(hostToLookup);

            //enable/disable execMode ComboBox
            int size = hosts.size();
            setEnableExecMode(size);
         }
      }

      hosts.sort();
      hostsModel->setStringList(hosts);
      ui.hostsLineEdit->clear();
   }
}

//Remove Button
//
void VRCWHost::remove()
{
   QItemSelectionModel* selModHostsLV = ui.hostsListView->selectionModel();

   while (selModHostsLV->hasSelection())
   {
      //Generate list of selected indexes
      //and select only the first index
      QModelIndex firstIndex = selModHostsLV->selectedIndexes().first();

      //Identify first selected entry
      QString selEntry =
            hostsModel->data(firstIndex, Qt::DisplayRole).toString();

      //Request
      QMessageBox msgBox(this);
      msgBox.setIcon(QMessageBox::Question);
      msgBox.setInformativeText(tr("Do you want to remove\n\n" "%1" "\n\n"
            "from the list of hostnames?").arg(selEntry));
      msgBox.setWindowTitle(tr("Configuration"));
      QPushButton* removeButton = msgBox.addButton(tr("Remove"), QMessageBox::YesRole);
      QPushButton* noButton = msgBox.addButton(QMessageBox::Cancel);
      msgBox.setDefaultButton(removeButton);
      msgBox.exec();

      if (msgBox.clickedButton() == removeButton)
      {
         //Delete entry
         hosts.removeAll(selEntry);
         hostsModel->removeRow(firstIndex.row());

         //enable/disable execMode ComboBox
         int size = hosts.size();
         setEnableExecMode(size);
      }
      else
      {
         //Deselect selection
         selModHostsLV->select(firstIndex, QItemSelectionModel::Deselect);
      }
   }
}

//Setzen der Variable expertMode und guiOs setzen
//
void VRCWHost::setExpertMode(const bool& changed)
{
   expertMode = changed;

   setGuiOs();
}
