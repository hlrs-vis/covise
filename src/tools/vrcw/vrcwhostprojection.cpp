#include "vrcwhostprojection.h"

#include <QMessageBox>
#include <QStringBuilder>

#include "vrcwfinal.h"
#include "tablemodel.h"


/*****
 * constructor - destructor
 *****/

VRCWHostProjection::VRCWHostProjection(QWidget* parent) :
   QWidget(parent), NOT_SEL("--not selected--")
{
   ui.setupUi(this);

   //Variablen setzen
   dataChanged = false;

   //Setup TableModel
   hostProjectionModel = new ListStrListTModel();
   tableHeader << "Projection" << "Eye" << "Host";
   for (QStringList::size_type i = 0; i < tableHeader.size(); ++i)
   {
      hostProjectionModel->setHeaderData(i, Qt::Horizontal, tableHeader[i],
            Qt::DisplayRole);
   }
   ui.tableView->setModel(hostProjectionModel);
   ui.tableView->setSortingEnabled(true);
   ui.tableView->sortByColumn(0, Qt::AscendingOrder);
   ui.tableView->resizeColumnsToContents();
}

VRCWHostProjection::~VRCWHostProjection()
{

}


/*****
 * public functions
 *****/

// Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
// Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWHostProjection::processGuiInput(
      const QList<VRCWBase*>& vrcwList)
{
   const int ERROR_1 = 51;
   const int ERROR_2 = 52;
   const int OK = 5;

   int success = DEF_ERROR;

   //Ueberpruefen, ob alle Projektionen konfiguriert wurden
   if (ui.projectionComboBox->count() > 0)
   {
      QString message = tr("Please configure every projection!");
      QMessageBox::warning(this, tr("Configuration"), message,
            QMessageBox::Ok);

      success = ERROR_1;
   }
   //Ueberpruefen, ob ein COVISE GUI host gewaehlt wurde
   else if (ui.coviseGuiHostComboBox->currentText() == NOT_SEL)
   {
      QString message = tr("Please select a host on which "
            "the COVISE GUI is running!");
      QMessageBox::warning(this, tr("Configuration"), message,
            QMessageBox::Ok);

      success = ERROR_2;
   }
   else
   {
      success = OK;
   }

   //sobald keine Fehler in der Eingabe sind:
   //- wird covGuiHost, masterHost und hostProjection an VRCWFinal uebergeben
   //
   if (success == OK)
   {
      int finalIndex = vrcwList.size() - 2;
      VRCWFinal* final = dynamic_cast<VRCWFinal*> (vrcwList[finalIndex]);

      QString covGuiHost = ui.coviseGuiHostComboBox->currentText();
      QList<QStringList> hostProjection =
            hostProjectionModel->getListStrList();

      final->setHostProjection(covGuiHost, hostProjection);
   }


   return success;
}

//Entgegennehmen der ProjectionData
//
void VRCWHostProjection::setGuiProjectionData(const QStringList& projection)
{
   if (projectionData != projection)
   {
      projectionData = projection;
      dataChanged = true;
   }
   //hier wird nur dataChanged gesetzt (von VRCWProjectionDim...)
   //setGui() wird nur bei setGuiHostData() aufgerufen
}

//Entgegennehmen der HostData
//
void VRCWHostProjection::setGuiHostData(const QStringList& host)
{
   if (hostData != host)
   {
      hostData = host;
      dataChanged = true;
   }

   //setGui() wird nur hier ausgeloest (von VRCWHost), weil diese Funktion
   //direkt vor VRCWHostProjection durchlaufen wird
   setGui();
}


/*****
 * private functions
 *****/

//Setzen der host-projection-Eintraege in der GUI
//
void VRCWHostProjection::setGui()
{
   if (dataChanged)
   {
      //Setzen der Eintraege in der projectionComboBox
      ui.projectionComboBox->clear();
      for (QStringList::size_type i = 0; i < projectionData.size(); ++i)
      {
         ui.projectionComboBox->addItem(projectionData[i]);
      }

      //Setzen der Eintraege in der hostComboBox
      ui.hostComboBox->clear();
      for (QStringList::size_type i = 0; i < hostData.size(); ++i)
      {
         ui.hostComboBox->addItem(hostData[i]);
      }

      //Setzen der Eintraege in der guiHostComboBox
      ui.coviseGuiHostComboBox->clear();
      ui.coviseGuiHostComboBox->addItem(NOT_SEL);
      for (QStringList::size_type i = 0; i < hostData.size(); ++i)
      {
         ui.coviseGuiHostComboBox->addItem(hostData[i]);
      }

      //Alle Eintraege aus der Tabelle entfernen
      hostProjectionModel->clearData();
      dataChanged = false;
   }
}


/*****
 * private slots
 *****/

//Add Button
//
void VRCWHostProjection::add() const
{
   if (ui.projectionComboBox->count() > 0)
   {
      //Eintrag fuer die Tabelle und fuer die masterRendererComboBox
      //aus der projection- und hostComboBox generieren
      QString projectionEye = ui.projectionComboBox->currentText();
      QString host = ui.hostComboBox->currentText();
      QStringList projectionEyeList = projectionEye.split(" ");
      QString projection = projectionEyeList[0];
      QString eye = projectionEyeList[2];
      QStringList projectionEyeHost =
            (QStringList() << projection << eye << host);

      //Eintrag in Tabelle am Ende einfuegen
      hostProjectionModel->appendData(projectionEyeHost);

      //Eintraege aus den Comboboxen entfernen
      ui.projectionComboBox->removeItem(ui.projectionComboBox->currentIndex());
      ui.hostComboBox->removeItem(ui.hostComboBox->currentIndex());
   }
}

//Remove Button
//
void VRCWHostProjection::remove()
{
   QItemSelectionModel* selModelTV = ui.tableView->selectionModel();

   while (selModelTV->hasSelection())
   {
      //List of selected indexes
      //(for one row a number of tableHeader.size() indexes are stored)
      QModelIndexList indexes = selModelTV->selectedIndexes();
      //select only the first index
      //and returns the row this model index refers to
      int firstIndexRow = indexes.first().row();

      //Eintraege fuer die ComboBoxen aus der Tabelle generieren
      QList<QStringList> tableData = hostProjectionModel->getListStrList();
      QStringList selRowEntry = tableData[firstIndexRow];
      QString selRowEntryHost = selRowEntry[2];
      QString selRowEntryEye = selRowEntry[1] % " Eye";
      if (selRowEntry[1] == "Both")
      {
         selRowEntryEye += "s";
      }
      QString selRowEntryProjection =
            selRowEntry[0] % " - " % selRowEntryEye;

      //Abfrage, ob loeschen des Eintrags erwuenscht
      QString message = tr("Do you want to remove\n\n"
         "Projection: " "%1" "\n" "Host: " "%2" "\n\n"
         "from the list?").arg(selRowEntryProjection).arg(selRowEntryHost);
      int ret = QMessageBox::question(this, tr("Configuration"), message,
            QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);

      if (ret == QMessageBox::Yes)
      {
         //Eintraege in projection- und hostComboBox einfuegen und sortieren
         QAbstractItemModel* projectCBModel =
               ui.projectionComboBox->model();
         ui.projectionComboBox->addItem(selRowEntryProjection);
         projectCBModel->sort(0);
         QAbstractItemModel* hostCBModel = ui.hostComboBox->model();
         ui.hostComboBox->addItem(selRowEntryHost);
         hostCBModel->sort(0);

         //Eintrag aus der Tabelle entfernen
         hostProjectionModel->removeRow(firstIndexRow);
      }
      else
      {
         //deselect the whole row
         //tableHeader.size() == Number of rows
         for (QStringList::size_type i = 0; i < tableHeader.size(); ++i)
         {
            selModelTV->select(indexes[i], QItemSelectionModel::Deselect);
         }
      }
   }
}
