#include "vrcwstart.h"

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QMessageBox>
#include <QUuid>
#include <cstdlib>
#include <cstdio>

#include "datatypes.h"
#include "vrcwfinal.h"


/*****
 * constructor - destructor
 *****/

VRCWStart::VRCWStart(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //Set background of textEdit
   QPalette readOnlyPalette = ui.hintStorLocTextEdit->palette();
   QColor mainWindowBgColor = palette().color(QPalette::Window);
   readOnlyPalette.setColor(QPalette::Base, mainWindowBgColor);
   ui.hintStorLocTextEdit->setPalette(readOnlyPalette);

   //get environment variable $COVISEDIR
   coDir = getEnvVariable("COVISEDIR");

   //set variables
   coDirSet = false;
   coConfDirWritable = false;
   expertMode = false;

   //setup GUI (variables must set before setupGui()
   setupGui();
   lineEdit_exec(ui.newConfNameLineEdit->text());

   //Validator for filename
   QRegExp rxFilename("([\\w\\-\\.]+)");
   ui.newConfNameLineEdit->setValidator(new QRegExpValidator(rxFilename,
         ui.newConfNameLineEdit));
}

VRCWStart::~VRCWStart()
{

}


/*****
 *public functions
 *****/
int VRCWStart::processGuiInput(const QList<VRCWBase*>& vrcwList)
{
   const int OK = 1;
   const int WARN_01 = 1;
   const int WARN_02 = 2;
   const int WARN_03 = 3;
   const int WARN_04 = 4;
   const int WARN_05 = 5;

   const int ERROR_01 = 101;
   const int ERROR_02 = 102;
   const int ERROR_03 = 103;
   const int ERROR_04 = 104;
   const int ERROR_05 = 105;

   int success = DEF_ERROR;
   int awarning = DEF_WARN;

   if (expertMode)
   {
      //check if configuration name is given
      if (coConfName.isEmpty())
      {
         awarning = WARN_02;
      }
      //check if storage location for config files is given
      else if (coConfTmpDir.isEmpty())
      {
         awarning = WARN_04;
      }
      else
      {
         success = OK;
      }
   }
   else
   {
      //stop if COVISEDIR environment variable is not set
      if (!coDirSet)
      {
         awarning = WARN_01;
      }
      else
      {
         //test if configuration name is given and file exists
         if (coConfName.isEmpty())
         {
            awarning = WARN_02;
         }
         else if (QFile::exists(
               QDir::toNativeSeparators(coConfDir + "/" + coConfName)))
         {
            awarning = WARN_03;
         }
         else
         {
            success = OK;
         }

         //test if coConfDir is writeable
         //no write permission is checked at initial startup
         if (coConfDirWritable)
         {
            success = OK;
         }

         //when we can choose the temp location, the newConfNameWidget is visible
         if (!coConfDirWritable && coConfTmpDir.isEmpty())
         {
            awarning = WARN_04;
         }

         if (!coConfTmpDir.isEmpty())
         {
            if (testWritePerm(coConfTmpDir))
            {
               success = OK;
            }
            else
            {
               //coConf is not writeable
               awarning = WARN_05;
            }
         }
      }
   }


   //warnings
   //
   switch (awarning)
   {
      case WARN_01:
      {
         QString message = tr("The environment variable COVISEDIR is not set "
               "or not set correctly.\n\n"
               "Please choose the path to the COVISE directory.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.tmpStorLocPushButton->setFocus();
         success = ERROR_01;
         break;
      }
      //new configname is not given
      case WARN_02:
      {
         QString message = tr("A name for the new configuration is not "
               "given.\n\n"
               "Please set a name for the configuration.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.newConfNameLineEdit->setFocus();
         success = ERROR_02;
         break;
      }
      //new configname exists
      case WARN_03:
      {
         QString message = tr("A config file with the specified name exists "
               "in the default location.\n\n"
               "Please set another name for the configuration.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.newConfNameLineEdit->selectAll();
         ui.newConfNameLineEdit->setFocus();
         success = ERROR_03;
         break;
      }
      //coConfDir not writeable and coConfTmpDir not specified
      case WARN_04:
      {
         QString message = tr("You have not specified a temporary "
               "location.\n\n"
               "Please choose a location were you have the permission "
               "to write.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_04;
         break;
      }
      //temporary save location is not writeable
      case WARN_05:
      {
         QString message = tr("You have no permission to write to the "
               "temporary location.\n\n"
               "Please choose a location were you have the permission "
               "to write.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.tmpStorLocPushButton->setFocus();
         success = ERROR_05;
         break;
      }
      case DEF_WARN:
      {
         //do nothing
         break;
      }
      default:
      {
          qDebug() << "WarningCode can't be evaluated";
          break;
      }
   }

   //wird bei Anzeige dieser Klasse auf "Next>" gedrueckt:
   //(sobald keine Fehler in der Eingabe sind:)
   //- werden die Werte ausgelesen
   //  und an VRCWFinal uebergeben

   if (success == OK)
   {
      configVal* coVal = new configVal();

      //set variables
      coVal->coConfDir = coConfDir;
      coVal->coConfDirWritable = coConfDirWritable;
      coVal->coConfName = coConfName;
      coVal->projName = getGuiProjName();
      coVal->coConfTmpDir = coConfTmpDir;

      //Uebergabe an final
      int finalIndex = vrcwList.size() - 2;
      VRCWFinal* final = dynamic_cast<VRCWFinal*> (vrcwList[finalIndex]);

      final->setstart(coVal);
   }


   return success;
}

//Auslesen des Namens des Projektes (aus dem der config-Name erzeugt wird)
//
QString VRCWStart::getGuiProjName() const
{
   return ui.newConfNameLineEdit->text();
}



/*****
 *private functions
 *****/

//setup GUI
//
void VRCWStart::setupGui()
{
   if (coDir.isEmpty() ||
         !QFile::exists(coDir + "/config/config.xml"))
   {
      setGuiCoDirFalse();
      coConfDir = "";
      coDirSet = false;
   }
   else
   {
      setGuiCoDirTrue();
      coConfDir = QDir::toNativeSeparators(coDir + "/config");
      ui.defStorLocLabel->setText(coConfDir);
      ui.newConfNameLineEdit->setFocus();
      coDirSet = true;
   }

   //test for write permission in $COVISEDIR/config
   bool writePerm = false;

   if (!coConfDir.isEmpty())
   {
      writePerm = testWritePerm(coConfDir);
   }

   if (writePerm)
   {
      ui.tmpStorLocWidget->hide();
      coConfDirWritable = true;
   }
   else if (!coConfDir.isEmpty())
   {
      //show warning about write permission
      ui.tmpStorLocWidget->show();
      ui.tmpStorLocPushButton->show();
      ui.tmpStorLocPushButton->setText("Choose temporary location");
      ui.tmpStorLocLabel->show();
      ui.tmpStorLocLabel->setText(coConfTmpDir);
   }
}

//setup GUI for expertMode
//
void VRCWStart::setupGuiExpert()
{
   //in ExpertMode we use the lineEdit for the new configuration name,
   //a description what to do, the PushButton for choosing a directory to
   //save the config an the line with the complete config name to use
   //
   //we do not set or check coConfDir
   coConfDir = "";
   coDirSet = false;
   coConfDirWritable = false;

   //set label with complete config name
   ui.tmpStorLocLabel->setText(coConfTmpDir);

   //setupGui
   ui.newConfNameWidget->show();
   ui.defStorLocWidget->hide();
   ui.tmpStorLocWidget->show();
   ui.attentionLabel->hide();
   ui.tmpStorLocPushButton->show();
   ui.tmpStorLocPushButton->setText("Where to store the config");
   ui.tmpStorLocLabel->show();
   ui.useConfigWidget->show();

   //ui.hintStorLocTextEdit->setText("Please choose a temporary location "
   //      "for the config files and move them manually "
   //      "to the default location.");

   ui.hintStorLocTextEdit->setText("<!DOCTYPE HTML PUBLIC "
         "\"-//W3C//DTD HTML 4.0//EN\" "
         "\"http://www.w3.org/TR/REC-html40/strict.dtd\">"
         "<html>"
         "<head><meta name=\"qrichtext\" content=\"1\"/>"
         "<style type=\"text/css\">p, li { white-space: pre-wrap; }"
         "</style>"
         "</head>"
         "<body style=\" font-family:'Oxygen-Sans'; font-size:10pt; "
         "font-weight:400; font-style:normal;\">"
         "<p align=\"center\" style=\" margin-top:5px; margin-bottom:5px; "
         "margin-left:0px; margin-right:0px; -qt-block-indent:0; "
         "text-indent:0px;\">"
         "Please choose a temporary location for the config files and move them "
         "manually to the default location."
         "</p>"
         "</body></html>");

   ui.hintStorLocTextEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
   ui.hintStorLocTextEdit->adjustSize();
}

//set the GUI if coDir is empty or not set correctly
//
void VRCWStart::setGuiCoDirFalse() const
{
   ui.newConfNameWidget->hide();
   ui.defStorLocWidget->hide();
   ui.useConfigWidget->hide();
   ui.tmpStorLocPushButton->show();
   ui.tmpStorLocPushButton->setText("Choose path to COVISE");
   ui.tmpStorLocLabel->hide();
   ui.attentionLabel->show();
   ui.attentionLabel->setText("<font color='red'>Attention: "
         "Environment variable COVISEDIR not set or not set "
         "correctly!</font");

   //ui.hintStorLocTextEdit->setText("Please choose the path to the COVISE
   //     "directory manually.\n"
   //      "This directory contains directories like bin, config, share."

   ui.hintStorLocTextEdit->setText("<!DOCTYPE HTML PUBLIC "
         "\"-//W3C//DTD HTML 4.0//EN\" "
         "\"http://www.w3.org/TR/REC-html40/strict.dtd\">"
         "<html>"
         "<head><meta name=\"qrichtext\" content=\"1\"/>"
         "<style type=\"text/css\">p, li { white-space: pre-wrap; }"
         "</style>"
         "</head>"
         "<body style=\" font-family:'Oxygen-Sans'; font-size:10pt; "
         "font-weight:400; font-style:normal;\">"
         "<p align=\"center\" style=\" margin-top:5px; margin-bottom:5px; "
         "margin-left:0px; margin-right:0px; -qt-block-indent:0; "
         "text-indent:0px;\">"
         "Please choose the path to the COVISE directory manually."
         "</p>"
         "<p align=\"center\" style=\" margin-top:5px; margin-bottom:5px; "
         "margin-left:0px; margin-right:0px; -qt-block-indent:0; "
         "text-indent:0px;\">"
         "Hint: This directory contains directories like bin, config, "
         "share."
         "</p>"
         "</body></html>");

   ui.hintStorLocTextEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
   ui.hintStorLocTextEdit->adjustSize();
}

//set the GUI if coConfDir is set but maybe has no permission to write
//
void VRCWStart::setGuiCoDirTrue() const
{
   ui.newConfNameWidget->show();
   ui.defStorLocWidget->show();
   ui.useConfigWidget->show();
   ui.attentionLabel->show();
   ui.attentionLabel->setText("<font color='red'>Attention: You have no "
         "permission to write to the default location!</font");

   //ui.hintStorLocTextEdit->setText("lease make sure you have write permission to the default location "
   //      "and start the Configuration Wizard again.\n"
   //      "Or choose a temporary location for the config files and move them "
   //      "manually to the default location."

   ui.hintStorLocTextEdit->setText("<!DOCTYPE HTML PUBLIC "
         "\"-//W3C//DTD HTML 4.0//EN\" "
         "\"http://www.w3.org/TR/REC-html40/strict.dtd\">"
         "<html>"
         "<head><meta name=\"qrichtext\" content=\"1\"/>"
         "<style type=\"text/css\">p, li { white-space: pre-wrap; }"
         "</style>"
         "</head>"
         "<body style=\" font-family:'Oxygen-Sans'; font-size:10pt; "
         "font-weight:400; font-style:normal;\">"
         "<p align=\"center\" style=\" margin-top:5px; margin-bottom:5px; "
         "margin-left:0px; margin-right:0px; -qt-block-indent:0; "
         "text-indent:0px;\">"
         "Please make sure you have write permission to the default location "
         "and start the Configuration Wizard again."
         "</p>"
         "<p align=\"center\" style=\" margin-top:5px; margin-bottom:5px; "
         "margin-left:0px; margin-right:0px; -qt-block-indent:0; "
         "text-indent:0px;\">"
         "Or choose a temporary location for the config files and move them "
         "manually to the default location."
         "</p>"
         "</body></html>");

   ui.hintStorLocTextEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
   ui.hintStorLocTextEdit->adjustSize();
}

//get environment variable key
//
QString VRCWStart::getEnvVariable(const QString& key) const
{
   //getenv aus <stdlib.h>

   const char* val = std::getenv(key.toStdString().c_str());

   return val == NULL ? "" : QString::fromStdString(val);
}

//test if User have permission to write in dir
//
bool VRCWStart::testWritePerm(const QString& dir)
{
    //fopen, fclose, remove aus <stdio.h>

    bool writePerm = false;

    //a more random file name
    QString uuid =
           QUuid::createUuid().toString().replace("{", "").replace("}", "");
    QString tmpfileName = "vrcw_" + uuid + ".tmp";
    std::string tmpfile =
           QDir::toNativeSeparators(dir + "/" + tmpfileName).toStdString();
    const char* fname = tmpfile.c_str();

    FILE* tmpFile = fopen(fname, "w");
    if (tmpFile != NULL)
    {
      writePerm = true;
      fclose(tmpFile);
      remove(fname);
    }

    return writePerm;
}


/*****
 * private slots
 *****/

//read lineEdit and set config name
//
void VRCWStart::lineEdit_exec(const QString& fName)
{
   if (fName.isEmpty())
   {
      ui.finalConfNameLabel->setText("config.<NAME>.xml");
      coConfName = "";
   }
   else
   {
      coConfName = "config." + fName + ".xml";
      ui.finalConfNameLabel->setText(coConfName);
   }
}

//choose COVISE directory, set coDir and execute setupGui() again
//
void VRCWStart::chooseDir_exec()
{
   QString dir = QFileDialog::getExistingDirectory(this,
         tr("Open Directory"),
         QDir::homePath(), QFileDialog::ShowDirsOnly |
         QFileDialog::DontResolveSymlinks);

   if (expertMode)
   {
      if (!dir.isEmpty())
      {
         coConfTmpDir = dir;
      }

      setupGuiExpert();
   }
   else
   {
      if (coDirSet && !dir.isEmpty())
      {
         coConfTmpDir = dir;
      }
      else
      {
         coDir = dir;

         if (!dir.isEmpty() && !QFile::exists(coDir + "/config/config.xml"))
         {
            QString message = tr("The path you have chosen was "
                  "not correct.\n\n"
                  "Please choose the path to the COVISE directory.");
            QMessageBox::warning(this, tr("Configuration"), message,
                  QMessageBox::Ok);
         }
      }

      setupGui();
   }
}

//Setzen der Variable expertMode
//
void VRCWStart::setExpertMode(const bool& changed)
{
   expertMode = changed;

   //in expertMode we use another GUI setup
   if (expertMode)
   {
      setupGuiExpert();
   }
   else
   {
      setupGui();
   }
}
