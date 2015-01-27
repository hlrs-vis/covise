/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#if QT_VERSION >= 0x040400
#include <QFormLayout>
#endif
#include <QGridLayout>
#include <QStyleFactory>
#include <QDialogButtonBox>
#include <QAction>

#include "MESessionSettings.h"
#include "MEUserInterface.h"
#include "handler/MEMainHandler.h"

/*!
   \class MESessionSettings
   \brief This class provides a dialog for m_settings session parameters
*/

MESessionSettings::MESessionSettings(QWidget *parent, Qt::WindowFlags f)
    : QDialog(parent, f)
{
    // set a proper title
    setWindowIcon(MEMainHandler::instance()->pm_logo);
    setWindowTitle(MEMainHandler::instance()->generateTitle("Settings"));

    // make the main layout
    QVBoxLayout *fbox = new QVBoxLayout(this);

    // create the main content
    createFormLayout(fbox);

    // create the dialog buttons
    QDialogButtonBox *bb = new QDialogButtonBox();
    bb->addButton("Ok", QDialogButtonBox::AcceptRole);
    bb->addButton("Cancel", QDialogButtonBox::RejectRole);
    QPushButton *resetButton = bb->addButton("Defaults", QDialogButtonBox::ResetRole);
    connect(bb, SIGNAL(accepted()), this, SLOT(save()));
    connect(bb, SIGNAL(rejected()), this, SLOT(cancel()));
    connect(resetButton, SIGNAL(clicked()), this, SLOT(resetValues()));
    fbox->addWidget(bb);

    // user can close the window pressing ESC
    QAction *m_escape_a = new QAction("Escape", this);
    m_escape_a->setShortcut(Qt::Key_Escape);
    connect(m_escape_a, SIGNAL(triggered()), this, SLOT(cancel()));
    addAction(m_escape_a);

    QAction *closeAction = new QAction("close", this);
    closeAction->setShortcut(QKeySequence::Close);
    connect(closeAction, SIGNAL(triggered(bool)), this, SLOT(close()));
    this->addAction(closeAction);

    // init options
    QStringList styles = QStyleFactory::keys();
    if (!styles.contains("Default"))
        styles.append("Default");
    styles.sort();

    qtStyleComboBox->addItems(styles);
    if (QString(MEMainHandler::instance()->cfg_QtStyle).isEmpty())
        qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText("Default"));
    else
        qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText(MEMainHandler::instance()->cfg_QtStyle));

    initState();
}

MESessionSettings::~MESessionSettings()
{
}

//!
//! init widgets with current values
//!
void MESessionSettings::initState()
{
    errorHandlingCheckBox->setChecked(MEMainHandler::instance()->cfg_ErrorHandling);
    storeWindowConfigBox->setChecked(MEMainHandler::instance()->cfg_storeWindowConfig);
    hideUnusedModulesBox->setChecked(MEMainHandler::instance()->cfg_HideUnusedModules);
    autoConnectBox->setChecked(MEMainHandler::instance()->cfg_AutoConnect);
    browserBox->setChecked(!MEMainHandler::instance()->cfg_TopLevelBrowser);
    developerModeCheckBox->setChecked(MEMainHandler::instance()->cfg_DeveloperMode);
    //imbeddedRenderBox->setChecked(MEMainHandler::instance()->cfg_ImbeddedRenderer);

    qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText(MEMainHandler::instance()->cfg_QtStyle));

    highlightColorEdit->setText(MEMainHandler::instance()->cfg_HighColor);
    autoSaveTimeEdit->setText(QString::number(MEMainHandler::instance()->cfg_AutoSaveTime));
}

#if QT_VERSION >= 0x040400

#define addSetting(type, widget, labeltext, tooltip) \
    do                                               \
    {                                                \
        QLabel *label = new QLabel(labeltext);       \
        label->setToolTip(tooltip);                  \
        widget = new type();                         \
        widget->setToolTip(tooltip);                 \
        grid->addRow(label, widget);                 \
    } while (0)

#else

#define addSetting(type, widget, labeltext, tooltip) \
    do                                               \
    {                                                \
        QLabel *label = new QLabel(labeltext);       \
        widget = new type();                         \
        grid->addWidget(label, i, 0);                \
        grid->addWidget(widget, i, 1);               \
        label->setToolTip(tooltip);                  \
        widget->setToolTip(tooltip);                 \
    } while (0)

#endif

//!
//! make a grid layout for setting options
//!
void MESessionSettings::createFormLayout(QVBoxLayout *mainLayout)
{
    QGroupBox *container = NULL;

    container = new QGroupBox("General", this);
    container->setFont(MEMainHandler::s_boldFont);
    mainLayout->addWidget(container);

#if QT_VERSION >= 0x040400
    QFormLayout *grid = new QFormLayout();
#else
    QGridLayout *grid = new QGridLayout();
#endif

    int i = 0;
    addSetting(QComboBox, qtStyleComboBox, "Qt style",
               "Qt widget style for the map editor");
    i++;
    addSetting(QCheckBox, storeWindowConfigBox, "Restore window layout",
               "Enabled: restore size, position and docking state of all windows\nDisabled: do not restore window layout");
    i++;
    addSetting(QCheckBox, developerModeCheckBox, "Developer features",
               "Enable features that are useful only to developers");
    i++;
    addSetting(QCheckBox, errorHandlingCheckBox, "Error message dialogs",
               "Enabled: pop up a dialog box for each error message\nDisabled: show error messages in message window");
    i++;
    addSetting(QCheckBox, browserBox, "Embedded browsers",
               "Enabled: Filebrowsers and color maps are embedded into Module Parameter windows and Control Panel \nDisabled: Filebrowsers and color maps appear as toplevel window");
    i++;
    //addSetting(QCheckBox, imbeddedRenderBox, "Embedded ViNCE Renderer",
    //  "Enabled: ViNCE renderer is embedded into the MEMainHandler::instance()\nDisabled: ViNCE renderer appears as a toplevel window");i++;
    addSetting(QCheckBox, autoConnectBox, "Auto connect hosts",
               "Enabled: automatically connect to host or partner if connection mode is ssh or RemoteDaemon \nDisabled: always prompt the user");
    i++;
    container->setLayout(grid);

    container = new QGroupBox("Saving", this);
    container->setFont(MEMainHandler::s_boldFont);
    mainLayout->addWidget(container);
#if QT_VERSION >= 0x040400
    grid = new QFormLayout();
#else
    grid = new QGridLayout();
#endif

    i = 0;
    addSetting(QLineEdit, autoSaveTimeEdit, "Autosave interval",
               "Time interval for automatic saving (seconds)");
    i++;
    container->setLayout(grid);

    container = new QGroupBox("Visual Programming", this);
    container->setFont(MEMainHandler::s_boldFont);
    mainLayout->addWidget(container);
#if QT_VERSION >= 0x040400
    grid = new QFormLayout();
#else
    grid = new QGridLayout();
#endif

    i = 0;
    addSetting(QCheckBox, hideUnusedModulesBox, "Hide unused modules",
               "Enabled: module browser only shows recently used modules of each category\nDisabled: module browser shows all modules of a category");
    i++;
    addSetting(QLineEdit, highlightColorEdit, "Highlight color",
               "Color name for highlighting module ports and connections");
    i++;
    container->setLayout(grid);
}

//!
//! save the settings
//!
void MESessionSettings::save()
{

    if (MEMainHandler::instance()->cfg_ErrorHandling != errorHandlingCheckBox->isChecked())
        MEMainHandler::instance()->cfg_ErrorHandling = errorHandlingCheckBox->isChecked();
    if (MEMainHandler::instance()->cfg_storeWindowConfig != storeWindowConfigBox->isChecked())
        MEMainHandler::instance()->cfg_storeWindowConfig = storeWindowConfigBox->isChecked();
    if (MEMainHandler::instance()->cfg_DeveloperMode != developerModeCheckBox->isChecked())
    {
        MEMainHandler::instance()->cfg_DeveloperMode = developerModeCheckBox->isChecked();
        MEMainHandler::instance()->developerModeHasChanged();
    }
    if (MEMainHandler::instance()->cfg_AutoConnect != autoConnectBox->isChecked())
        MEMainHandler::instance()->cfg_AutoConnect = autoConnectBox->isChecked();
    if (MEMainHandler::instance()->cfg_TopLevelBrowser == browserBox->isChecked())
        MEMainHandler::instance()->cfg_TopLevelBrowser = !browserBox->isChecked();

    if (MEMainHandler::instance()->cfg_HideUnusedModules != hideUnusedModulesBox->isChecked())
    {
        MEMainHandler::instance()->cfg_HideUnusedModules = hideUnusedModulesBox->isChecked();
        MEUserInterface::instance()->hideUnusedModulesHasChanged();
    }
    /*if (MEMainHandler::instance()->cfg_ImbeddedRenderer     != imbeddedRenderBox->isChecked())
   {
      MEMainHandler::instance()->cfg_ImbeddedRenderer      = imbeddedRenderBox->isChecked();
      MEMainHandler::instance()->rendererModeHasChanged();
   }*/

    if (QString(MEMainHandler::instance()->cfg_QtStyle) != qtStyleComboBox->currentText())
    {
        if (qtStyleComboBox->currentText() == "Default")
            MEMainHandler::instance()->cfg_QtStyle = "";
        else
            MEMainHandler::instance()->cfg_QtStyle = qtStyleComboBox->currentText();
    }

    if (QString::number(MEMainHandler::instance()->cfg_AutoSaveTime) != autoSaveTimeEdit->text())
        MEMainHandler::instance()->cfg_AutoSaveTime = autoSaveTimeEdit->text().toInt();

    if (QString(MEMainHandler::instance()->cfg_HighColor) != highlightColorEdit->text())
        MEMainHandler::instance()->cfg_HighColor = highlightColorEdit->text();

    accept();
}

//!
//! restore the settings
//!
void MESessionSettings::cancel()
{
    initState();
    reject();
}

//!
void MESessionSettings::resetValues()
{
    autoSaveTimeEdit->setText(QString::number(120));

    errorHandlingCheckBox->setChecked(false);

    hideUnusedModulesBox->setChecked(true);

    autoConnectBox->setChecked(true);

    storeWindowConfigBox->setChecked(false);

    //imbeddedRenderBox->setChecked(false);

    highlightColorEdit->setText("red");

    // means default
    qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText("Default"));

    browserBox->setChecked(true);
}
