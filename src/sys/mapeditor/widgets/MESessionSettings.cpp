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
#include <QFormLayout>
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
    if (MEMainHandler::instance()->cfg_QtStyle().isEmpty())
        qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText("Default"));
    else
        qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText(MEMainHandler::instance()->cfg_QtStyle()));

    initState(bb);
}

MESessionSettings::~MESessionSettings()
{
}

void syncBools(QDialogButtonBox *dialog, covise::ConfigBool &config, QCheckBox *checkBox)
{
    checkBox->setChecked(config.value());
    if(!dialog)
        return;
    QObject::connect(dialog, &QDialogButtonBox::accepted, [&config, checkBox](){
        config = checkBox->isChecked();
    });
    config.setUpdater([checkBox](bool state)
    {
        checkBox->setChecked(state);
    });
}

void syncText(QDialogButtonBox *dialog, covise::ConfigString &config, QLineEdit *lineEdit)
{
    lineEdit->setText(config.value().c_str());
    if(!dialog)
        return;
    QObject::connect(dialog, &QDialogButtonBox::accepted, [&config, lineEdit](){
        config = lineEdit->text().toStdString();
    });
    config.setUpdater([lineEdit](const std::string &val)
    {
        lineEdit->setText(val.c_str());
    });
}

void syncInt(QDialogButtonBox *dialog, covise::ConfigInt &config, QLineEdit *lineEdit)
{
    lineEdit->setText(QString::number(config.value()));
    if(!dialog)
        return;
    QObject::connect(dialog, &QDialogButtonBox::accepted, [&config, lineEdit](){
        config = lineEdit->text().toInt();
    });
    config.setUpdater([lineEdit](const int64_t &val)
    {
        lineEdit->setText(QString::number(val));
    });
}
//!
//! init widgets with current values
//!
void MESessionSettings::initState(QDialogButtonBox *dialog)
{
    syncBools(dialog, *MEMainHandler::instance()->cfg_ErrorHandling, errorHandlingCheckBox);
    syncBools(dialog, *MEMainHandler::instance()->cfg_HideUnusedModules, hideUnusedModulesBox);
    syncBools(dialog, *MEMainHandler::instance()->cfg_storeWindowConfig, storeWindowConfigBox);
    syncBools(dialog, *MEMainHandler::instance()->cfg_AutoConnect, autoConnectBox);
    syncBools(dialog, *MEMainHandler::instance()->cfg_TopLevelBrowser, browserBox);
    syncBools(dialog, *MEMainHandler::instance()->cfg_TabletUITabs, tabletUITabsBox);

    qtStyleComboBox->setCurrentIndex(qtStyleComboBox->findText(MEMainHandler::instance()->cfg_QtStyle()));
    syncText(dialog, *MEMainHandler::instance()->cfg_HighColor, highlightColorEdit);
    syncInt(dialog, *MEMainHandler::instance()->cfg_AutoSaveTime, autoSaveTimeEdit);
    
    if(!dialog)
        return;
    developerModeCheckBox->setChecked(MEMainHandler::instance()->cfg_DeveloperMode->value());

    QObject::connect(dialog, &QDialogButtonBox::accepted, [this](){
        if(MEMainHandler::instance()->cfg_DeveloperMode->value() != developerModeCheckBox->isChecked())
        {
            (*MEMainHandler::instance()->cfg_DeveloperMode) = developerModeCheckBox->isChecked();
            MEMainHandler::instance()->developerModeHasChanged();

        }

    });
    MEMainHandler::instance()->cfg_DeveloperMode->setUpdater([this](bool state)
    {
        developerModeCheckBox->setChecked(state);
    });

}


template<typename T>
T* addSetting(const QString &labeltext, const QString &tooltip, QFormLayout* grid)
{
    QLabel* label = new QLabel(labeltext);       
    label->setToolTip(tooltip);                  
    T* widget = new T();                         
    widget->setToolTip(tooltip);                 
    grid->addRow(label, widget);
    return widget;
}

//!
//! make a grid layout for setting options
//!
void MESessionSettings::createFormLayout(QVBoxLayout *mainLayout)
{
    QGroupBox *container = NULL;

    container = new QGroupBox("General", this);
    container->setFont(MEMainHandler::s_boldFont);
    mainLayout->addWidget(container);

    QFormLayout *grid = new QFormLayout();

    qtStyleComboBox = addSetting<QComboBox>("Qt style", "Qt widget style for the map editor", grid);
    storeWindowConfigBox = addSetting<QCheckBox>("Restore window layout", "Enabled: restore size, position and docking state of all windows\nDisabled: do not restore window layout", grid);
    developerModeCheckBox = addSetting<QCheckBox>("Developer features", "Enable features that are useful only to developers", grid);
    errorHandlingCheckBox = addSetting<QCheckBox>("Error message dialogs", "Enabled: pop up a dialog box for each error message\nDisabled: show error messages in message window", grid);
    browserBox = addSetting<QCheckBox>("Embedded browsers", "Enabled: Filebrowsers and color maps are embedded into Module Parameter windows and Control Panel \nDisabled: Filebrowsers and color maps appear as toplevel window", grid);
    tabletUITabsBox = addSetting<QCheckBox>("Tablet UI as tabs", "Enabled: show tabs from tablet UI as siblings of map editor tabs\nDisabled: show tabs from tablet UI in their own sub-tab", grid);
    //addSetting(QCheckBox, imbeddedRenderBox, "Embedded ViNCE Renderer",
    //  "Enabled: ViNCE renderer is embedded into the MEMainHandler::instance()\nDisabled: ViNCE renderer appears as a toplevel window");i++;
    autoConnectBox = addSetting<QCheckBox>("Auto connect hosts", "Enabled: automatically connect to host or partner if connection mode is ssh or RemoteDaemon \nDisabled: always prompt the user", grid);
    container->setLayout(grid);

    container = new QGroupBox("Saving", this);
    container->setFont(MEMainHandler::s_boldFont);
    mainLayout->addWidget(container);
    grid = new QFormLayout();

    autoSaveTimeEdit = addSetting<QLineEdit>("Autosave interval", "Time interval for automatic saving (seconds)", grid);
    auto  configFile = new QLabel(("config file: " + MEMainHandler::instance()->getConfig().pathname()).c_str());
    configFile->setToolTip("the file where these settings are stored");
    grid->addRow(configFile); 

    auto  qsettings = new QLabel("usage setings: " + MEMainHandler::instance()->getUserBehaviour().fileName());
    qsettings->setToolTip("the registry where user behaviour is stored");
    grid->addRow(qsettings); 
    container->setLayout(grid);

    container = new QGroupBox("Visual Programming", this);
    container->setFont(MEMainHandler::s_boldFont);
    mainLayout->addWidget(container);
    grid = new QFormLayout();

    hideUnusedModulesBox = addSetting<QCheckBox>("Hide unused modules", "Enabled: module browser only shows recently used modules of each category\nDisabled: module browser shows all modules of a category", grid);
    highlightColorEdit = addSetting<QLineEdit>("Highlight color","Color name for highlighting module ports and connections", grid);
    container->setLayout(grid);
}

//!
//! save the settings
//!
void MESessionSettings::save()
{

    if (QString(MEMainHandler::instance()->cfg_QtStyle()) != qtStyleComboBox->currentText())
    {
        if (qtStyleComboBox->currentText() == "Default")
            MEMainHandler::instance()->cfg_QtStyle("");
        else
            MEMainHandler::instance()->cfg_QtStyle(qtStyleComboBox->currentText());
    }

    accept();
}

//!
//! restore the settings
//!
void MESessionSettings::cancel()
{
    initState(nullptr);
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
    tabletUITabsBox->setChecked(true);
}
