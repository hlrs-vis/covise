#include "filesettings.hpp"
#include "ui_filesettings.h"
#include <QMessageBox>

FileSettings::FileSettings()
    :ui(new Ui::FileSettings)
{
    ui->setupUi(this);
    connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(okClicked()));
    connect(ui->buttonBox, SIGNAL(rejected()), this, SLOT(reject()));
}

FileSettings::~FileSettings()
{
    delete ui;
}

void FileSettings::addTab(QWidget *widget)
{
    connect(this, SIGNAL(emitOK()),widget, SLOT(okPressed()));

    ui->tabWidget->addTab((widget), ((widget))->objectName());
}

void FileSettings::okClicked()
{
    emit emitOK();
}
