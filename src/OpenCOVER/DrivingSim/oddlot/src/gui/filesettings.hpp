#ifndef FILESETTINGS_HPP
#define FILESETTINGS_HPP

#include <QDialog>
#include "ui_filesettings.h"

namespace Ui
{
    class FileSettings;
}

class FileSettings : public QDialog
{
    Q_OBJECT
    public:
        explicit FileSettings();
        virtual ~FileSettings();

        void addTab(QWidget *widget);

    signals:
        void emitOK();

    private slots:
        void okClicked();

    private:
        Ui::FileSettings *ui;
};

#endif // FILESETTINGS_HPP
