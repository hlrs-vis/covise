#ifndef MAPEDITOR_WAIT_FOR_CONNECTION_H
#define MAPEDITOR_WAIT_FOR_CONNECTION_H

#include <QWidget>
#include <QDialog>

namespace Ui
{
    class MEWaitingForConnection;
}

class MEWaitingForConnection : public QDialog
{
    Q_OBJECT
public:
    explicit MEWaitingForConnection(QWidget *parent = nullptr);

private:
    const QWidget *m_parent = nullptr;
    Ui::MEWaitingForConnection *m_ui;
protected:
    void showEvent(QShowEvent *event) override; 
private:
};

#endif