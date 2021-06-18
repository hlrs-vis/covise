#include "MEWaitingForConnection.h"
#include "ui_MEWaitingForConnection.h"
#include <QPushButton>
MEWaitingForConnection::MEWaitingForConnection(QWidget *parent)
:QDialog(parent), m_ui(new Ui::MEWaitingForConnection), m_parent(parent)
{
    m_ui->setupUi(this);
    connect(m_ui->okBtn, &QPushButton::pressed, this, &QWidget::hide);
}

void MEWaitingForConnection::showEvent(QShowEvent * event)
{
    if (m_parent)
    {
        move(m_parent->pos() + QPoint{m_parent->rect().width() / 2, m_parent->rect().height() / 2} - rect().center());
    }
    show();
}