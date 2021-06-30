/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "NonBlockingDialogue.h"
#include "ui_NonBlockingDialogue.h"

#include <QString>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QBoxLayout>

using namespace covise;

NonBlockingDialogue::NonBlockingDialogue(QWidget *parent)
    : m_parent(parent), m_ui(new Ui::dialog)
{
  m_ui->setupUi(this);
}

void NonBlockingDialogue::setInfo(const QString &text)
{
  m_ui->infoLbl->setText(text);
  m_ui->infoLbl->setWordWrap(true);
}

void NonBlockingDialogue::setQuestion(const QString &text)
{
  m_ui->questionLbl->setText(text);
  m_ui->questionLbl->setWordWrap(true);
}

int NonBlockingDialogue::addOption(const QString &text)
{
  static int numOptions = 1;
  auto b = new QPushButton(text, m_ui->verticalLayoutWidget);
  m_ui->optionsLayout->addWidget(b);
  int num = numOptions++;
  connect(b, &QPushButton::clicked, this, [this, num]()
          { emit answer(num); });
  connect(b, &QPushButton::clicked, this, &NonBlockingDialogue::hide);
  return num;
}

void NonBlockingDialogue::showEvent(QShowEvent *event)
{
  if (m_parent)
  {
    move(m_parent->pos() + QPoint{m_parent->rect().width() / 2, m_parent->rect().height() / 2} - rect().center());
  }
  show();
}