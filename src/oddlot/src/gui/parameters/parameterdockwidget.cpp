/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   11/2/2010
 **
 **************************************************************************/

#include "parameterdockwidget.hpp"

#include "src/mainwindow.hpp"

 // GUI //
 //
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/editors/projecteditor.hpp"

#include <QPushButton>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QGroupBox>
#include <QLabel>
#include <QSizePolicy>
#include <QToolTip>

//######################//
// ParameterDockWidget  //
//######################//

ParameterDockWidget::ParameterDockWidget(const QString &title, MainWindow *parent)
    :QDockWidget(title, parent)
    , mainWindow_(parent)
{
    init();
}

ParameterDockWidget::~ParameterDockWidget()
{
};

void
ParameterDockWidget::init()
{

    // ParameterDialog Widget //
    //

    QWidget *parameterWidget = new QWidget();
    setWidget(parameterWidget);

    QVBoxLayout *layout = new QVBoxLayout();

    QScrollArea *scrollArea = new QScrollArea(parameterWidget);

    paramBox_ = new QFrame(this);
    dialogBox_ = new QFrame(this);
    paramBox_->installEventFilter(this);
    dialogBox_->installEventFilter(this);

    paramGroupBox_ = new QGroupBox();
    QVBoxLayout *paramLayout = new QVBoxLayout();
    paramLayout->addWidget(paramBox_);
    paramLayout->addWidget(dialogBox_);

    paramGroupBox_->setLayout(paramLayout);

    scrollArea->setWidget(paramGroupBox_);
    scrollArea->setWidgetResizable(true);

    layout->addWidget(scrollArea);
    parameterWidget->setLayout(layout);
    parameterWidget->setVisible(false);
}

void
ParameterDockWidget::setVisibility(bool visible, const QString &helpText, const QString &windowTitle)
{
    paramGroupBox_->setTitle(windowTitle);
    setWhatsThis(helpText);
    widget()->setVisible(visible);
    if (visible)
    {
        projectEditor_ = mainWindow_->getActiveProject()->getProjectEditor();
    }
}

bool
ParameterDockWidget::eventFilter(QObject *object, QEvent *event)
{

    if ((object == paramBox_) || (object == dialogBox_)) {
        if (event->type() == QEvent::Enter) {
            projectEditor_->focusParameterDialog(true);
            return true;
        }
        else {
            return false;
        }
    }

    // pass the event on to the parent class
    return QDockWidget::eventFilter(object, event);
}

void
ParameterDockWidget::enterEvent(QEvent *event)
{
    if (widget()->isVisible())
    {
        if (event->type() == QEvent::Enter)
        {
            QEnterEvent *enterEvent = dynamic_cast<QEnterEvent *>(event);
            if (enterEvent)
            {
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
                int d = mapToParent(enterEvent->position().toPoint()).y() - pos().y();
#else
                int d = mapToParent(enterEvent->localPos().toPoint()).y() - pos().y();
#endif
                if ((d >= 0) && (d < 20))
                {
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
                    QToolTip::showText(enterEvent->globalPosition().toPoint(), whatsThis());
#else
                    QToolTip::showText(enterEvent->globalPos(), whatsThis());
#endif
                }
            }
        }
        projectEditor_->focusParameterDialog(true);
    }
}

void
ParameterDockWidget::leaveEvent(QEvent *event)
{
    if (widget()->isVisible())
    {
        projectEditor_->focusParameterDialog(false);
    }
}




