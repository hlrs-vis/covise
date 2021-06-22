#include "childOutput.h"
#include <QScrollArea>
#include <QScrollBar>
#include <QTabWidget>
#include <QTextBrowser>

void updateTextBrowser(QTextBrowser *tb, const QString &msg)
{
	QScrollBar *scrollbar = tb->verticalScrollBar();
	bool scrollbarAtBottom = (scrollbar->value() >= (scrollbar->maximum() - 4));
	int scrollbarPrevValue = scrollbar->value();
	tb->setText(tb->toPlainText() + msg);
	if (scrollbarAtBottom)
		scrollbar->setValue(scrollbar->maximum());
	else
		scrollbar->setValue(scrollbarPrevValue);
}

ChildOutput::ChildOutput(const QString &childId, QTabWidget *tabWidget)
:m_childId(childId)
, m_tabWidget(tabWidget)
{
        auto textArea = new QScrollArea(tabWidget);
		textArea->setWidgetResizable(true);

		m_textBrowser = new QTextBrowser(textArea);
		textArea->setWidget(m_textBrowser);
		static int numSpawns = 0;
		++numSpawns;
		m_index = tabWidget->addTab(textArea, childId);
}

ChildOutput::~ChildOutput()
{
    m_tabWidget->removeTab(m_index);
}

void ChildOutput::addText(const QString &txt)
{
    updateTextBrowser(m_textBrowser, txt);
}

//bool ChildOutput::operator==(const QString &childId) const;
//{
//    return m_childId == m_childId;
//}
