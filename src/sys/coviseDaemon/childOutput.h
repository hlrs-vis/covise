#ifndef COVISE_DAEMON_CHILD_OUTPUT_H
#define COVISE_DAEMON_CHILD_OUTPUT_H

#include <QString>

class QTabWidget;
class QTextBrowser;
struct ChildOutput
{
    ChildOutput(const QString &childId, QTabWidget *tabWidget);
    ~ChildOutput();
    void addText(const QString &txt);
    bool operator==(const QString &childId) const
    {
        return m_childId == m_childId;
    }

private:
    QString m_childId;
    QTextBrowser *m_textBrowser;
    QTabWidget *m_tabWidget;
    int m_index;
};

#endif // !COVISE_DAEMON_CHILD_OUTPUT_H