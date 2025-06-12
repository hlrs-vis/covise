#ifndef HLRS_DEMO_WINDOW_H
#define HLRS_DEMO_WINDOW_H

#include "flowlayout.h"
#include "launch.h"
#include <QWidget>
#include <QResizeEvent>
#include <QLineEdit>
#include <QString>
#include <QProcess>
#include <nlohmann/json.hpp>

#include <map>
#include <vector>

nlohmann::json readDemosJson(const QString &path);


class DemoWindow : public QWidget
{
public:
    DemoWindow(const nlohmann::json &demos, QWidget *parent = nullptr);
        
protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    nlohmann::json demos_;
    FlowLayout *flowLayout;
    std::vector<QWidget *> cellWidgets;
    QLineEdit *searchEdit;
    int runningDemoId = -1;
    QWidget *runningDemoWidget = nullptr;
    DemoLauncher launcher;
    void createCells(const QString &filter = QString());
    void updateSearch(const QString &text);
    QWidget *createDemoWidget(const nlohmann::json &demo);
    
};


#endif // HLRS_DEMO_WINDOW_H