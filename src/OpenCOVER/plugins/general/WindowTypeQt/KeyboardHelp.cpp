#include "KeyboardHelp.h"
#include "ui_KeyboardHelp.h"
#include <cover/ui/Manager.h>
#include <QTreeWidget>
#include <QTreeWidgetItem>

KeyboardHelp::KeyboardHelp(opencover::ui::Manager *mgr, QWidget *parent)
: QDialog(parent)
, ui(new Ui::KeyboardHelp)
{
    ui->setupUi(this);

    auto tw = ui->treeWidget;

    auto elems = mgr->getAllElements();
    for (auto e: elems)
    {
        QStringList columns;
        columns.append(QString::fromStdString(e->text()));
        columns.append(QString::fromStdString(e->path()));
        size_t numShortcuts = e->shortcutCount();
        std::string shortcuts;
        for (size_t i=0; i<numShortcuts; ++i)
        {
            if (i > 0)
                shortcuts += "; ";
            shortcuts += e->shortcutText(i);
        }
        columns.append(QString::fromStdString(shortcuts));
        auto item = new QTreeWidgetItem(tw, columns);
        tw->addTopLevelItem(item);
    }
    tw->setColumnWidth(0, 200);
    tw->setColumnWidth(1, 350);
}

KeyboardHelp::~KeyboardHelp()
{
    delete ui;
}
