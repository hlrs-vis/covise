#ifndef KEYBOARDHELP_H
#define KEYBOARDHELP_H

#include <QDialog>

namespace opencover {
namespace ui {
class Manager;
}
}

namespace Ui {
class KeyboardHelp;
}

class KeyboardHelp : public QDialog
{
    Q_OBJECT

public:
    explicit KeyboardHelp(opencover::ui::Manager *mgr, QWidget *parent = 0);
    ~KeyboardHelp();

private:
    Ui::KeyboardHelp *ui;
};

#endif // KEYBOARDHELP_H
