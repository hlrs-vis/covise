#ifndef UI_QT_VIEW_H
#define UI_QT_VIEW_H

#include "View.h"

#include <QObject>

class QMenuBar;
class QAction;
class QLabel;

namespace opencover {
namespace ui {

struct QtViewElement: public View::ViewElement
{
    QtViewElement(Element *elem, QObject *obj);

    QObject *object = nullptr;
    QAction *action = nullptr;
    QLabel *label = nullptr;
};

class QtView: public QObject, public View
{
    Q_OBJECT

 public:
   QtView(QMenuBar *menubar);

 private:
   QMenuBar *m_menubar;

   void add(QtViewElement *ve);

   QtViewElement *qtViewElement(const Element *elem) const;
   QtViewElement *qtViewParent(const Element *elem) const;
   QtViewElement *qtViewContainer(const Element *elem) const;
   QObject *qtObject(const Element *elem) const;
   QObject *qtObject(const QtViewElement *elem) const;
   QWidget *qtWidget(const Element *elem) const;
   QWidget *qtWidget(const QtViewElement *elem) const;
   QWidget *qtContainerWidget(const Element *elem) const;

   void updateEnabled(const Element *elem) override;
   void updateVisible(const Element *elem) override;
   void updateText(const Element *elem) override;
   void updateState(const Button *) override;
   void updateChildren(const Menu *menu) override;
   void updateInteger(const Slider *slider) override;
   void updateValue(const Slider *slider) override;
   void updateBounds(const Slider *slider) override;

   QtViewElement *elementFactoryImplementation(Menu *menu) override;
   QtViewElement *elementFactoryImplementation(RadioGroup *rg) override;
   QtViewElement *elementFactoryImplementation(Group *group) override;
   QtViewElement *elementFactoryImplementation(Label *label) override;
   QtViewElement *elementFactoryImplementation(Action *action) override;
   QtViewElement *elementFactoryImplementation(Button *button) override;
   QtViewElement *elementFactoryImplementation(Slider *slider) override;
};

}
}
#endif
