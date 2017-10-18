#ifndef UI_QT_VIEW_H
#define UI_QT_VIEW_H

#include <cover/ui/View.h>

#include <QObject>

class QMenuBar;
class QAction;
class QActionGroup;
class QLabel;

namespace opencover {
namespace ui {

//! store the data for the representation of a UI Element within a QtView
struct QtViewElement: public QObject, public View::ViewElement
{
    //! create for @param elem which has a corresponding @param obj
    QtViewElement(Element *elem, QObject *obj);
    ~QtViewElement();
    void markForDeletion(QObject *obj);

    QObject *object = nullptr;
    QAction *action = nullptr;
    QLabel *label = nullptr;
    QActionGroup *group = nullptr;

    std::vector<QObject *> toDelete;
};

//! concrete implementation of View for showing user interface \ref Element "elements" in a QMenuBar
class QtView: public QObject, public View
{
    Q_OBJECT

 public:
   QtView(QMenuBar *menubar);

 private:
   QMenuBar *m_menubar;

   //! add a previously created QtViewElement to its parent
   void add(QtViewElement *ve);

   // helper functions for navigating UI element hierarchy
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
   void updateParent(const Element *elem) override;
   void updateState(const Button *) override;
   void updateChildren(const SelectionList *sl) override;
   void updateIntegral(const Slider *slider) override;
   void updateScale(const Slider *slider) override;
   void updateValue(const Slider *slider) override;
   void updateBounds(const Slider *slider) override;

   QtViewElement *elementFactoryImplementation(Menu *menu) override;
   QtViewElement *elementFactoryImplementation(Group *group) override;
   QtViewElement *elementFactoryImplementation(Label *label) override;
   QtViewElement *elementFactoryImplementation(Action *action) override;
   QtViewElement *elementFactoryImplementation(Button *button) override;
   QtViewElement *elementFactoryImplementation(Slider *slider) override;
   QtViewElement *elementFactoryImplementation(SelectionList *sl) override;
};

}
}
#endif
