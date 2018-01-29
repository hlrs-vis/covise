#ifndef UI_QT_VIEW_H
#define UI_QT_VIEW_H

#include <cover/ui/View.h>

#include <QObject>
#include <QWidgetAction>
#include <QBoxLayout>

class QMenuBar;
class QToolBar;
class QAction;
class QActionGroup;
class QLabel;
class QSlider;

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


//! for simultaneous rendering of \ref Label labels in toolbars and menues
class QtLabelAction: public QWidgetAction
{
    Q_OBJECT

public:
    QtLabelAction(QObject *parent);
    void setText(const QString &text);

protected:
    QWidget *createWidget(QWidget *parent) override;
};


//! q Qt widget for showing a slider with a label showing its value
class QtSliderWidget: public QWidget
{
    Q_OBJECT

public:
    QtSliderWidget(QBoxLayout::Direction dir, QWidget *parent);
    void setDirection(QBoxLayout::Direction dir);

    void setText(const QString &text);
    void setWidthText(const QString &text);
    void setRange(int min, int max);
    void setValue(int value);
    int value() const;
    int minimum() const;
    int maximum() const;

signals:
    void sliderMoved(int value);
    void sliderReleased();

private:
    QBoxLayout *m_layout = nullptr;
    QLabel *m_label = nullptr;;
    QSlider *m_slider = nullptr;
};


//! for simultaneous rendering of \ref Slider sliders in toolbars and menues
class QtSliderAction: public QWidgetAction
{
    Q_OBJECT

public:
    QtSliderAction(QObject *parent);
    void setToolTip(const QString &tip);
    void setText(const QString &text);
    void setWidthText(const QString &text);
    void setRange(int min, int max);
    void setValue(long value);
    int value() const;
    int minimum() const;
    int maximum() const;

signals:
    void sliderMoved(int value);
    void sliderReleased();

protected:
    QWidget *createWidget(QWidget *parent) override;
    //void deleteWidget(QWidget *parent) override;

private:
    QString m_text, m_tip, m_widthText;
    int m_value = 0;
    int m_min = 0, m_max = 0;
};

//! concrete implementation of View for showing user interface \ref Element "elements" in a QMenuBar and QToolBar
class QtView: public QObject, public View
{
    Q_OBJECT

 public:
   QtView(QMenuBar *menubar, QToolBar *m_toolbar = nullptr);
   QtView(QToolBar *toolbar);
   ViewType typeBit() const override;

   void setInsertPosition(QAction *item);

 private:
   QMenuBar *m_menubar = nullptr;
   QAction *m_insertBefore = nullptr;
   QToolBar *m_toolbar = nullptr;

   Group *m_lastToolbarGroup = nullptr;

   //! add a previously created QtViewElement to its parent
   void add(QtViewElement *ve, bool update=false);

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
   void updateValue(const Input *input) override;

   QtViewElement *elementFactoryImplementation(Menu *menu) override;
   QtViewElement *elementFactoryImplementation(Group *group) override;
   QtViewElement *elementFactoryImplementation(Label *label) override;
   QtViewElement *elementFactoryImplementation(Action *action) override;
   QtViewElement *elementFactoryImplementation(Button *button) override;
   QtViewElement *elementFactoryImplementation(Slider *slider) override;
   QtViewElement *elementFactoryImplementation(SelectionList *sl) override;
   QtViewElement *elementFactoryImplementation(Input *input) override;

   void updateContainer(const Element *elem);
   void updateMenu(const Menu *menu, const Group *subGroup);
};

}
}
#endif
