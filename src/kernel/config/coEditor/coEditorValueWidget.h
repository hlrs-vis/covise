/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VALUEWIDGET
#define VALUEWIDGET
#include <QWidget>
#include <QString>
#include <QRegExp>
#include <QMessageBox>

class coEditorValueWidget : public QWidget
{
    Q_OBJECT
public:
    enum Type
    {
        Text = 1,
        Bool = 2,
        Info = 3,
        InfoName = 4
    };

    coEditorValueWidget(QWidget *parent, Type type)
        : QWidget(parent)
    {
        fType = type;
    }

    ~coEditorValueWidget(){};

    Type getType() const;

//    virtual setValue(const QString & value);
//    virtual setName(const QString & value);
signals:
    void saveValue(const QString &variable, const QString &value);
    void deleteValue(const QString &variable);

public slots:
    virtual void setValue(const QString &valueName, const QString &value,
                          const QString &readableAttrRule = QString::null,
                          const QString &attributeDescription = QString::null,
                          bool required = false, const QRegExp &rx = QRegExp("^.*")) = 0;
    virtual void suicide() = 0;
    virtual void undo() = 0;

    const QString &getVariable() const;
    const QString &getValue() const;

private slots:

    void explainShowInfoButton()
    {
        QMessageBox::information(this, tr("Covise Config Editor"),
                                 tr("There is no need to delete a empty Field.\n"
                                    "To hide this field, press the Button with the \n"
                                    "blue I at the top. \n"));
    };

    // virtual setValue(const QString & variable, const QString & value, const QString & section);

protected:
    Type fType;
    QString fvariable, fvalue;
};

inline coEditorValueWidget::Type coEditorValueWidget::getType() const
{
    return this->fType;
}

inline const QString &coEditorValueWidget::getVariable() const
{
    return this->fvariable;
}

inline const QString &coEditorValueWidget::getValue() const
{
    return this->fvalue;
}
#endif
