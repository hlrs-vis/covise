#ifndef VERTICALLABEL_H
#define VERTICALLABEL_H
#include <QLabel>
#include <QPainter>
#include <QWidget>
#include <QString>

class VerticalLabel : public QLabel
{
public:
    VerticalLabel(const QString &text, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;
    QSize sizeHint() const override;

};

#endif // VERTICALLABEL_H