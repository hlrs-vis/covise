#include "verticallabel.h"


VerticalLabel::VerticalLabel(const QString &text, QWidget *parent) : QLabel(text, parent) {}

void VerticalLabel::paintEvent(QPaintEvent *event) 
{
    QPainter painter(this);
    painter.translate(width(), 0);
    painter.rotate(90);
    painter.drawText(0, 0, height(), width(), alignment(), text());
}

QSize VerticalLabel::sizeHint() const 
{
    QSize s = QLabel::sizeHint();
    return QSize(s.height(), s.width());
}
