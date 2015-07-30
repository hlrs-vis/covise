#include "vrconfwizard.h"

#include <QtGui>
#include <QApplication>

int main(int argc, char *argv[])
{
   QApplication a(argc, argv);
   VRConfWizard w;
   w.show();
   return a.exec();
}
