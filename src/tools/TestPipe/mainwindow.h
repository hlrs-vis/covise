#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <stdio.h>
#include <math.h>
#ifndef WIN32
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#else
#ifndef NOMINMAX
#define NOMINMAX        // to avoid "not enough actual parameters for macro 'min'" with VS2010
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>    // for sleep
#define popen _popen
#define pclose _pclose
#endif


#define TOL 1e-7

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    int pipe_open(char *argv[], FILE **fp_read, FILE **fp_write);
    int pipe_close(FILE *fp_read, FILE *fp_write);

    bool coviseIsRunning;   // is Covise running?

#ifndef WIN32
    FILE *pfp_read;       // pipe to read output from Covise (scriptingInterface) - bidirectional communication not on windows
#endif
    FILE *pfp_write;      // pipe to send commands to Covise (scriptingInterface)

    float cuttsurf1_scalar;

private slots:
    void startcovise();
    void stopcovise();

};

#endif // MAINWINDOW_H
