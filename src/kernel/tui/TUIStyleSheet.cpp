#include "TUIStyleSheet.h"

std::string TUI_STYLESHEET = R"(
    QSlider:horizontal {
        min-height: 24px;
        max-height: 24px;
        height: 24px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #999999;
        height: 6px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
        margin: 0;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
        border: 1px solid #5c5c5c;
        width: 24px;
        height: 24px;
        margin: -9px 0;
        border-radius: 6px;
    }
)";
