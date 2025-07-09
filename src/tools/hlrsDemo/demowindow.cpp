
#include "demowindow.h"
#include <demo.h>
#include "verticallabel.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <QDesktopServices>
#include <QDir>
#include <QDir>
#include <QFile>
#include <QIcon>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QStandardPaths>
#include <QTextBrowser>
#include <QTextEdit>
#include <QUrl>
#include <QVBoxLayout>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

// Helper function to read demos.json using nlohmann::json
json readDemosJson(const QString &path)
{
    std::ifstream file(path.toStdString());
    if (!file)
        return {};
    json j;
    file >> j;
    return j;
}

QString autoLinkUrls(const QString &text)
{
    QRegularExpression re(R"((https?://[^\s<]+))");
    QString html = text;
    html.replace(re, R"(<a href="\1">\1</a>)");
    html.replace("\n", "<br>");
    return html;
}

void highlightDemo(QWidget *cellWidget, bool highlight)
{
    if (!cellWidget)
        return;
    cellWidget->setStyleSheet(highlight
                                  ? "QWidget { border: 2px solid #3399ff; border-radius: 8px; background: #e0f0ff; }"
                                  : "");
}

DemoWindow::DemoWindow(const json &demos, QWidget *parent)
    : QWidget(parent), demos_(demos)
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    // Set minimum size so only one demo cell fits in a row
    setMinimumWidth(200);  // Adjust to your cell width + margins
    setMinimumHeight(300); // Reasonable minimum height

    // Search field
    searchEdit = new QLineEdit(this);
    searchEdit->setPlaceholderText("Search...");
    mainLayout->addWidget(searchEdit);
    // Make the demo area scrollable if not all fit vertically
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);

    QWidget *flowWidget = new QWidget(this);
    flowLayout = new FlowLayout(flowWidget, 2, 6, 6);
    flowWidget->setLayout(flowLayout);

    scrollArea->setWidget(flowWidget);
    mainLayout->addWidget(scrollArea);

    setLayout(mainLayout);

    connect(searchEdit, &QLineEdit::textChanged, this, &DemoWindow::updateSearch);

    createCells();
}

void DemoWindow::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    // FlowLayout handles reflow automatically
}

// Helper: returns lowercase version of a QString
QString lower(const QString &s) { return s.toLower(); }

// Helper: returns true if haystack contains needle (case-insensitive)
bool containsCI(const QString &haystack, const QString &needle)
{
    return haystack.toLower().contains(needle.toLower());
}

void DemoWindow::createCells(const QString &filter)
{
    // Remove old widgets if any
    for (auto *w : cellWidgets)
        delete w;
    cellWidgets.clear();

    // Remove all widgets from layout
    QLayoutItem *item;
    while ((item = flowLayout->takeAt(0)) != nullptr)
    {
        if (item->widget())
            item->widget()->setParent(nullptr);
        delete item;
    }

    // Iterate over all categories
    for (auto it = demos_.begin(); it != demos_.end(); ++it)
    {
        QString category = QString::fromStdString(it.key());
        const json &demoArray = it.value();

        // Collect filtered demos for this category
        struct DemoEntry
        {
            const json *demo;
            int matchRank;
        };
        std::vector<DemoEntry> filtered;

        if (demoArray.is_array())
        {
            for (const auto &demo : demoArray)
            {
                QString headline = QString::fromStdString(demo.value("headline", "Unnamed"));
                QString description = QString::fromStdString(demo.value("description", ""));
                QStringList tags;
                if (demo.contains("tags") && demo["tags"].is_array())
                {
                    for (const auto &tag : demo["tags"])
                        tags << QString::fromStdString(tag.get<std::string>());
                }

                int rank = 3;
                if (filter.isEmpty())
                    rank = 0;
                else if (containsCI(headline, filter))
                    rank = 0;
                else if (std::any_of(tags.begin(), tags.end(), [&](const QString &tag)
                                     { return containsCI(tag, filter); }))
                    rank = 1;
                else if (containsCI(description, filter))
                    rank = 2;

                if (rank < 3)
                    filtered.push_back(DemoEntry{&demo, rank});
            }
        }

        // Sort by match rank, then by headline
        std::sort(filtered.begin(), filtered.end(), [](const DemoEntry &a, const DemoEntry &b)
                  {
        if (a.matchRank != b.matchRank)
            return a.matchRank < b.matchRank;
        QString ha = QString::fromStdString(a.demo->value("headline", ""));
        QString hb = QString::fromStdString(b.demo->value("headline", ""));
        return ha.toLower() < hb.toLower(); });

        // Only add category label if there are demos to show
        if (!filtered.empty())
        {
            // Category label as its own full-width row
            QWidget *catRow = new QWidget(this);
            QHBoxLayout *catLayout = new QHBoxLayout(catRow);
            catLayout->setContentsMargins(0, 0, 0, 0);
            catLayout->setSpacing(0);

            VerticalLabel *catLabel = new VerticalLabel(category, catRow);
            QFont font = catLabel->font();
            font.setBold(true);
            font.setPointSize(font.pointSize() + 2);
            catLabel->setFont(font);
            catLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
            catLayout->addWidget(catLabel, 0, Qt::AlignLeft);

            catRow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            flowLayout->addWidget(catRow);
            cellWidgets.push_back(catRow);

            // Add all demo widgets
            for (const auto &entry : filtered)
            {
                const json &demo = *entry.demo;

                if (demo.contains("launch") && demo["launch"].is_array())
                {
                    std::vector<std::pair<QString, QStringList>> programs;
                    for (const auto &launch : demo["launch"])
                    {
                        QString program = QString::fromStdString(launch.value("program", ""));
                        QStringList args;
                        if (launch.contains("args") && launch["args"].is_array())
                        {
                            for (const auto &arg : launch["args"])
                            {
                                args << QString::fromStdString(arg.get<std::string>());
                            }
                        }
                        programs.push_back({program, args});
                    }

                    auto *cellWidget = createDemoWidget(demo);
                    cellWidgets.push_back(cellWidget);
                    flowLayout->addWidget(cellWidget);
                }
            }
        }
    }
}

void DemoWindow::updateSearch(const QString &text)
{
    createCells(text);
}

QWidget *DemoWindow::createDemoWidget(const nlohmann::json &demo)
{
    QWidget *cellWidget = new QWidget(this);
    QVBoxLayout *cellLayout = new QVBoxLayout(cellWidget);
    cellLayout->setContentsMargins(2, 2, 2, 2);
    cellLayout->setSpacing(2);

    // Create a button with the image
    QPushButton *btn = new QPushButton(cellWidget);
    btn->setFixedSize(140, 80);
    btn->setIconSize(QSize(135, 135));
    btn->setText("");

    // Remove button border and background
    btn->setStyleSheet(
        "QPushButton { border: none; background: transparent; }"
        "QPushButton:hover { background: #e0eaff; border: 1px solid #3399ff; border-radius: 6px; }");

    QString imagePath;
    if (demo.contains("image") && demo["image"].is_string())
    {
        std::string imageName = demo["image"];
        std::filesystem::path p = demo::imageDir;
        p = p / imageName;
        imagePath = p.string().c_str();
    }
    if (!imagePath.isEmpty() && QFile::exists(imagePath))
    {
        QPixmap pix(imagePath);
        btn->setIcon(QIcon(pix.scaled(128, 128, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    }
    cellLayout->addWidget(btn, 0, Qt::AlignHCenter | Qt::AlignTop);

    // Headline as clickable label
    QPushButton *headlineBtn = new QPushButton(demo.value("headline", "Unnamed").c_str(), cellWidget);
    headlineBtn->setFlat(true);
    headlineBtn->setStyleSheet("QPushButton { color: blue; text-decoration: underline; background: transparent; border: none; }");
    headlineBtn->setCursor(Qt::PointingHandCursor);
    cellLayout->addWidget(headlineBtn, 0, Qt::AlignHCenter | Qt::AlignTop);

    // Description browser (default, supports links)
    QTextBrowser *descBrowser = new QTextBrowser(cellWidget);
    descBrowser->setOpenExternalLinks(true);
    descBrowser->setHtml(demo.value("description", "").c_str());
    descBrowser->setVisible(false);
    descBrowser->setMinimumHeight(60);
    descBrowser->setMaximumHeight(150);

    // Description editor (hidden by default)
    QTextEdit *descEdit = new QTextEdit(cellWidget);
    descEdit->setAcceptRichText(true);
    descEdit->setText(demo.value("description", "").c_str());
    descEdit->setVisible(false);
    descEdit->setMinimumHeight(60);
    descEdit->setMaximumHeight(150);

    // Edit button to toggle between view and edit
    QPushButton *editBtn = new QPushButton("Edit", cellWidget);
    editBtn->setVisible(false); // Only show when description is visible
    cellLayout->addWidget(editBtn, 0, Qt::AlignHCenter | Qt::AlignTop);

    // Scroll area for browser
    QScrollArea *descScroll = new QScrollArea(cellWidget);
    descScroll->setWidget(descBrowser);
    descScroll->setWidgetResizable(true);
    descScroll->setVisible(false);
    descScroll->setMinimumHeight(60);
    descScroll->setMaximumHeight(150);
    descScroll->setFrameShape(QFrame::NoFrame);
    cellLayout->addWidget(descScroll, 0, Qt::AlignHCenter | Qt::AlignTop);

    // Scroll area for editor
    QScrollArea *editScroll = new QScrollArea(cellWidget);
    editScroll->setWidget(descEdit);
    editScroll->setWidgetResizable(true);
    editScroll->setVisible(false);
    editScroll->setMinimumHeight(60);
    editScroll->setMaximumHeight(150);
    editScroll->setFrameShape(QFrame::NoFrame);
    cellLayout->addWidget(editScroll, 0, Qt::AlignHCenter | Qt::AlignTop);

    // Toggle description on headline click
    connect(headlineBtn, &QPushButton::clicked, [descScroll, editBtn, descBrowser, editScroll, descEdit]()
    {
        bool show = !descScroll->isVisible() && !editScroll->isVisible();
        descScroll->setVisible(show);
        editBtn->setVisible(show);
        descBrowser->setVisible(show);
        if (!show) {
            editScroll->setVisible(false);
            descEdit->setVisible(false);
            descBrowser->setVisible(false);
            descScroll->setVisible(false);
            editBtn->setVisible(false);
        }
    });

    // Toggle between browser and editor
    connect(editBtn, &QPushButton::clicked, [=]() mutable
    {
        bool editing = editScroll->isVisible();
        if (!editing) {
            // Switch to edit mode
            descEdit->setText(descBrowser->toHtml());
            descScroll->setVisible(false);
            descBrowser->setVisible(false);
            editScroll->setVisible(true);
            descEdit->setVisible(true);
            editBtn->setText("Save");
        } else {
            // Save and switch back to browser mode
            QString editedText = descEdit->toPlainText();
            QString htmlText = autoLinkUrls(editedText);
            descBrowser->setHtml(htmlText);
            editScroll->setVisible(false);
            descEdit->setVisible(false);
            descScroll->setVisible(true);
            descBrowser->setVisible(true);
            editBtn->setText("Edit");

            // --- Save changes to myDemos.json ---
            // Find the demo in demos_ (respecting categories) and update its description
            bool updated = false;
            for (auto &cat : demos_.items()) {
                if (!cat.value().is_array())
                    continue;
                for (auto &otherDemo : cat.value()) {
                    if (otherDemo.value("headline", "") == demo.value("headline", "")) {
                        otherDemo["description"] = htmlText.toStdString();
                        updated = true;
                        break;
                    }
                }
                if (updated) break;
            }
            std::ofstream out(demo::collection);
            if (out) {
                out << demos_.dump(2);
                out.close();
            }
        }
    });

    // Toggle description on headline click
    connect(headlineBtn, &QPushButton::clicked, [descEdit]()
            { descEdit->setVisible(!descEdit->isVisible()); });

    // Launch application on image click
    connect(btn, &QPushButton::clicked, [btn, demo, this, cellWidget]()
    {
        int id = demo.value("id", 0);
        
        // If this demo is running, offer to terminate it
        if (runningDemoWidget && runningDemoWidget == cellWidget) {

            auto reply = QMessageBox::question(this, "Terminate Demo",
                "This demo is running. Do you really want to terminate it?",
                QMessageBox::Yes | QMessageBox::No);
            if (reply == QMessageBox::Yes) {
                launcher.terminateDemo(id);
                highlightDemo(cellWidget, false);
                runningDemoWidget = nullptr;
            }
            return;
        }
    
        // If another demo is running, ask if it should be terminated
        if (runningDemoWidget) {
            auto reply = QMessageBox::question(this, "Terminate Running Demo",
                "Another demo is currently running. Do you want to terminate it and start this one?",
                QMessageBox::Yes | QMessageBox::No);
            if (reply == QMessageBox::No)
                return;
            launcher.terminateDemo(runningDemoId);
            highlightDemo(cellWidget, false);
        }
        launcher.launchDemo(demo);
        highlightDemo(cellWidget, true);
        runningDemoWidget = cellWidget;
        runningDemoId = id;
    });

    return cellWidget;
}
