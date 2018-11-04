#pragma once
#include"stdafx.h"
#include <cover/ui/View.h>

#define MENU_ID 5000
#define BUTTON_ID 6000
#define SLIDER_ID 7000

namespace opencover {
	namespace ui {

		struct mfcViewElement : public opencover::ui::View::ViewElement
		{
			//! create for @param elem which has a corresponding @param obj
			mfcViewElement(opencover::ui::Element *elem, CWnd *obj);

			CWnd *object = nullptr;
				//QAction *action = nullptr;
				//QLabel *label = nullptr;
		};

		class CMFCView : public opencover::ui::View
		{
		public:
			CMFCView();
			CMenu *m_menubar;
			int menu_ID_Count;
			int button_ID_Count;
			int slider_ID_Count;
			CMFCView(CMenu *m_menubar);
			//QWidget *CMFCView::mfcContainerWidget(const opencover::ui::Element *elem) const;

			mfcViewElement *CMFCView::mfcViewParent(const opencover::ui::Element *elem) const;

			mfcViewElement *CMFCView::mfcViewElement(const Element *elem) const;

			virtual ~CMFCView();
			void updateEnabled(const opencover::ui::Element *elem);
			//! reflect changed visibility in graphical representation
			void updateVisible(const opencover::ui::Element *elem);
			//! reflect changed text in graphical representation
			void updateText(const opencover::ui::Element *elem);
			//! reflect changed button state in graphical representation
			void updateState(const opencover::ui::Button *button);
			//! reflect change of child tems in graphical representation
			void updateChildren(const opencover::ui::Menu *menu);
			//! reflect change of slider type in graphical representation
			void updateInteger(const opencover::ui::Slider *slider);
			//! reflect change of slider value in graphical representation
			void updateValue(const opencover::ui::Slider *slider);
			//! reflect change of slider range in graphical representation
			void updateBounds(const opencover::ui::Slider *slider);
			std::wstring toWide(const std::string &source);


		protected:
			//! implement to create graphical representation of a menu
			ViewElement *elementFactoryImplementation(opencover::ui::Menu *Menu);
			//! implement to create graphical representation of an item group (e.g. a frame, possible with a label)
			ViewElement *elementFactoryImplementation(opencover::ui::Group *group);
			//! implement to create graphical representation of an radio group of toggle buttons
			ViewElement *elementFactoryImplementation(opencover::ui::ButtonGroup *bg);
			//! implement to create graphical representation of a text label
			ViewElement *elementFactoryImplementation(opencover::ui::Label *label);
			//! implement to create graphical representation of a stateless button
			ViewElement *elementFactoryImplementation(opencover::ui::Action *action);
			//! implement to create graphical representation of a button with binary state
			ViewElement *elementFactoryImplementation(opencover::ui::Button *button);
			//! implement to create graphical representation of a slider
			ViewElement *elementFactoryImplementation(opencover::ui::Slider *slider);
		};
	}
}
