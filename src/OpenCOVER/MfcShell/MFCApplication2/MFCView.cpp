#include "stdafx.h"
#include "MFCView.h"

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
//#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <atlbase.h>

namespace opencover {
	namespace ui {
		CMFCView::CMFCView() :View("MFC")
		{
			m_menubar = NULL;
		}
		CMFCView::CMFCView(CMenu *m_menubar) : View("MFC"), m_menubar(m_menubar)
		{
			menu_ID_Count = 0;
			button_ID_Count = 0;

		}

		CMFCView::~CMFCView()
		{
		}


		void CMFCView::updateEnabled(const opencover::ui::Element *elem)
		{
			auto w = (CWnd*)elem;
			//if (w)
				//w->EnableWindow(elem->enabled());
			
		}

		//! reflect changed visibility in graphical representation
		void CMFCView::updateVisible(const opencover::ui::Element *elem)
		{
			auto w = (CWnd*)elem;
			if (w)
			{
				
				auto m = dynamic_cast<CMenu *>(w);
				if (!m)
				{
					////w->ShowWindow  (elem->visible());
				}
				else if (auto ve = mfcViewElement(elem))
				{
					auto a = ve->object;
					//if (a)
						//a->ShowWindow(elem->visible());
				}
			}
		}
		//! reflect changed text in graphical representation
		void CMFCView::updateText(const opencover::ui::Element *elem)
		{
			/*auto ve = mfcViewElement(elem);
			assert(ve);
			auto o = (CWnd*)elem;
			auto t = elem->text();
			if (auto l = dynamic_cast<CStatic *>(o))
			{
			

				l->SetWindowTextW(toWide(t).c_str());
			}
			//else if (auto a = dynamic_cast<QAction *>(o))
			//{
				//a->setText(t);
			//}
			else if (auto m = dynamic_cast<CMenu *>(o))
			{
				//UINT id = m->GetMenuItemID(m.get);
				m->ModifyMenu(111, MF_BYCOMMAND, 11, toWide(t).c_str()); // 새로운 문자열로 변경한다.
			
			}
			else if (auto qs = dynamic_cast<QSlider *>(o))
			{
				/////auto s = dynamic_cast<const Slider *>(elem);
				//if (ve->label)
					//ve->label->setText(sliderText(s));
			}
			*/
		}
		//! reflect changed button state in graphical representation
		void CMFCView::updateState(const opencover::ui::Button *button)
		{
			/*auto o = qtObject(button);
			auto a = dynamic_cast<QAction *>(o);
			if (a)
				a->setChecked(button->state());
				*/
		}
		//! reflect change of child tems in graphical representation
		void CMFCView::updateChildren(const opencover::ui::Menu *menu)
		{

#if 0
			auto o = qtObject(menu);
			auto m = dynamic_cast<QMenu *>(o);
			if (m)
			{
				m->clear();
				for (size_t i = 0; i<menu->numChildren(); ++i)
				{
					auto ve = qtViewElement(menu->child(i));
					if (!ve)
						continue;
					auto obj = ve->object;
					auto act = ve->action;
					if (auto a = dynamic_cast<QAction *>(obj))
						m->addAction(a);
					else if (auto mm = dynamic_cast<QMenu *>(obj))
						m->addMenu(mm);
					else if (auto ag = dynamic_cast<QActionGroup *>(obj))
						m->addActions(ag->actions());
				}
			}
#endif
		}
		//! reflect change of slider type in graphical representation
		void CMFCView::updateInteger(const opencover::ui::Slider *slider)
		{
			updateBounds(slider);
			updateValue(slider);
		}
		//! reflect change of slider value in graphical representation
		void CMFCView::updateValue(const opencover::ui::Slider *slider)
		{
			/*auto ve = qtViewElement(slider);
			auto o = qtObject(slider);
			auto s = dynamic_cast<QSlider *>(o);
			if (s)
			{
				if (slider->integer())
				{
					s->setValue(slider->value());
				}
				else
				{
					auto min = slider->min();
					auto r = slider->max() - min;
					s->setValue((slider->value() - min) / r*SliderIntMax);
				}
			}
			if (ve && ve->label)
				ve->label->setText(sliderText(slider));
				*/
		}
		//! reflect change of slider range in graphical representation
		void CMFCView::updateBounds(const opencover::ui::Slider *slider)
		{/*
			auto o = qtObject(slider);
			auto s = dynamic_cast<QSlider *>(o);
			if (s)
			{
				if (slider->integer())
				{
					s->setRange(slider->min(), slider->max());
				}
				else
				{
					s->setRange(0, SliderIntMax);
					updateValue(slider);
				}
			}
			*/
		}

		mfcViewElement *CMFCView::mfcViewElement(const Element *elem) const
		{
			auto ve = viewElement(elem);
			return dynamic_cast<opencover::ui::mfcViewElement *>(ve);
		}

		std::wstring CMFCView::toWide(const std::string & source)
		{
			// Assumes std::string is encoded in the current Windows ANSI codepage
			int bufferlen = ::MultiByteToWideChar(CP_ACP, 0, source.c_str(), source.size(), NULL, 0);

			// Allocate new LPWSTR - must deallocate it later
			LPWSTR widestr = new WCHAR[bufferlen + 1];

			::MultiByteToWideChar(CP_ACP, 0, source.c_str(), source.size(), widestr, bufferlen);

			// Ensure wide string is null terminated
			widestr[bufferlen] = 0;
			return std::wstring(widestr);
		}

#if 0
		QWidget *CMFCView::mfcContainerWidget(const opencover::ui::Element *elem) const
		{
			auto parent = viewParent(elem);
			auto ve = qtViewContainer(elem);
			auto w = qtWidget(ve);
			if (!parent && !w)
				return m_menubar;
			return w;
		}

		QtViewElement *CMFCView::qtViewContainer(const Element *elem) const
		{
			auto parent = qtViewParent(elem);
			if (parent)
			{
				if (dynamic_cast<QWidget *>(parent->object))
					return parent;
				return qtViewParent(parent->element);
			}
			return nullptr;
		}
#endif

		mfcViewElement *CMFCView::mfcViewParent(const opencover::ui::Element *elem) const
		{
			auto ve = viewParent(elem);
			return dynamic_cast<opencover::ui::mfcViewElement *>(ve);
		}


		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::Menu * menu)
		{
			auto vparent = viewParent(menu);
			auto m = new CMenu;
			auto ve = new opencover::ui::mfcViewElement(menu, (CWnd*)m);
			//m->SetMenuInfo
			m->CreatePopupMenu();

			if (vparent)
			{

			}
			else
			{
				//m->AppendMenu(MF_STRING, ID_APP_EXIT, _T("E&xit"));
				m_menubar->AppendMenuW(MF_POPUP|MF_STRING, (UINT_PTR)m->m_hMenu, toWide(menu->text()).c_str());

			
			}

			//++menu_ID_Count;
			//auto parent = mfcContainerWidget(menu);


			/*
			//ve->action = m->menuAction();
			if (auto pmb = dynamic_cast<CMenu *>(m_menubar))
			{
				////	pmb->
				//CMenu * subMenu = m->GetSubMenu(4);
				pmb->GetSubMenu(0)->AppendMenuW(MF_STRING, MENU_ID + menu_ID_Count, _T("MenuName"));
			}
			else if (auto pm = dynamic_cast<CMenu *>(m_menubar))
			{
				pmb->GetSubMenu(0)->AppendMenuW(MF_STRING, MENU_ID + menu_ID_Count, _T("MenuName"));
			}
			*/
			return ve;

		}

		mfcViewElement::mfcViewElement(Element *elem, CWnd *obj)
			: View::ViewElement(elem)
			, object(obj)
		{
			//QString n = QString::fromStdString(elem->name());
			//if (object)
				//object->setObjectName(n);

			//elem->name();
			//obj->SetDlgCtrlID()
		}

		//! implement to create graphical representation of an item group (e.g. a frame, possible with a label)
		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::Group *group)
		{
			return nullptr;
		}
		//! implement to create graphical representation of an radio group of toggle buttons
		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::ButtonGroup *bg)
		{
			return nullptr;
		}
		//! implement to create graphical representation of a text label
		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::Label *label)
		{
			return nullptr;
		}
		//! implement to create graphical representation of a stateless button
		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::Action *action)
		{
			return nullptr;
		}
		//! implement to create graphical representation of a button with binary state
		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::Button *button)
		{
			auto parent = viewParent(button);
			if (!parent)
				return nullptr;

			CButton * new_button = new CButton();
			++button_ID_Count;

			new_button->Create(_T("Button text"), WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON | DT_CENTER, CRect(5, 5, 55, 19), ((CWnd*)parent), button_ID_Count);
			new_button->SetCheck(true);

			auto ve = new opencover::ui::mfcViewElement(button, (CWnd*)new_button);
			ve->object = new_button;
			//add(ve);

	
			// Add EventAction
			return ve;

			
		}
		//! implement to create graphical representation of a slider
		opencover::ui::View::ViewElement * CMFCView::elementFactoryImplementation(opencover::ui::Slider *slider)
		{
			return nullptr;
		}
	}
}