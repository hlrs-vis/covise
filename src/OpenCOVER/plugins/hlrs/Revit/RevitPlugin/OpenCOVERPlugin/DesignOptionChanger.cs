using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Automation;
using System.Windows.Forms;
using System.Windows.Input;

namespace DesignOptionModifier
{
    public partial class Switcher
    {
        #region Windows API

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetForegroundWindow(
          IntPtr hWnd);

        [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        static extern int GetWindowText(
          IntPtr hWnd, [Out] StringBuilder lpString,
          int nMaxCount);

        [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        static extern int GetWindowTextLength(
          IntPtr hWnd);

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool EnumChildWindows(
          IntPtr window, EnumWindowProc callback, IntPtr i);

        [DllImport("user32.dll", EntryPoint = "GetClassName")]
        public static extern int GetClass(
          IntPtr hWnd, StringBuilder className, int nMaxCount);

        public delegate bool EnumWindowProc(
          IntPtr hWnd, IntPtr parameter);

        [DllImport("user32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetWindowRect(
          IntPtr hWnd, out RECT lpRect);

        public static string GetText(IntPtr hWnd)
        {
            int length = GetWindowTextLength(hWnd);
            StringBuilder sb = new StringBuilder(length + 1);
            GetWindowText(hWnd, sb, sb.Capacity);
            return sb.ToString();
        }

        private static bool EnumWindow(
          IntPtr handle,
          IntPtr pointer)
        {
            GCHandle gch = GCHandle.FromIntPtr(pointer);
            List<IntPtr> list = gch.Target as List<IntPtr>;
            if (list != null)
            {
                list.Add(handle);
            }
            return true;
        }

        public static List<IntPtr> GetChildWindows(
          IntPtr parent)
        {
            List<IntPtr> result = new List<IntPtr>();
            GCHandle listHandle = GCHandle.Alloc(result);
            try
            {
                EnumWindowProc childProc = new EnumWindowProc(EnumWindow);
                EnumChildWindows(parent, childProc, GCHandle.ToIntPtr(listHandle));
            }
            finally
            {
                if (listHandle.IsAllocated)
                    listHandle.Free();
            }
            return result;
        }

        internal struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }
        #endregion

        /// <summary>
        /// Cache combobox
        /// </summary>
        private AutomationElement comboBoxElement = null;

        public Switcher()
        {

        }

        

        private void GetComboBox()
        {
            int maxX = -1;

            Process[] revits = Process.GetProcessesByName(
              "Revit");

            IntPtr cb = IntPtr.Zero;

            if (revits.Length > 0)
            {
                List<IntPtr> children = GetChildWindows(
                  revits[0].MainWindowHandle);

                foreach (IntPtr child in children)
                {
                    StringBuilder classNameBuffer
                      = new StringBuilder(100);

                    int className = GetClass(child,
                      classNameBuffer, 100);

                    if (classNameBuffer.ToString().Contains(
                      "msctls_statusbar32"))
                    {
                        List<IntPtr> grandChildren
                          = GetChildWindows(child);

                        foreach (IntPtr grandChild in grandChildren)
                        {
                            StringBuilder classNameBuffer2
                              = new StringBuilder(100);

                            int className2 = GetClass(grandChild,
                              classNameBuffer2, 100);

                            if (classNameBuffer2.ToString().Contains(
                              "ComboBox"))
                            {
                                RECT r;

                                GetWindowRect(grandChild, out r);

                                // There are at least two comboboxes, 
                                // and we want the rightmost one.


                                if (r.Left > maxX)
                                {
                                    comboBoxElement = AutomationElement.FromHandle(
                  grandChild);
                                    if (comboBoxElement!=null && comboBoxElement.Current.IsEnabled)
                                    {
                                        maxX = r.Left;
                                        cb = grandChild;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (cb != IntPtr.Zero)
            {
                comboBoxElement = AutomationElement.FromHandle(
                  cb);
            }
        }

        public void SetSelection(string option)
        {
            if(comboBoxElement == null)
            {
                GetComboBox();
            }
            

            if ((comboBoxElement != null)
              && (comboBoxElement.Current.IsEnabled))
            {
                AutomationElement sel = null;

                ExpandCollapsePattern expandPattern
                  = (ExpandCollapsePattern)comboBoxElement
                    .GetCurrentPattern(
                      ExpandCollapsePattern.Pattern);

                expandPattern.Expand();

                CacheRequest cacheRequest = new CacheRequest();
                cacheRequest.Add(AutomationElement.NameProperty);
                cacheRequest.TreeScope = TreeScope.Element
                  | TreeScope.Children;

                AutomationElement comboboxItems
                  = comboBoxElement.GetUpdatedCache(
                    cacheRequest);

                foreach (AutomationElement item
                  in comboboxItems.CachedChildren)
                {
                    if (item.Current.Name == "")
                    {
                        CacheRequest cacheRequest2 = new CacheRequest();
                        cacheRequest2.Add(AutomationElement.NameProperty);
                        cacheRequest2.TreeScope = TreeScope.Element
                          | TreeScope.Children;

                        AutomationElement comboboxItems2
                          = item.GetUpdatedCache(cacheRequest);

                        foreach (AutomationElement item2
                          in comboboxItems2.CachedChildren)
                        {
                            if (item2.Current.Name == option)
                            {
                                sel = item2;
                            }
                        }
                    }
                }

                if (sel != null)
                {
                    SelectionItemPattern select =
                      (SelectionItemPattern)sel.GetCurrentPattern(
                        SelectionItemPattern.Pattern);

                    select.Select();
                }
                expandPattern.Collapse();
            }
        }

    }
}