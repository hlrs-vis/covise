/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using Autodesk.Revit.UI;
using System.IO;
using System.Reflection;
using System.Windows.Forms;
using System.Windows.Media.Imaging;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using RevitView = Autodesk.Revit.DB.View;

namespace OpenFOAMInterface.BIM
{
    public sealed class Exporter
    {
        public Settings settings = null;
        public static Exporter Instance
        {
            get
            {
                return Nested.instance;
            }
        }
        public Exporter()
        {
        }

        /// <summary>
        /// Get view by view name.
        /// </summary>
        /// <param name="doc">The document to find the view.</param>
        /// <param name="activeViewName">The view name.</param>
        /// <returns>The element id of the view found.</returns>
        public RevitView FindView(Document doc, string activeViewName)
        {
            FilteredElementCollector collector = new(doc);
            collector.OfClass(typeof(RevitView));

            IEnumerable<Element> selectedView = from view in collector.ToList()
                                                where view.Name == activeViewName
                                                select view;

            if (selectedView.Count() > 0)
            {
                return selectedView.First() as RevitView;
            }

            return null;
        }

        /// <summary>
        /// Get view by view name.
        /// </summary>
        /// <param name="doc">The document to find the view.</param>
        /// <param name="activeViewName">The view name.</param>
        /// <returns>The element id of the view found.</returns>
        public ElementId FindViewId(Document doc, string activeViewName)
        {
            FilteredElementCollector collector = new(doc);
            collector.OfClass(typeof(RevitView));

            IEnumerable<Element> selectedView = from view in collector.ToList()
                                                where view.Name == activeViewName
                                                select view;

            if (selectedView.Count() > 0)
            {
                return (selectedView.First() as RevitView).Id;
            }

            return ElementId.InvalidElementId;
        }
        class Nested
        {
            // Explicit static constructor to tell C# compiler
            // not to mark type as beforefieldinit
            static Nested()
            {
            }
            internal static readonly Exporter instance = new Exporter();
        }
    }
    public class OpenFOAMInterfaceApp : IExternalApplication
    {
        // Fields
        private static string AddInPath;

        // Methods
        static OpenFOAMInterfaceApp()
        {
            AddInPath = typeof(OpenFOAMInterfaceApp).Assembly.Location;
        }

        Result IExternalApplication.OnShutdown(UIControlledApplication application)
        {
            return Result.Succeeded;
        }

        Result IExternalApplication.OnStartup(UIControlledApplication application)
        {
            try
            {

                Exporter.Instance.settings = new(SaveFormat.ascii, ElementsExportRange.OnlyVisibleOnes, true,
                    false, false,
                    false, 0, 101, 1, 100, 2/*purgeWrite*/, 8, 7, 4);

                string str = "OpenFOAM Interface";
                RibbonPanel panel = application.CreateRibbonPanel(str);
                string directoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

                string assemblyname = typeof(OpenFOAMInterfaceApp).Assembly.GetName().Name;
                string dllName = directoryName + @"\" + assemblyname + ".dll";

                PushButtonData simBtnData = new("OpenFOAM Simulate", "Simulate", dllName, "OpenFOAMInterface.BIM.OpenFOAMSimulateCommand");
                PushButton simulateButton = panel.AddItem(simBtnData) as PushButton;
                using (Stream xstr = new MemoryStream())
                {
                    Properties.Resources.openfoaminterface.Save(xstr, System.Drawing.Imaging.ImageFormat.Bmp);
                    xstr.Seek(0, SeekOrigin.Begin);
                    BitmapDecoder bdc = new BmpBitmapDecoder(xstr, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
                    simulateButton.LargeImage = bdc.Frames[0];
                }
                simulateButton.ToolTip = "The OpenFOAM Interface for Revit is designed to produce a stereolithography file (STL) of your building model and a OpenFOAM-Config.";
                simulateButton.LongDescription = "The OpenFOAM Iterface for Autodesk Revit is a project designed to create an STL file from a 3D building information model for OpenFOAM with a Config-File that includes the boundary conditions for airflow simulations.";
                ContextualHelp help = new(ContextualHelpType.ChmFile, directoryName + @"\Resources\OpenFoamInterfaceHelp.html");
                simulateButton.SetContextualHelp(help);

                return Result.Succeeded;
            }
            catch (Exception exception)
            {
                MessageBox.Show(exception.ToString(), "OpenFOAM Interface for Revit");
                return Result.Failed;
            }
        }

        private static System.Windows.Media.ImageSource LoadPNGImageFromResource(string imageResourceName)
        {
            string[] names = Assembly.GetExecutingAssembly().GetManifestResourceNames();
            int i = 0;
            foreach (var name in names)
            {

                MessageBox.Show(names[i], name);
                i++;
            }
            PngBitmapDecoder decoder = new(Assembly.GetExecutingAssembly().GetManifestResourceStream(imageResourceName), BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.Default);
            return decoder.Frames[0];
        }
    }
}
