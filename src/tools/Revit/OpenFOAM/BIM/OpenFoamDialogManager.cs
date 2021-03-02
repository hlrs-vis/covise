/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using System.Diagnostics;
using System.Windows.Forms;

namespace BIM.OpenFOAMExport
{
   /// <summary>
   /// Manager class for dialogs in the project.
   /// </summary>
   public class OpenFOAMDialogManager
   {
      /// <summary>
      /// Pop up a standard SaveAs dialog.
      /// </summary>
      /// <returns>The filename return by SaveAs dialog.</returns>
      public static string SaveDialog()
      {
         // save file dialog options
         using (SaveFileDialog saveDialog = new SaveFileDialog())
         {
            saveDialog.OverwritePrompt = true;
            saveDialog.AddExtension = true;
            saveDialog.DefaultExt = OpenFOAMExportResource.SAVE_DIALOG_DEFAULT_FILE_EXTEND;
            saveDialog.Filter = OpenFOAMExportResource.SAVE_DIALOG_FILE_FILTER;

            if (System.Windows.Forms.DialogResult.OK != saveDialog.ShowDialog())
            {
               return String.Empty;
            }
            return saveDialog.FileName;
         }
      }

      /// <summary>
      /// Used to show error message when debug.
      /// </summary>
      /// <param name="exception">The exception message.</param>
      [Conditional("DEBUG")]
      public static void ShowDebug(string exception)
      {
         MessageBox.Show(exception, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                        MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
      }
   }
}
