/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using System.Diagnostics;
using System.Windows.Forms;
using System.Security;
using System.IO;

namespace OpenFOAMInterface.BIM
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
            using SaveFileDialog saveDialog = new();
            saveDialog.OverwritePrompt = true;
            saveDialog.AddExtension = true;
            saveDialog.DefaultExt = OpenFOAMInterfaceResource.SAVE_DIALOG_DEFAULT_FILE_EXTEND;
            saveDialog.Filter = OpenFOAMInterfaceResource.SAVE_DIALOG_FILE_FILTER;

            if (DialogResult.OK != saveDialog.ShowDialog())
            {
                return string.Empty;
            }
            return saveDialog.FileName;
        }

        /// <summary>
        /// Used to show error message when debug.
        /// </summary>
        /// <param name="exception">The exception message.</param>
        [Conditional("DEBUG")]
        public static void ShowDebug(in string exception)
        {
            ShowError(exception);
        }

        /// <summary>
        /// Used to show OpenFOAMInterfaceResource error message.
        /// </summary>
        /// <param name="opErrMsg">The error message.</param>
        public static void ShowError(in string opErrMsg)
        {
            MessageBox.Show(opErrMsg,
                            OpenFOAMInterfaceResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Exclamation);
        }

        /// <summary>
        /// Shows error dialog for corresponding exception.
        /// </summary>
        /// <param name="e">Catched exception.</param>
        public static void ShowDialogException(in Exception e)
        {
            switch (e)
            {
                case FormatException:
                    ShowError(OpenFOAMInterfaceResource.ERR_FORMAT);
                    break;
                case IOException:
                    ShowError(OpenFOAMInterfaceResource.ERR_IO_EXCEPTION);
                    break;
                case SecurityException:
                    ShowError(OpenFOAMInterfaceResource.ERR_SECURITY_EXCEPTION);
                    break;
                default:
                    ShowError(OpenFOAMInterfaceResource.ERR_EXCEPTION + 
                    "\nUnregistered exception: " + e.ToString());
                    break;
            }
        }
    }
}
