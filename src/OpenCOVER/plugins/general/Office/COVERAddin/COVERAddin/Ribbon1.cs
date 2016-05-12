using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Office.Tools.Ribbon;

namespace COVERAddin
{
    public partial class Ribbon1
    {
        private void Ribbon1_Load(object sender, RibbonUIEventArgs e)
        {

        }

        private void port_Click(object sender, RibbonControlEventArgs e)
        {
            host.Text = "visent.hlrs.de";
            port.Text = "31315";
        }

        private void host_TextChanged(object sender, RibbonControlEventArgs e)
        {
        }

        private void checkBox1_Click(object sender, RibbonControlEventArgs e)
        {

        }
    }
}
