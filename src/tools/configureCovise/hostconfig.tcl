proc HOSTCONFIGITEM { nameOfHost sharedmemory executionmode timeout hostname } {
    global ListForSection
    #FIXME further check required: 
    #the case where a hostname is given by numeric _and_ by symbolic IP-Adresses ist not
    #yet recognized. Neither the case where e.g. foo.bar.com _and_ foo is given
    if { [lsearch $ListForSection(HostConfig,hostnames,$hostname) $nameOfHost] == -1 } {
        lappend ListForSection(HostConfig,hostnames,$hostname) $nameOfHost
        set ListForSection(HostConfig,$nameOfHost,sharedmemory,$hostname) $sharedmemory
        set ListForSection(HostConfig,$nameOfHost,executionmode,$hostname) $executionmode
        set ListForSection(HostConfig,$nameOfHost,timeout,$hostname) $timeout

        computeLabelWidth HostConfig $nameOfHost
    }
}


proc makeHostConfigEntry {frame nameOfHost row hostname} {
    global ListForSection
    label $frame.nameOfHost$row -text $nameOfHost
    grid  $frame.nameOfHost$row -row $row -column 0 -sticky w
    
    frame $frame.sharedmemory$row
    foreach sharedmemory {shm map none} {
        radiobutton $frame.sharedmemory$row.r$sharedmemory -text $sharedmemory \
                -variable  ListForSection(HostConfig,$nameOfHost,sharedmemory,$hostname) \
                -value $sharedmemory
        pack $frame.sharedmemory$row.r$sharedmemory -side left
    }
    frame $frame.sharedmemory$row.space -width 15
    pack  $frame.sharedmemory$row.space -side left
    grid $frame.sharedmemory$row -row $row -column 1 
    
    frame $frame.executionmode$row
    foreach executionmode {rexec rsh ssh covised manual} {
        radiobutton $frame.executionmode$row.r$executionmode -text $executionmode \
                -variable  ListForSection(HostConfig,$nameOfHost,executionmode,$hostname) \
                -value $executionmode
        pack $frame.executionmode$row.r$executionmode -side left
    }
    grid $frame.executionmode$row -row $row -column 2
    
    
    entry $frame.timeout$row -textvariable ListForSection(HostConfig,$nameOfHost,timeout,$hostname)
    grid $frame.timeout$row -row $row -column 3
    
    button $frame.delete$row -text delete -command "deleteHostConfigEntry $frame $row $hostname"
    grid $frame.delete$row -row $row -column 4
}

proc makeHostConfigGUI { hostname pk } {
    set f [getFrame HostConfig $pk]
    global ListForSection
    global Counters
    label $f.nameOfHost -text Hostname
    label $f.executionmode -text Executionmode
    label $f.sharedmemory -text "Shared Memory"
    label $f.timeout -text "Timeout in seconds"
    grid  $f.nameOfHost -row 0 -column 0 -sticky w
    grid  $f.executionmode -row 0 -column 1 -sticky w
    grid  $f.sharedmemory -row 0 -column 2 -sticky w
    grid  $f.timeout -row 0 -column 3 -sticky w

    set row 1
    foreach nameOfHost $ListForSection(HostConfig,hostnames,$hostname) {
        makeHostConfigEntry $f $nameOfHost $row $hostname
        incr row
    }
    set Counters(HostConfigRows,$hostname) $row
    
#    button $f.destroy -command "set Counters(HostConfigRows) 0; destroy $f" -text "Back to main"
#    grid  $f.destroy -row 0 -column 5 -sticky w

    button $f.add -command "" -text "Add" -command "addHostConfigEntryDialogue $f $hostname"
    grid  $f.add -row 1 -column 5 -sticky w
}


#For interactively adding a new HostConfig entry
proc addHostConfigEntryDialogue { frame hostname } {
    global currentNameOfHost
    global Counters
    set currentNameOfHost ""
    toplevel .waddHostConfigEntry
    message .waddHostConfigEntry.m -text "Insert the hostname" -width 100
    pack  .waddHostConfigEntry.m -side top
    entry .waddHostConfigEntry.hostname  -textvariable currentNameOfHost -width 50
    pack  .waddHostConfigEntry.hostname  -side top
    frame .waddHostConfigEntry.f
    button .waddHostConfigEntry.f.ok -text "OK" -command " HOSTCONFIGITEM \$currentNameOfHost shm rexec 30 $hostname; \
            makeHostConfigEntry $frame \$currentNameOfHost \$Counters(HostConfigRows,$hostname) $hostname;incr Counters(HostConfigRows,$hostname); \
            destroy  .waddHostConfigEntry"  -width 15 
    bind .waddHostConfigEntry.hostname <Return> " HOSTCONFIGITEM \$currentNameOfHost shm rexec 30 $hostname; \
            makeHostConfigEntry $frame \$currentNameOfHost \$Counters(HostConfigRows,$hostname) $hostname;incr Counters(HostConfigRows,$hostname); \
            destroy  .waddHostConfigEntry"
    button .waddHostConfigEntry.f.cancel -text "Cancel" -command "destroy  .waddHostConfigEntry" -width 15 
    pack .waddHostConfigEntry.f.ok -side left
    pack .waddHostConfigEntry.f.cancel -side right
    pack .waddHostConfigEntry.f -side top -fill x 
    tkwait visibility .waddHostConfigEntry
    grab  .waddHostConfigEntry
    focus .waddHostConfigEntry.hostname 
}




proc deleteHostConfigEntry { frame row hostname} {
    #first remove the hostname and the according
    #data from the lists and other data structures
    set nameOfHost [ $frame.nameOfHost$row cget -text ]
    
    global ListForSection
    set pos [lsearch $ListForSection(HostConfig,hostnames,$hostname) $nameOfHost]

    set ListForSection(HostConfig,hostnames) [lreplace $ListForSection(HostConfig,hostnames,$hostname) $pos $pos]

    unset ListForSection(HostConfig,$nameOfHost,sharedmemory,$hostname)
    unset ListForSection(HostConfig,$nameOfHost,executionmode,$hostname)
    unset ListForSection(HostConfig,$nameOfHost,timeout,$hostname)
    
    #destroy the widgets according to the entry
    #Note that when we delete an entry
    #the numbers of the rows are not set up newly 
    #i.e. when e.g. row number 3 is deleted 
    #row number 4 comes after fow number 2
    #therefore when we add a new entry the number of its row
    #must be NOT computed by counting the elements of ListForSection(HostConfig,hostnames)
    
    destroy  $frame.nameOfHost$row
    destroy  $frame.sharedmemory$row
    destroy  $frame.executionmode$row
    destroy  $frame.timeout$row
    destroy  $frame.delete$row
}

proc saveHostConfig { body hostname } {
    upvar $body b
    global ListForSection
    foreach nameOfhost $ListForSection(HostConfig,hostnames,$hostname) {
        set b "$b     $nameOfhost $ListForSection(HostConfig,$nameOfhost,sharedmemory,$hostname) \
               $ListForSection(HostConfig,$nameOfhost,executionmode,$hostname) \
               $ListForSection(HostConfig,$nameOfhost,timeout,$hostname)\n"
    }
}

