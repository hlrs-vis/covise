#proc refreshWindowConfigGUI { arrayname arrayindex op } {
#    global ListForSection
#    if [winfo exists .wWindowConfig {
#       destroy .wWindowConfig
#   }
#    trace variable ListForSection(COVERConfig,NUM_WINDOWS) wu refreshWindowConfigGUI
#}

#trace variable ListForSection(COVERConfig,NUM_WINDOWS) wu refreshWindowConfigGUI

proc WINDOWCONFIGITEM { winNo name softPipeNo origX origY sizeX sizeY hostname} {
    global ListForSection
    set ListForSection(WindowConfig,name,$winNo,$hostname) $name
    set ListForSection(WindowConfig,softPipeNo,$winNo,$hostname) $softPipeNo
    set ListForSection(WindowConfig,origX,$winNo,$hostname) $origX
    set ListForSection(WindowConfig,origY,$winNo,$hostname) $origY
    set ListForSection(WindowConfig,sizeX,$winNo,$hostname) $sizeX
    set ListForSection(WindowConfig,sizeY,$winNo,$hostname) $sizeY
}

proc saveWindowConfig { body hostname } {
    upvar $body b
    global ListForSection
    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_WINDOWS,$hostname) } { incr i } {
        if { [canBeSaved ListForSection(WindowConfig,name,$i,$hostname)] &&
        [canBeSaved ListForSection(WindowConfig,softPipeNo,$i,$hostname)] &&
        [canBeSaved ListForSection(WindowConfig,origX,$i,$hostname)] &&
        [canBeSaved ListForSection(WindowConfig,origY,$i,$hostname)] &&
        [canBeSaved ListForSection(WindowConfig,sizeX,$i,$hostname)] &&
        [canBeSaved ListForSection(WindowConfig,sizeY,$i,$hostname)]} {
            set b "$b     $i     \
                    $ListForSection(WindowConfig,name,$i,$hostname) \
                    $ListForSection(WindowConfig,softPipeNo,$i,$hostname) \
                    $ListForSection(WindowConfig,origX,$i,$hostname) \
                    $ListForSection(WindowConfig,origY,$i,$hostname) \
                    $ListForSection(WindowConfig,sizeX,$i,$hostname) \
                    $ListForSection(WindowConfig,sizeY,$i,$hostname)\n"
        }
    }
}


proc saveWindowConfig { body hostname } {
    upvar $body b
    global ListForSection
    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_WINDOWS,$hostname) } { incr i } {
        set names "name,$i softPipeNo,$i origX,$i origY,$i sizeX,$i sizeY,$i"
        if { [canBeSavedList WindowConfig $names $hostname] } {
            set b "$b     $i"
            foreach name $names {
                set b  "$b $ListForSection(WindowConfig,$name,$hostname)"
            }
            set b "$b\n"
        }
    }
}
      

proc makeWindowConfigGUI {hostname pk } {
    global ListForSection
    set f [getFrame WindowConfig $pk]

    button $f.default -text default -command "destroy $f.f; frame .wWindowConfig.f ; pack .wWindowConfig.f -side top; makeWindowConfigframe default"
    button .wWindowConfig.wallace -text wallace -command "destroy .wWindowConfig.f; frame .wWindowConfig.f ; pack .wWindowConfig.f -side top; makeWindowConfigframe wallace"
    pack .wWindowConfig.default .wWindowConfig.wallace  -side top
    frame .wWindowConfig.f 
    pack .wWindowConfig.f -side top
    makeWindowConfigframe default
}

proc makeWindowConfigframe { hostname } {
    global ListForSection
    set i 0
    foreach { l t } { number "Window Number" name Name softPipeNo Softpipe origX "Origin X" origY "Origin Y" sizeX "X Size" sizeY "Y Size" }  {
        label .wWindowConfig.f.$l -text $t
        grid .wWindowConfig.f.$l -row 0 -column $i -sticky w
        incr i
    }
    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_WINDOWS) } { incr i }  {
        label .wWindowConfig.f.lWinNo$i -text $i
        grid .wWindowConfig.f.lWinNo$i -row [expr $i+1] -column 0
        set j 1
        foreach e { name softPipeNo origX origY sizeX sizeY } {
            entry .wWindowConfig.f.e$e$i -textvariable ListForSection(WindowConfig,$e,$i,$hostname)
            bind  .wWindowConfig.f <Button-3> "showHelp WindowConfig $t " 
            grid .wWindowConfig.f.e$e$i -column $j -row [expr $i+1]
            incr j
        }
    }
}
proc makeWindowConfigGUI { hostname pk} {
    global ListForSection
    set f [getFrame WindowConfig $pk]
    set i 0
    foreach { l t } { number "Window Number" name Name softPipeNo Softpipe origX "Origin X" origY "Origin Y" sizeX "X Size" sizeY "Y Size" }  {
        label $f.$l -text $t
        bind  $f.$l <Button-3> "showHelp WindowConfig $l" 
        grid $f.$l -row 0 -column $i -sticky w
        incr i
    }

    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_WINDOWS,$hostname) } { incr i }  {
        label $f.lWinNo$i -text $i
        grid $f.lWinNo$i -row [expr $i+1] -column 0
        set j 1
        foreach e { name softPipeNo origX origY sizeX sizeY } {
            entry $f.e$e$i -textvariable ListForSection(WindowConfig,$e,$i,$hostname)
            grid $f.e$e$i -column $j -row [expr $i+1]
            incr j
        }
    }
}
