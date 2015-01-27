proc UIShortCuts { stringList hostname } {
    puts "stringlist = $stringList, hostname=$hostname"
    global ListForSection
    set ListForSection(UIConfig,ShortCuts,$hostname) $stringList
}

