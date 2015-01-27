TOOLTIPS = $(COVISEDIR)/doc/moduledescriptions.txt

all: $(TOOLTIPS)

$(TOOLTIPS): $(COVISEDIR)/$(ARCHSUFFIX)/bin/*/*
	$(COVISEDIR)/doc/scripts/maketooltips.sh > $(TOOLTIPS)

clean:
	$(RM) $(TOOLTIPS)
