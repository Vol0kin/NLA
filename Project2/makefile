.PHONY: clean zip
IN = report
FILES = $(IN).aux $(IN).log $(IN).out $(IN).synctex.gz $(IN).toc
CXX = pdflatex

all: $(IN).pdf

$(IN).pdf: $(IN).tex
	$(CXX) $<
	$(CXX) $<

clean:
	@echo Limpiando archivos extra generados por LaTex...
	rm -f $(FILES)

zip:
	tar cvfz project2.tar.gz report.pdf lsp.py image_compression.py pca.py *.jpeg
