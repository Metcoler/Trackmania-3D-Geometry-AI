all: main.pdf

main.pdf: main.tex variables.tex preamble.tex definitions.tex references.bib chapters/*.tex appendices/*.tex images/zadanie.pdf
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	del /Q *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot 2>NUL
