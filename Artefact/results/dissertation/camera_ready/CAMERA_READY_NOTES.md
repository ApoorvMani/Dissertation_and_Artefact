# Camera-ready figure notes

## Main dissertation

### Figure 1
Experimental design of the primary evaluation. The clean BPE/Unigram
comparison is shown as supporting evidence only and is visually separated
from the primary WordPiece attack pipeline.

### Figure 2
Undefended attack success rates on the 156 eligible toxic inputs.

### Figure 3
Attack-by-defence outcome matrix. Cells report defended ASR.
Patterns, text and grayscale shading are redundant so the figure remains
interpretable in monochrome. "Complete" means complete suppression only
in the evaluated setting. "N/A" indicates that the corresponding
undefended attack had zero ASR.

### Figure 4
Paired TokenBreak ASR reduction with 95% confidence intervals.
This figure should be used to support the bounded claim that no tested
defence completely suppressed TokenBreak in this experiment.

### Figure 5
Complex legitimate-input utility. Panel A reports operational
preservation. Panel B reports legitimate inputs whose representation
changed. These are deliberately separated.

### Figure 6
Attack success stratified by clean victim confidence. Bars use both
grayscale and hatching, and error bars are Wilson 95% confidence
intervals.

### Figure 7
Defence-induced regression for invisible Unicode under tokenizer
translation, reporting the paired ASR increase and confidence interval.

## Appendix

### Figure A1
Recorded defence actions for AdvTok. This is an implementation-audit
view, not a claimed novel security result.

### Figure A2
Clean eligibility across different model-tokenizer setups.
Model and tokenizer change together, so this figure must not be used to
attribute the differences to tokenizer family alone.

## File formats

Use PDF in the final dissertation whenever the Word/LaTeX workflow
preserves vector graphics correctly.

SVG is provided as an alternative vector format.

PNG is a preview/fallback only and should not be used when a vector
version can be inserted.

## Print check

Before final submission:
1. print/export one page in grayscale;
2. view it at 100%;
3. verify every axis label, cell value and confidence interval remains
   legible;
4. do not enlarge a figure in the PDF viewer when deciding whether the
   text is readable at dissertation size.
