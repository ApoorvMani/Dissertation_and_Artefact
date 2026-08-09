# Polished dissertation figure captions

## Figure 1 — Experimental design

Experimental design of the primary WordPiece study. Of 250 toxic Jigsaw
examples, 156 were correctly classified by the victim and formed the common
eligible attack set. Six attack variants generated 936 attack instances, which
were evaluated under seven defences in 6,552 paired evaluations. Security
effectiveness, defence mechanism, and legitimate-input utility were evaluated
separately. The side branch is a supporting clean-baseline comparison across
different model–tokenizer setups; no BPE or Unigram attacks were executed.

## Figure 3 — Attack × defence coverage and mechanism

Coverage of seven defences across six attack variants. Each cell reports
defended attack success rate together with a qualitative coverage classification.
Where relevant, the cell additionally identifies whether suppression resulted
from explicit detection/blocking, repair, or re-encoding. "Complete" therefore
means complete suppression in this evaluated setting and does not imply that
all complete cells operate through equivalent mechanisms.

## Figure 4 — AdvTok mechanism decomposition

AdvTok suppression and defence mechanism. Several defences reduce defended
AdvTok ASR to zero, but they do so through different mechanisms. Canonical
rejection explicitly detects and blocks non-canonical token IDs, whereas
tokenizer translation and several string transformations suppress the attack
through re-encoding. Outcome-level robustness should therefore not be
interpreted automatically as attack detection.

## Figure 5 — TokenBreak residual coverage gap

Paired reduction in TokenBreak attack-success rate with 95% confidence
intervals. Tokenizer translation produces the largest measured reduction, but
46.5% defended ASR remains. None of the evaluated defences completely suppresses
TokenBreak in this setting.

## Figure 6 — Utility costs

Utility of the seven defences on six complex legitimate-input categories.
Panel A reports operational preservation, while Panel B reports the proportion
of legitimate inputs whose representation was changed. Reporting both
distinguishes retained classifier behaviour from preservation of the original
legitimate input representation.

## Figure 7 — Confidence-stratified attack success

Attack success stratified by the clean victim toxic-class probability, with
Wilson 95% confidence intervals. The highest-confidence bin contains 128 of
the 156 eligible examples. High ASR persists in this bin for TokenBreak,
AdvTok, homoglyph, compatibility and reorder attacks, showing that the headline
attack rates are not explained only by borderline clean classifications.

## Figure 8 — Defence-induced regression

Tokenizer translation increases invisible-Unicode ASR from 0% to 13.4% among
the clean-retained paired evaluation set. The paired difference and confidence
interval demonstrate that defensive preprocessing can introduce a failure mode
that was absent in the undefended victim.

## Figure 9 — Model–tokenizer clean eligibility

Clean toxic-example eligibility across the three evaluated model–tokenizer
setups and their shared intersection. The comparison is supporting baseline
evidence only. Because model architecture and tokenizer family change together,
differences in eligibility cannot be attributed to tokenizer family alone, and
no cross-family attack-generalisation result is claimed.
