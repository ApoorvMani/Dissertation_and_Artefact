# Suggested dissertation captions

## Figure 1 — Experimental design

Experimental design of the primary study. Jigsaw toxic examples correctly
classified by the WordPiece victim formed the eligible attack set. Six attack
variants produced 936 attack instances, which were evaluated under seven
defences using a paired clean-versus-attacked design. Security effectiveness,
defence mechanism, and legitimate-input utility were evaluated separately.
The BPE and Unigram branch represents clean-baseline generalisation groundwork
only; no cross-family attack evaluation was executed.

## Figure 2 — Undefended attack effectiveness

Undefended attack success rate on the 156 eligible WordPiece toxic inputs.
TokenBreak and AdvTok each achieved 96.2% ASR, while three visible Unicode
manipulation families achieved high ASR. Invisible Unicode achieved 0% in the
evaluated victim because the perturbations did not alter the model-bound token
IDs.

## Figure 3 — Attack × defence coverage matrix

Coverage of seven defences across six attack variants. Each cell reports the
outcome classification together with defended ASR. "Complete" denotes complete
suppression in this evaluated setting rather than a universal claim that an
attack is solved. "No base attack" denotes an attack whose undefended ASR was
already zero.

## Figure 4a — Detection

Attack detection rate by attack and defence. Detection is separated from attack
suppression so that successful defence outcomes are not automatically
interpreted as explicit attack detection.

## Figure 4b — Blocking

Attack blocking rate by attack and defence. Blocking is treated separately from
repair or re-encoding.

## Figure 4c — Re-encoding

Re-encoding rate by attack and defence. This exposes cases in which apparent
robustness arises because adversarial token IDs are replaced by a canonical
string-to-token encoding rather than because the attack was detected.

## Figure 5 — TokenBreak defence effectiveness

Paired reduction in TokenBreak attack-success rate with 95% confidence
intervals. Tokenizer translation gives the largest reduction but does not
eliminate TokenBreak, illustrating the remaining coverage gap.

## Figure 6 — Complex legitimate-input utility

Operational preservation of seven defences across six complex legitimate-input
categories. The figure exposes security-utility trade-offs that are hidden by
ordinary clean benign controls, particularly for CPT-based defences on URLs,
non-English, mixed-script, and code inputs.

## Figure 7 — Margin-stratified attack success

Attack success stratified by the clean victim's toxic-class probability.
Wilson 95% confidence intervals are shown. High attack success persists among
the 128 high-confidence examples with clean P(toxic)>0.80, indicating that the
headline ASRs are not explained only by borderline classifier decisions.

## Figure 8 — Defence-induced regression

Tokenizer translation introduces successful invisible-Unicode evasions even
though the same attack has zero undefended ASR in the evaluated WordPiece
victim. This demonstrates that defensive preprocessing can itself create a new
failure mode.

## Figure 9 — Cross-tokenizer clean baseline

Clean toxic-example eligibility under WordPiece, BPE, and Unigram classifier
setups, including the shared eligible intersection. This is clean-baseline
generalisation groundwork only; no BPE or Unigram attack-generalisation result
is claimed.

## Figure 10 — Research evolution

Research evolution from the preliminary cross-tokenizer disagreement programme
to the current attack × defence study. The preliminary programme is retained as
evidence of research development, negative-result handling, and reproducibility
discipline; its numerical results are not pooled with the primary study.
