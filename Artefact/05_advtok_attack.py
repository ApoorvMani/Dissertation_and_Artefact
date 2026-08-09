"""
Step 5 - AdvTok adapted to the DistilBERT toxicity classifier.

Algorithm 2 (Geh, Shao & Van den Broeck, ACL 2025):
    for i = 1..k:   v <- argmax_{u in Ne(v)} p(target | u)

Two adaptations, both forced by attacking a classifier instead of a chat LLM:
  - objective becomes  min P(toxic | u)   instead of max p_LLM(r | q, u)
  - vocabulary is WordPiece, not the BPE the paper assumes

The visible string never changes. Only the token IDs change, and they go
straight to the model so the tokenizer never gets to undo the attack.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import json
import random
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from attack_schema import build_record, boundary_distance

VICTIM = "martin-ha/toxic-comment-model"
MAX_ITERS      = 25    # k in algorithm 2 - early stopping means we rarely hit it
NEIGHBOR_SAMPLE = 128  # paper samples 128 from Ne(v); lower this if too slow
MAX_TOKENS     = 510   # 512 minus [CLS] and [SEP]
BATCH          = 32
SEED           = 42
RANDOM_INIT    = True  # paper found random seeding beat canonical seeding

rng = random.Random(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(VICTIM)
model = AutoModelForSequenceClassification.from_pretrained(VICTIM).to(device).eval()

VOCAB = tokenizer.get_vocab()
CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
UNK = tokenizer.unk_token
TOXIC = 1  # confirmed from config.id2label in step 1

# wordpiece gives up and emits [UNK] on very long words
MAX_CHARS_PER_WORD = getattr(tokenizer.backend_tokenizer.model,
                             "max_input_chars_per_word", 100)

_normalizer = tokenizer.backend_tokenizer.normalizer
_pretok = tokenizer.backend_tokenizer.pre_tokenizer


# --------------------------------------------------------------- tokenization
def piece_text(p):
    """Strip the ## continuation marker to get the raw characters."""
    return p[2:] if p.startswith("##") else p


def canonical_pieces(word):
    """Real WordPiece: greedy longest-match first, [UNK] if it dead-ends."""
    if len(word) > MAX_CHARS_PER_WORD:
        return [UNK]
    pieces, start = [], 0
    while start < len(word):
        end, found = len(word), None
        while start < end:
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in VOCAB:
                found = sub
                break
            end -= 1
        if found is None:
            return [UNK]        # no valid split - whole word becomes [UNK]
        pieces.append(found)
        start = end
    return pieces


def random_pieces(word):
    """A random valid segmentation, walking left to right and picking freely.
    Not uniform over all segmentations, but the paper's ablation shows the
    seed matters far less than the neighbour count."""
    if len(word) > MAX_CHARS_PER_WORD:
        return [UNK]
    pieces, start = [], 0
    while start < len(word):
        options = []
        for end in range(start + 1, len(word) + 1):
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in VOCAB:
                options.append((sub, end))
        if not options:
            return canonical_pieces(word)   # dead end - fall back
        sub, end = rng.choice(options)
        pieces.append(sub)
        start = end
    return pieces


def split_words(text):
    """Normalise and pre-tokenize exactly as the tokenizer does, so our
    reconstruction can be checked against it."""
    norm = _normalizer.normalize_str(text)
    return [w for w, _span in _pretok.pre_tokenize_str(norm)]


def to_ids(seg):
    """Flatten a list-of-words-of-pieces into a flat token-ID list."""
    return [VOCAB.get(p, VOCAB[UNK]) for word in seg for p in word]


# --------------------------------------------------------------- parity check
def parity_check(texts):
    """Halt unless our canonical rebuild matches the tokenizer EXACTLY.
    Without full equality, a buggy rebuild would masquerade as a
    'noncanonical' tokenization and every result would be meaningless."""
    for t in texts:
        mine = to_ids([canonical_pieces(w) for w in split_words(t)])
        theirs = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
        if mine != theirs:
            raise RuntimeError(
                f"PARITY FAILURE\n  text: {t[:100]}\n"
                f"  mine  ({len(mine)}): {mine[:15]}\n"
                f"  theirs({len(theirs)}): {theirs[:15]}")
    print(f"parity check passed on {len(texts)} texts\n")


# --------------------------------------------------------------- scoring
@torch.no_grad()
def p_toxic(id_seqs):
    """P(toxic) for raw token-ID sequences. IDs go straight to the model -
    the tokenizer is never consulted, which is the whole point."""
    out = []
    for i in range(0, len(id_seqs), BATCH):
        chunk = id_seqs[i:i + BATCH]
        width = max(len(s) for s in chunk) + 2
        ids = torch.full((len(chunk), width), PAD, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, s in enumerate(chunk):
            seq = [CLS] + s + [SEP]
            ids[j, :len(seq)] = torch.tensor(seq)
            attn[j, :len(seq)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
        out.extend(torch.softmax(logits, dim=-1)[:, TOXIC].cpu().tolist())
    return out


# --------------------------------------------------------------- neighbourhood
def word_neighbours(word, pieces):
    """Every way to replace one contiguous run of pieces with 1 or 2 new pieces
    covering the same characters.

    Cost = number of pieces inserted (deletions are free), so 1 piece is a
    merge at distance 1 and 2 pieces is a split at distance 2. This is the
    paper's Ne(v) restricted to a single edit site."""
    starts, off = [], 0
    for p in pieces:                      # char offset where each piece begins
        starts.append(off)
        off += len(piece_text(p))
    starts.append(off)

    out = []
    for i in range(len(pieces)):
        for j in range(i, len(pieces)):
            a, b = starts[i], starts[j + 1]
            span = word[a:b]

            # merge the whole run into one piece (cost 1)
            if j > i:
                one = span if a == 0 else "##" + span
                if one in VOCAB:
                    out.append((i, j, [one]))

            # replace the run with two pieces (cost 2)
            for cut in range(1, len(span)):
                left = span[:cut] if a == 0 else "##" + span[:cut]
                right = "##" + span[cut:]
                if left in VOCAB and right in VOCAB:
                    new = [left, right]
                    if new != pieces[i:j + 1]:
                        out.append((i, j, new))
    return out


def neighbourhood(words, seg):
    """Collect Ne(v) across every word, as (word_index, new_pieces) moves."""
    moves = []
    for wi, (word, pieces) in enumerate(zip(words, seg)):
        if pieces == [UNK]:
            continue                       # can't resegment an [UNK]
        for i, j, new in word_neighbours(word, pieces):
            moves.append((wi, pieces[:i] + new + pieces[j + 1:]))
    return moves


# --------------------------------------------------------------- algorithm 2
def advtok(words, seg):
    """Greedy local search. Each iteration samples the neighbourhood, takes the
    single best move, and stops at a local optimum or once the classifier flips."""
    current = [list(w) for w in seg]
    score = p_toxic([to_ids(current)])[0]
    p_seed = score          # score at the seed, before any greedy move
    n_moves = 0

    for _ in range(MAX_ITERS):
        moves = neighbourhood(words, current)
        if not moves:
            break
        if len(moves) > NEIGHBOR_SAMPLE:   # paper samples 128 without replacement
            moves = rng.sample(moves, NEIGHBOR_SAMPLE)

        # build every candidate, dropping any that overflow the model
        cands, seqs = [], []
        for wi, new_pieces in moves:
            trial = list(current)
            trial[wi] = new_pieces
            ids = to_ids(trial)
            if len(ids) <= MAX_TOKENS:
                cands.append((wi, new_pieces))
                seqs.append(ids)
        if not seqs:
            break

        scores = p_toxic(seqs)
        best = int(np.argmin(scores))
        if scores[best] >= score - 1e-9:
            break                          # local optimum, no improving move

        wi, new_pieces = cands[best]
        current[wi] = new_pieces
        score = scores[best]
        n_moves += 1
        if score < 0.5:                    # classifier flipped - done
            break

    return current, score, n_moves, p_seed


# --------------------------------------------------------------- run
if __name__ == "__main__":
    df = pd.read_csv("data/clean_baseline_toxic.csv")
    eligible = df[df["clean_prediction"] == 1].reset_index(drop=True)
    print(f"attacking {len(eligible)} eligible rows (same set as tokenbreak)\n")

    parity_check(eligible["text"].tolist())

    rows = []
    for n, r in eligible.iterrows():
        words = split_words(r["text"])
        canon = [canonical_pieces(w) for w in words]

        # drop trailing words until the canonical form fits the model
        while words and len(to_ids(canon)) > MAX_TOKENS:
            words.pop(); canon.pop()
        if not words:
            continue

        canon_ids = to_ids(canon)
        start_seg = [random_pieces(w) for w in words] if RANDOM_INIT else canon
        # a random seed can overflow, since it may use more pieces than canonical
        if len(to_ids(start_seg)) > MAX_TOKENS:
            start_seg = canon

        t0 = time.perf_counter()
        adv_seg, p_adv, n_moves, p_seed = advtok(words, start_seg)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        adv_ids = to_ids(adv_seg)
        p_canon = p_toxic([canon_ids])[0]
        visible = tokenizer.decode(canon_ids)

        # flatten pieces so we can compare word by word
        canon_flat = [p for w in canon for p in w]
        adv_flat = [p for w in adv_seg for p in w]

        # --- the three checks that decide whether this row counts ---
        same_text = tokenizer.decode(adv_ids) == visible
        noncanonical = adv_ids != canon_ids
        specials = {VOCAB[UNK], CLS, SEP, PAD}
        no_specials = not (set(adv_ids) & specials)
        valid = same_text and noncanonical and no_specials

        rows.append(build_record(
            attack="advtok",
            sample_id=r["sample_id"],
            original_text=visible,
            attacked_text=visible,        # identical by construction - that's the point
            # words whose segmentation differs from canonical
            modified_words=[words[i] for i in range(len(words))
                            if adv_seg[i] != canon[i]],
            inserted_characters=[],       # advtok never changes a character
            clean_p_toxic=p_canon,
            attacked_p_toxic=p_adv,
            clean_tokens=canon_flat,
            attacked_tokens=adv_flat,
            clean_ids=canon_ids,
            attacked_ids=adv_ids,
            search_latency_ms=elapsed_ms,
            extras={
                # advtok's own distance - valid here because both tokenizations
                # cover the same string, unlike the tokenbreak case
                "boundary_distance": boundary_distance(canon_flat, adv_flat),
                "moves_applied": n_moves,
                "p_toxic_seed": p_seed,
                "n_tokens_seed": len(to_ids(start_seg)),
                "seed_alone_success": int(p_seed < 0.5),
                "valid": int(valid),
                "same_text": int(same_text),
                "noncanonical": int(noncanonical),
                "no_specials": int(no_specials),
            },
        ))
        print(f"  {n + 1}/{len(eligible)}", end="\r")

    out = pd.DataFrame(rows)
    out.to_csv("data/advtok_results.csv", index=False)
    out[out["attack_success"] == 1].to_csv("data/advtok_successes.csv", index=False)

    v = out[out["valid"] == 1]
    print(f"\n\n--- advtok (algorithm 2) ---")
    print(f"attempted            : {len(out)}")
    print(f"valid noncanonical   : {len(v)}  ({len(v)/len(out):.3f})")
    print(f"  same text failures : {(out['same_text'] == 0).sum()}")
    print(f"  unchanged ids      : {(out['noncanonical'] == 0).sum()}")
    print(f"  special-token hits : {(out['no_specials'] == 0).sum()}")
    if len(v):
        print(f"successful (of valid): {v['attack_success'].sum()}  "
              f"ASR {v['attack_success'].mean():.3f}")
        print(f"mean conf drop       : "
              f"{(v['clean_toxic_prob'] - v['attacked_toxic_prob']).mean():.4f}")
        print(f"mean moves applied   : {v['moves_applied'].mean():.1f}")
        print(f"mean tokens can/adv  : {v['clean_token_count'].mean():.1f} / "
              f"{v['attacked_token_count'].mean():.1f}")
        print(f"seed alone succeeded : {v['seed_alone_success'].sum()}  "
              f"(search added {v['attack_success'].sum() - v['seed_alone_success'].sum()})")