"""
Perturbation Engine Module for IndoBERT Clickbait Detection System

Final perturbation design:

    Clean : 0%
    Low   : 10% eligible words
    Medium: 20% eligible words
    High  : 30% eligible words

All three levels use the SAME perturbation method:

    semantic similarity-based word substitution

Candidate generation:
    Indonesian Tesaurus

Candidate filtering:
    1. Same POS
    2. Exclude original word
    3. Exclude antonyms
    4. IndoBERT cosine similarity in [0.80, 0.95]

Important:
    The 0.80-0.95 cosine constraint is applied to:

        original_word <-> replacement_word

    NOT:

        original_headline <-> perturbed_headline

The headline-level similarity is optional metadata only.
"""

import os
import json
import random
import re
import logging

from pathlib import Path
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

from debug_logger import (
    dbg_perturbation_samples,
    dbg_perturbation_stats,
)

from tqdm.auto import tqdm

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# FINAL EXPERIMENT CONFIGURATION
# =============================================================================

SIM_MIN = 0.80
SIM_MAX = 0.95

PERTURBATION_INTENSITIES = {
    "low": 0.10,
    "medium": 0.20,
    "high": 0.30,
}


# =============================================================================
# WORD TOKENIZATION
# =============================================================================

WORD_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+",
    flags=re.UNICODE,
)


def tokenize_words(text: str) -> List[str]:
    """
    Extract alphabetic word tokens.

    Punctuation is excluded from the perturbation count.
    """
    return WORD_PATTERN.findall(text)

def tokenize_word_spans(
    text: str,
) -> List[Dict[str, object]]:
    """
    Return words together with their character positions.

    Example:

        "Bagaimana pengobatannya?"

    returns approximately:

        [
            {
                "word": "Bagaimana",
                "start": 0,
                "end": 9
            },
            {
                "word": "pengobatannya",
                "start": 10,
                "end": 23
            }
        ]
    """

    return [
        {
            "word": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in WORD_PATTERN.finditer(text)
    ]

# =============================================================================
# INDOBERT CONTEXTUAL SEMANTIC SIMILARITY
# =============================================================================

class _SimilarityChecker:
    """
    IndoBERT contextual similarity checker.

    Optimization:
        1. Original contextual embedding is calculated ONCE per target word.
        2. Candidate sentences are encoded in a batch.
        3. Target-span embeddings are extracted from the batch.
        4. Cosine similarities are calculated against the single
           original embedding.

    Methodology remains unchanged:

        original word in original context
                    VS
        candidate word in modified context

    The target span is represented by the mean of the hidden states
    belonging to the target span.
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        # self.device = "cpu"
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.batch_size = batch_size

        logger.info(
            "Loading IndoBERT contextual similarity model: %s",
            model_name,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        logger.info(
            "IndoBERT contextual similarity model loaded on %s",
            self.device,
        )

        logger.info(
            "Similarity inference device: %s",
            self.device,
        )

        logger.info(
            "Similarity batch size: %d",
            self.batch_size,
        )

    # =========================================================================
    # SINGLE TEXT EMBEDDING
    # =========================================================================

    @torch.no_grad()
    def _contextual_span_embedding(
        self,
        text: str,
        start: int,
        end: int,
    ) -> Optional[torch.Tensor]:
        """
        Extract contextual embedding for one target span.

        The full sentence is passed through IndoBERT.

        Target representation =
            mean of hidden states overlapping the target span.

        Output is L2-normalized.
        """

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        offsets = encoded.pop(
            "offset_mapping"
        )[0]

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        output = self.model(**encoded)

        hidden = output.last_hidden_state[0]

        selected_vectors = []

        for token_index, (
            token_start,
            token_end,
        ) in enumerate(offsets.tolist()):

            # Ignore special tokens
            if token_start == token_end:
                continue

            # Token overlaps target character span
            if (
                token_end > start
                and token_start < end
            ):
                selected_vectors.append(
                    hidden[token_index]
                )

        if not selected_vectors:
            return None

        pooled = torch.stack(
            selected_vectors
        ).mean(dim=0)

        pooled = torch.nn.functional.normalize(
            pooled,
            p=2,
            dim=0,
        )

        return pooled.detach().cpu()

    # =========================================================================
    # BATCHED CONTEXTUAL EMBEDDINGS
    # =========================================================================

    @torch.no_grad()
    def _contextual_span_embeddings_batch(
        self,
        texts: List[str],
        spans: List[tuple],
    ) -> List[Optional[torch.Tensor]]:
        """
        Extract contextual span embeddings for multiple texts at once.

        Args:
            texts:
                Candidate sentences.

            spans:
                Character spans corresponding to the candidate
                replacement in each sentence.

        Returns:
            List of normalized span embeddings.

        Example:

            texts = [
                "Pemerintah membuka bursa baru",
                "Pemerintah membuka market baru",
                "Pemerintah membuka pusat baru",
            ]

            spans = [
                (20, 25),
                (20, 26),
                (20, 26),
            ]

        All candidate sentences are processed in one or more
        IndoBERT batches.
        """

        if not texts:
            return []

        all_embeddings = []

        for batch_start in range(
            0,
            len(texts),
            self.batch_size,
        ):

            batch_texts = texts[
                batch_start:
                batch_start + self.batch_size
            ]

            batch_spans = spans[
                batch_start:
                batch_start + self.batch_size
            ]

            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )

            offsets = encoded.pop(
                "offset_mapping"
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            output = self.model(**encoded)

            hidden = output.last_hidden_state

            for batch_index, (
                start,
                end,
            ) in enumerate(batch_spans):

                selected_vectors = []

                token_offsets = offsets[
                    batch_index
                ].tolist()

                for token_index, (
                    token_start,
                    token_end,
                ) in enumerate(token_offsets):

                    # Ignore special tokens / padding
                    if token_start == token_end:
                        continue

                    # Token overlaps target span
                    if (
                        token_end > start
                        and token_start < end
                    ):
                        selected_vectors.append(
                            hidden[
                                batch_index,
                                token_index,
                            ]
                        )

                if not selected_vectors:

                    all_embeddings.append(None)

                    continue

                pooled = torch.stack(
                    selected_vectors
                ).mean(dim=0)

                pooled = torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=0,
                )

                all_embeddings.append(
                    pooled.detach().cpu()
                )

        return all_embeddings

    # =========================================================================
    # SPAN REPLACEMENT
    # =========================================================================

    @staticmethod
    def _replace_span(
        text: str,
        start: int,
        end: int,
        replacement: str,
    ) -> str:
        """
        Replace exact character span.
        """

        return (
            text[:start]
            + replacement
            + text[end:]
        )

    # =========================================================================
    # BATCHED CANDIDATE SCORING
    # =========================================================================

    def score_candidates(
        self,
        text: str,
        target_start: int,
        target_end: int,
        candidates: List[str],
    ) -> List[Optional[float]]:
        """
        Calculate contextual cosine similarity for multiple candidates.

        IMPORTANT:

        Original embedding:
            calculated ONCE.

        Candidate embeddings:
            calculated in batches.

        Returns:
            One cosine similarity per candidate.
        """

        if not text.strip():
            return [
                0.0
                for _ in candidates
            ]

        candidates = [
            str(candidate).strip()
            for candidate in candidates
        ]

        candidates = [
            candidate
            for candidate in candidates
            if candidate
        ]

        if not candidates:
            return []

        # =====================================================================
        # STEP 1
        # Original contextual embedding calculated ONCE
        # =====================================================================

        original_embedding = (
            self._contextual_span_embedding(
                text=text,
                start=target_start,
                end=target_end,
            )
        )

        if original_embedding is None:
            return [
                0.0
                for _ in candidates
            ]

        # =====================================================================
        # STEP 2
        # Build candidate sentences
        # =====================================================================

        candidate_texts = []
        candidate_spans = []

        for candidate in candidates:

            candidate_text = self._replace_span(
                text,
                target_start,
                target_end,
                candidate,
            )

            candidate_start = target_start

            candidate_end = (
                target_start
                + len(candidate)
            )

            candidate_texts.append(
                candidate_text
            )

            candidate_spans.append(
                (
                    candidate_start,
                    candidate_end,
                )
            )

        # =====================================================================
        # STEP 3
        # Candidate embeddings calculated in batches
        # =====================================================================

        candidate_embeddings = (
            self._contextual_span_embeddings_batch(
                texts=candidate_texts,
                spans=candidate_spans,
            )
        )

        # =====================================================================
        # STEP 4
        # Cosine similarity
        #
        # Embeddings are already L2-normalized.
        # Therefore:
        #
        # cosine(a,b) = dot(a,b)
        # =====================================================================

        scores = []

        for candidate_embedding in candidate_embeddings:

            if candidate_embedding is None:

                scores.append(0.0)

                continue

            score = torch.dot(
                original_embedding,
                candidate_embedding,
            )

            scores.append(
                float(score.item())
            )

        return scores

    # =========================================================================
    # BACKWARD-COMPATIBLE SINGLE CANDIDATE METHOD
    # =========================================================================

    def score(
        self,
        text: str,
        target_start: int,
        target_end: int,
        candidate: str,
    ) -> float:
        """
        Single-candidate interface.

        Kept for compatibility.

        For performance-critical code, use score_candidates().
        """

        scores = self.score_candidates(
            text=text,
            target_start=target_start,
            target_end=target_end,
            candidates=[candidate],
        )

        if not scores:
            return 0.0

        return scores[0]


_checker = _SimilarityChecker(
    batch_size=32
)


# =============================================================================
# INDONESIAN THESAURUS
# =============================================================================

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class IndonesianThesaurus:
    """
    Candidate generator using:

        victoriasovereigne/tesaurus

    Lookup strategy:

        1. Direct exact lookup
        2. Reverse synonym -> parent lemma lookup
        3. Stemmed lookup
        4. If no lookup succeeds -> word is not eligible

    Important:
        Stemming is used ONLY for dictionary lookup.

        The actual replacement text always comes from the thesaurus.
        We do NOT replace the original word with its stem.

    Example:

        Sentence:
            "aku mengetahui cara menggunakan abakus"

        "abakus"
            ↓
        reverse synonym lookup
            ↓
        parent = "dekak-dekak"
            ↓
        POS = noun
            ↓
        synonyms:
            cempoa
            sempoa
            swipoa

    For a morphological case:

        "menggunakan"
            ↓
        exact lookup fails
            ↓
        stemmed lookup
            ↓
        matching thesaurus entry
            ↓
        synonyms become candidates
    """

    def __init__(
        self,
        path: str,
    ):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Tesaurus file not found: {self.path}"
            )

        logger.info("START loading thesaurus JSON")
        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:
            self.data = json.load(file)

        logger.info(
            "Thesaurus JSON loaded: %d entries",
            len(self.data),
        )

        # ---------------------------------------------------------------------
        # Normalize dictionary keys once
        # ---------------------------------------------------------------------

        self.data = {
            str(key).strip().lower(): value
            for key, value in self.data.items()
        }

        # ---------------------------------------------------------------------
        # Sastrawi stemmer
        # ---------------------------------------------------------------------

        logger.info("START creating Sastrawi stemmer")

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        logger.info("Sastrawi stemmer created")

        # ---------------------------------------------------------------------
        # Reverse synonym index
        #
        # Instead of doing:
        #
        #     for main_word, candidate_entry in self.data.items():
        #
        # for EVERY lookup, build the reverse mapping once.
        #
        # Example:
        #
        #     "abakus" -> [
        #         ("dekak-dekak", entry)
        #     ]
        # ---------------------------------------------------------------------

        self.reverse_index = {}
        logger.info("START building reverse index")
        for parent_word, entry in self.data.items():

            if not isinstance(entry, dict):
                continue

            if not entry.get("tag"):
                continue

            for synonym in entry.get(
                "sinonim",
                [],
            ):

                synonym_key = (
                    str(synonym)
                    .strip()
                    .lower()
                )

                if not synonym_key:
                    continue

                self.reverse_index.setdefault(
                    synonym_key,
                    [],
                ).append(
                    {
                        "parent_word": parent_word,
                        "entry": entry,
                    }
                )

        logger.info(
            "Reverse index complete: %d terms",
            len(self.reverse_index),
        )
        # ---------------------------------------------------------------------
        # Stem index
        #
        # Maps stemmed dictionary words and synonyms to parent entries.
        #
        # Used ONLY after exact lookup fails.
        # ---------------------------------------------------------------------

        self.stem_index = {}
        logger.info("START building stem index")
        for parent_word, entry in self.data.items():

            if not isinstance(entry, dict):
                continue

            if not entry.get("tag"):
                continue

            # Parent lemma itself
            parent_stem = self._stem(
                parent_word
            )

            if parent_stem:
                self.stem_index.setdefault(
                    parent_stem,
                    [],
                ).append(
                    {
                        "parent_word": parent_word,
                        "entry": entry,
                        "source_word": parent_word,
                    }
                )

            # Synonyms
            for synonym in entry.get(
                "sinonim",
                [],
            ):

                synonym_key = (
                    str(synonym)
                    .strip()
                    .lower()
                )

                if not synonym_key:
                    continue

                synonym_stem = self._stem(
                    synonym_key
                )

                if synonym_stem:
                    self.stem_index.setdefault(
                        synonym_stem,
                        [],
                    ).append(
                        {
                            "parent_word": parent_word,
                            "entry": entry,
                            "source_word": synonym_key,
                        }
                    )

        logger.info(
            "Stem index complete: %d stems",
            len(self.stem_index),
        )
                
        logger.info(
            "Loaded Indonesian thesaurus: %d entries",
            len(self.data),
        )

        logger.info(
            "Built reverse synonym index: %d terms",
            len(self.reverse_index),
        )

        logger.info(
            "Built stem index: %d stems",
            len(self.stem_index),
        )

    # =========================================================================
    # STEMMING
    # =========================================================================

    @lru_cache(maxsize=100000)
    def _stem(
        self,
        word: str,
    ) -> str:
        """
        Stem a word using Sastrawi.

        Cached because the same words appear repeatedly.
        """

        word = str(word).strip().lower()

        if not word:
            return ""

        # Do not stem multi-word expressions as one replacement unit.
        #
        # They will be handled using exact/reverse lookup.
        if " " in word:
            return word

        return self.stemmer.stem(word)

    # =========================================================================
    # INTERNAL ENTRY SELECTION
    # =========================================================================

    @staticmethod
    def _select_entry(
        matches: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        """
        Select first valid parent entry.

        All indexed entries already have POS information.
        """

        for item in matches:

            entry = item.get("entry")

            if (
                isinstance(entry, dict)
                and entry.get("tag")
            ):
                return item

        return None

    # =========================================================================
    # ENTRY LOOKUP
    # =========================================================================

    @lru_cache(maxsize=100000)
    def get_entry_with_metadata(
        self,
        word: str,
    ) -> Optional[Dict[str, object]]:
        """
        Find thesaurus entry.

        Lookup order:

            1. Direct exact entry
            2. Reverse synonym -> parent
            3. Stemmed lookup

        Returns metadata describing how the entry was found.

        Example:

            {
                "entry": {...},
                "parent_word": "dekak-dekak",
                "lookup_source": "reverse_parent"
            }
        """

        key = str(
            word
        ).strip().lower()

        if not key:
            return None

        # =====================================================================
        # 1. DIRECT LOOKUP
        # =====================================================================

        entry = self.data.get(key)

        if (
            entry is not None
            and entry.get("tag")
        ):

            return {
                "entry": entry,
                "parent_word": key,
                "lookup_source": "direct",
            }

        # =====================================================================
        # 2. REVERSE SYNONYM LOOKUP
        #
        # Example:
        #
        #     abakus
        #       ↓
        #     dekak-dekak
        # =====================================================================

        reverse_matches = (
            self.reverse_index.get(
                key,
                [],
            )
        )

        selected = self._select_entry(
            reverse_matches
        )

        if selected is not None:

            return {
                "entry": selected["entry"],
                "parent_word": selected[
                    "parent_word"
                ],
                "lookup_source": "reverse_parent",
            }

        # =====================================================================
        # 3. STEMMED LOOKUP
        #
        # Only used when exact/reverse lookup failed.
        #
        # Example:
        #
        #     menggunakan
        #          ↓
        #     stemmer
        #          ↓
        #     menggunakan -> gunakan
        #          ↓
        #     stem index
        # =====================================================================

        stem = self._stem(key)

        if stem and stem != key:

            stem_matches = (
                self.stem_index.get(
                    stem,
                    [],
                )
            )

            selected = self._select_entry(
                stem_matches
            )

            if selected is not None:

                return {
                    "entry": selected["entry"],
                    "parent_word": selected[
                        "parent_word"
                    ],
                    "lookup_source": "stemmed",
                }

        return None

    # =========================================================================
    # BACKWARD-COMPATIBLE GET ENTRY
    # =========================================================================

    def get_entry(
        self,
        word: str,
    ) -> Optional[Dict]:
        """
        Backward-compatible method.

        Returns only the thesaurus entry.
        """

        result = self.get_entry_with_metadata(
            word
        )

        if result is None:
            return None

        return result["entry"]

    # =========================================================================
    # POS
    # =========================================================================

    def get_pos(
        self,
        word: str,
    ) -> Optional[str]:
        """
        Return POS tag.
        """

        result = self.get_entry_with_metadata(
            word
        )

        if result is None:
            return None

        return result["entry"].get(
            "tag"
        )

    # =========================================================================
    # CANDIDATES
    # =========================================================================

    def get_candidates_with_pos(
        self,
        word: str,
    ) -> List[Dict[str, object]]:
        """
        Generate thesaurus candidates.

        Lookup strategy:

            direct
                ↓
            reverse parent
                ↓
            stemmed

        Candidate filtering:

            1. Candidate != original
            2. Candidate != antonym
            3. Same POS when candidate POS can be established
            4. Unknown POS is allowed only when candidate is explicitly
               listed as a synonym of the selected parent entry

        Returns:

            [
                {
                    "candidate": str,
                    "candidate_pos": Optional[str],
                    "pos_verified": bool,
                    "pos_source": str,
                    "lookup_source": str,
                    "parent_word": str,
                }
            ]
        """

        # =====================================================================
        # Find original word's thesaurus entry
        # =====================================================================

        original_lookup = (
            self.get_entry_with_metadata(
                word
            )
        )

        if original_lookup is None:
            return []

        entry = original_lookup["entry"]

        original_word = (
            str(word)
            .strip()
            .lower()
        )

        original_pos = entry.get(
            "tag"
        )

        if not original_pos:
            return []

        original_parent = (
            original_lookup["parent_word"]
        )

        original_lookup_source = (
            original_lookup["lookup_source"]
        )

        # =====================================================================
        # Antonyms
        # =====================================================================

        antonyms = {
            str(x)
            .strip()
            .lower()
            for x in entry.get(
                "antonim",
                [],
            )
        }

        candidates = []

        # =====================================================================
        # Candidate generation
        # =====================================================================

        for raw_candidate in entry.get(
            "sinonim",
            [],
        ):

            candidate = (
                str(raw_candidate)
                .strip()
            )

            if not candidate:
                continue

            candidate_lower = (
                candidate.lower()
            )

            # -----------------------------------------------------------------
            # Exclude original
            # -----------------------------------------------------------------

            if candidate_lower == original_word:
                continue

            # -----------------------------------------------------------------
            # Exclude antonyms
            # -----------------------------------------------------------------

            if candidate_lower in antonyms:
                continue

            # -----------------------------------------------------------------
            # Try to determine candidate POS
            #
            # Same lookup strategy:
            #
            #     direct
            #     reverse parent
            #     stemmed
            # -----------------------------------------------------------------

            candidate_lookup = (
                self.get_entry_with_metadata(
                    candidate
                )
            )

            if candidate_lookup is not None:
                candidate_entry = candidate_lookup["entry"]
                candidate_pos = candidate_entry.get("tag")

                # Candidate POS is known
                if candidate_pos is not None:
                    # Reject candidate if POS differs
                    if candidate_pos != original_pos:
                        continue
                    candidates.append(
                        {
                            "candidate": candidate,
                            "candidate_pos": (
                                candidate_pos
                            ),
                            "pos_verified": True,
                            "pos_source": (
                                candidate_lookup[
                                    "lookup_source"
                                ]
                            ),
                            "lookup_source": (
                                candidate_lookup[
                                    "lookup_source"
                                ]
                            ),
                            "parent_word": (
                                candidate_lookup[
                                    "parent_word"
                                ]
                            ),
                            "original_lookup_source": (
                                original_lookup_source
                            ),
                            "original_parent_word": (
                                original_parent
                            ),
                        }
                    )

                    continue

            # =================================================================
            # Candidate has no independently recoverable POS
            #
            # BUT:
            #
            # It is explicitly listed under the original thesaurus entry.
            #
            # Therefore we can safely inherit the parent's POS.
            # =================================================================

            candidates.append(
                {
                    "candidate": candidate,

                    "candidate_pos": (
                        original_pos
                    ),

                    "pos_verified": True,

                    "pos_source": (
                        "inherited_from_original_parent"
                    ),

                    "lookup_source": (
                        "synonym_list"
                    ),

                    "parent_word": (
                        original_parent
                    ),

                    "original_lookup_source": (
                        original_lookup_source
                    ),

                    "original_parent_word": (
                        original_parent
                    ),
                }
            )

        # =====================================================================
        # Remove duplicate candidates
        # =====================================================================

        unique = {}

        for item in candidates:

            key = (
                item["candidate"]
                .lower()
            )

            if key not in unique:
                unique[key] = item

        return list(
            unique.values()
        )


# =============================================================================
# BASE PERTURBATION
# =============================================================================

class BasePerturbation(ABC):

    @abstractmethod
    def perturb(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def perturb_with_metadata(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> Dict[str, object]:
        raise NotImplementedError


# =============================================================================
# SEMANTIC WORD SUBSTITUTION
# =============================================================================

class SemanticWordSubstitution(
    BasePerturbation
):
    """
    Single perturbation method used for ALL levels.

    Intensity controls only the percentage of eligible words changed.

        Low    = 10%
        Medium = 20%
        High   = 30%
    """

    def __init__(
        self,
        thesaurus_path: str,
        random_seed: int = 42,
        sim_min: float = SIM_MIN,
        sim_max: float = SIM_MAX,
    ):
        self.thesaurus = IndonesianThesaurus(
            thesaurus_path
        )

        self.random_seed = random_seed

        self.rng = random.Random(
            random_seed
        )

        self.sim_min = sim_min
        self.sim_max = sim_max

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def perturb(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> str:

        result = self.perturb_with_metadata(
            text=text,
            intensity=intensity,
        )

        return result["perturbed_text"]

    # =========================================================================
    # MAIN
    # =========================================================================

    def perturb_with_metadata(
        self,
        text: str,
        intensity: Optional[float] = None,
    ) -> Dict[str, object]:

        if text is None:
            text = ""

        text = str(text)

        if intensity is None:
            raise ValueError(
                "Intensity must be explicitly supplied."
            )

        intensity = float(
            intensity
        )

        if not (
            0 < intensity <= 1
        ):
            raise ValueError(
                f"Invalid intensity: {intensity}"
            )

        if not text.strip():
            return self._infeasible_result(
                text,
                "EMPTY_TEXT",
                target_intensity=intensity,
            )

        word_spans = tokenize_word_spans(text)

        words = [
            item["word"]
            for item in word_spans
        ]

        if not words:
            return self._infeasible_result(
                text,
                "NO_WORDS",
                target_intensity=intensity,
            )

        # ---------------------------------------------------------------------
        # Determine eligible words
        # ---------------------------------------------------------------------
        lookup_stats = {
            "direct": 0,
            "reverse_parent": 0,
            "stemmed": 0,
            "failed": 0,
        }

        eligible = []


        for word_position, word_info in enumerate(word_spans):
            word = word_info["word"]
            word_start = word_info["start"]
            word_end = word_info["end"]

            lookup = (
                self.thesaurus.get_entry_with_metadata(
                    word
                )
            )

            if lookup is None:

                lookup_stats["failed"] += 1

                continue

            lookup_source = lookup[
                "lookup_source"
            ]

            lookup_stats[
                lookup_source
            ] += 1

            candidates = (
                self.thesaurus
                .get_candidates_with_pos(
                    word
                )
            )

            if not candidates:
                continue

            valid_candidates = []

            # Candidate strings
            candidate_strings = [
                item["candidate"]
                for item in candidates
            ]

            # =====================================================================
            # ORIGINAL EMBEDDING:
            # calculated ONCE
            #
            # CANDIDATE EMBEDDINGS:
            # calculated in batches
            # =====================================================================

            similarities = _checker.score_candidates(
                text=text,
                target_start=word_start,
                target_end=word_end,
                candidates=candidate_strings,
            )

            for candidate_info, similarity in zip(
                candidates,
                similarities,
            ):

                if (
                    self.sim_min
                    <= similarity
                    <= self.sim_max
                ):
                    valid_candidates.append(
                        {
                            "candidate": candidate_info[
                                "candidate"
                            ],

                            "similarity": similarity,

                            "candidate_pos": candidate_info[
                                "candidate_pos"
                            ],

                            "pos_verified": candidate_info[
                                "pos_verified"
                            ],

                            "pos_source": candidate_info[
                                "pos_source"
                            ],

                            "lookup_source": candidate_info[
                                "lookup_source"
                            ],

                            "parent_word": candidate_info[
                                "parent_word"
                            ],

                            "original_lookup_source": (
                                candidate_info[
                                    "original_lookup_source"
                                ]
                            ),

                            "original_parent_word": (
                                candidate_info[
                                    "original_parent_word"
                                ]
                            ),
                        }
                    )

            if valid_candidates:

                eligible.append(
                    {
                        "position": word_position,
                        "word": word,
                        "start": word_start,
                        "end": word_end,
                        "candidates": valid_candidates,
                    }
                )

        if not eligible:
            return self._infeasible_result(
                text,
                "NO_VALID_SEMANTIC_CANDIDATE",
                target_intensity=intensity,
                total_words=len(words),

                direct_lookup_words=(
                    lookup_stats["direct"]
                ),

                reverse_parent_lookup_words=(
                    lookup_stats["reverse_parent"]
                ),

                stemmed_lookup_words=(
                    lookup_stats["stemmed"]
                ),

                failed_lookup_words=(
                    lookup_stats["failed"]
                ),
            )

        # ---------------------------------------------------------------------
        # Target number of changed words
        #
        # Intensity is defined against eligible words because only eligible
        # words can actually be perturbed.
        # ---------------------------------------------------------------------

        target_count = int(
            np.floor(
                len(words) * intensity + 0.5
            )
        )

        target_count = min(
            target_count,
            len(eligible),
        )

        if target_count == 0:
            return self._infeasible_result(
                text,
                "TARGET_COUNT_ROUNDED_TO_ZERO",
                target_intensity=intensity,
                total_words=len(words),
                eligible_words=len(eligible),
                target_words=0,
                valid_replacements=0,
                direct_lookup_words=lookup_stats["direct"],
                reverse_parent_lookup_words=lookup_stats["reverse_parent"],
                stemmed_lookup_words=lookup_stats["stemmed"],
                failed_lookup_words=lookup_stats["failed"],
            )

        # ---------------------------------------------------------------------
        # Shuffle eligible words so the selected positions vary while remaining
        # reproducible through the random seed.
        # ---------------------------------------------------------------------

        shuffled = eligible.copy()

        self.rng.shuffle(
            shuffled
        )

        replacements = []

        # ---------------------------------------------------------------------
        # Select enough valid word replacements.
        #
        # If selected word cannot be replaced, move to another eligible word.
        # ---------------------------------------------------------------------

        for item in shuffled:

            if len(replacements) >= target_count:
                break

            candidates = item[
                "candidates"
            ]

            # highest contextual semantic similarity: 0.95
            best = max(
                candidates,
                key=lambda x: x["similarity"],
            )

            replacements.append(
                {
                    "position": item["position"],

                    "original": item["word"],

                    "replacement": best[
                        "candidate"
                    ],

                    "word_cosine_similarity": (
                        best["similarity"]
                    ),

                    "candidate_pos": (
                        best["candidate_pos"]
                    ),

                    "pos_verified": (
                        best["pos_verified"]
                    ),

                    "pos_source": (
                        best["pos_source"]
                    ),

                    "lookup_source": (
                        best["lookup_source"]
                    ),

                    "parent_word": (
                        best["parent_word"]
                    ),

                    "original_lookup_source": (
                        best["original_lookup_source"]
                    ),

                    "original_parent_word": (
                        best["original_parent_word"]
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Cannot satisfy requested intensity
        # ---------------------------------------------------------------------

        if len(replacements) < target_count:

            return self._infeasible_result(
                text,
                (
                    "INSUFFICIENT_VALID_REPLACEMENTS"
                ),
                target_intensity=intensity,
                total_words=len(words),
                eligible_words=len(eligible),
                target_words=target_count,
                valid_replacements=len(
                    replacements
                ),
            )

        # ---------------------------------------------------------------------
        # Apply substitutions
        # ---------------------------------------------------------------------

        perturbed_words = words.copy()

        for item in replacements:

            replacement = item[
                "replacement"
            ]

            original = item[
                "original"
            ]

            replacement = (
                self._preserve_case(
                    original,
                    replacement,
                )
            )

            perturbed_words[
                item["position"]
            ] = replacement

        # ---------------------------------------------------------------------
        # Reconstruct headline
        # ---------------------------------------------------------------------

        perturbed_text = self._replace_words(
            text,
            perturbed_words,
        )

        words_changed = sum(
            original.lower()
            != replacement.lower()
            for original, replacement
            in zip(
                words,
                perturbed_words,
            )
        )

        actual_ratio_all_words = (
            words_changed
            / len(words)
        )

        actual_ratio_eligible = (
            words_changed
            / len(eligible)
        )

        # ---------------------------------------------------------------------
        # Validate every replacement
        # ---------------------------------------------------------------------
        logger.info(
            "DEBUG replacements=%d target_count=%d eligible=%d",
            len(replacements),
            target_count,
            len(eligible),
        )
        
        replacement_similarities = [
            x["word_cosine_similarity"]
            for x in replacements
        ]

        similarity_valid = (
            bool(replacement_similarities)
            and all(
                self.sim_min <= score <= self.sim_max
                for score in replacement_similarities
            )
        )

        if replacement_similarities:
            similarity_mean = float(np.mean(replacement_similarities))
            similarity_min = float(np.min(replacement_similarities))
            similarity_max = float(np.max(replacement_similarities))
        else:
            similarity_mean = np.nan
            similarity_min = np.nan
            similarity_max = np.nan

        return {
            "original_text": text,
            "perturbed_text": perturbed_text,

            "perturbation_level": None,

            "perturbation_rule": (
                "semantic_similarity_based_word_substitution"
            ),

            "target_intensity": intensity,

            "total_words": len(words),

            "direct_lookup_words": lookup_stats["direct"],
            "reverse_parent_lookup_words": lookup_stats["reverse_parent"],
            "stemmed_lookup_words": lookup_stats["stemmed"],
            "failed_lookup_words": lookup_stats["failed"],

            "eligible_words": len(eligible),

            "target_words": target_count,

            "words_changed": words_changed,

            "actual_ratio_all_words": actual_ratio_all_words,

            "actual_ratio_eligible_words": actual_ratio_eligible,

            "is_same_as_original": text == perturbed_text,

            "similarity_in_range": similarity_valid,

            "perturbation_in_range": (
                words_changed == target_count
            ),

            "word_cosine_similarity_mean": similarity_mean,

            "word_cosine_similarity_min": similarity_min,

            "word_cosine_similarity_max": similarity_max,

            "replacements": replacements,
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _preserve_case(
        original: str,
        replacement: str,
    ) -> str:

        if original.isupper():
            return replacement.upper()

        if original.istitle():
            return replacement.capitalize()

        return replacement.lower()

    @staticmethod
    def _replace_words(
        original_text: str,
        replacement_words: List[str],
    ) -> str:
        """
        Replace word tokens in original text while preserving
        original punctuation and spacing as much as possible.
        """

        replacement_iter = iter(
            replacement_words
        )

        def replace_match(match):
            try:
                return next(
                    replacement_iter
                )
            except StopIteration:
                return match.group(0)

        return WORD_PATTERN.sub(
            replace_match,
            original_text,
        )

    @staticmethod
    def _infeasible_result(
        text: str,
        reason: str,
        target_intensity: float,
        total_words: int = 0,
        eligible_words: int = 0,
        target_words: int = 0,
        valid_replacements: int = 0,
        direct_lookup_words: int = 0,
        reverse_parent_lookup_words: int = 0,
        stemmed_lookup_words: int = 0,
        failed_lookup_words: int = 0,
    ) -> Dict[str, object]:

        return {
            "original_text": text,
            "perturbed_text": text,

            "perturbation_level": None,

            "perturbation_rule": (
                f"INFEASIBLE:{reason}"
            ),

            "target_intensity": (
                target_intensity
            ),

            "total_words": total_words,

            "eligible_words": (
                eligible_words
            ),

            "target_words": (
                target_words
            ),

            "words_changed": (
                valid_replacements
            ),

            "direct_lookup_words": direct_lookup_words,

            "reverse_parent_lookup_words": (
                reverse_parent_lookup_words
            ),

            "stemmed_lookup_words": (
                stemmed_lookup_words
            ),

            "failed_lookup_words": (
                failed_lookup_words
            ),

            "actual_ratio_all_words": 0.0,

            "actual_ratio_eligible_words": 0.0,

            "is_same_as_original": True,

            "similarity_in_range": False,

            "perturbation_in_range": False,

            "word_cosine_similarity_mean": np.nan,

            "word_cosine_similarity_min": np.nan,

            "word_cosine_similarity_max": np.nan,

            "replacements": [],
        }


# =============================================================================
# PERTURBATION ENGINE
# =============================================================================

class PerturbationEngine:
    """
    Main perturbation engine.

    Same method:
        semantic word substitution

    Different intensity:
        low    = 10%
        medium = 20%
        high   = 30%
    """

    def __init__(
        self,
        thesaurus_path: str,
        random_seed: int = 42,
        sim_min: float = SIM_MIN,
        sim_max: float = SIM_MAX,
    ):

        self.random_seed = random_seed

        self.sim_min = sim_min
        self.sim_max = sim_max

        self.perturbation = (
            SemanticWordSubstitution(
                thesaurus_path=thesaurus_path,
                random_seed=random_seed,
                sim_min=sim_min,
                sim_max=sim_max,
            )
        )

        logger.info(
            "PerturbationEngine initialized"
        )

        logger.info(
            "Method: semantic similarity-based "
            "word substitution"
        )

        logger.info(
            "Low=%.0f%%, Medium=%.0f%%, High=%.0f%%",
            PERTURBATION_INTENSITIES["low"] * 100,
            PERTURBATION_INTENSITIES["medium"] * 100,
            PERTURBATION_INTENSITIES["high"] * 100,
        )

        logger.info(
            "Word cosine similarity range: %.2f-%.2f",
            sim_min,
            sim_max,
        )

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def apply_perturbation(
        self,
        text: str,
        level: str,
        intensity: Optional[float] = None,
    ) -> str:

        result = self.apply_perturbation_with_metadata(
            text=text,
            level=level,
            intensity=intensity,
        )

        return result["perturbed_text"]

    def apply_perturbation_with_metadata(
        self,
        text: str,
        level: str,
        intensity: Optional[float] = None,
    ) -> Dict[str, object]:

        level = level.lower().strip()

        if level not in PERTURBATION_INTENSITIES:
            raise ValueError(
                f"Invalid perturbation level: {level}. "
                "Choose low, medium, or high."
            )

        if intensity is None:
            intensity = (
                PERTURBATION_INTENSITIES[
                    level
                ]
            )

        result = (
            self.perturbation
            .perturb_with_metadata(
                text=text,
                intensity=intensity,
            )
        )

        result[
            "perturbation_level"
        ] = level

        return result

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        level: str = "low",
        intensity: Optional[float] = None,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        level = level.lower().strip()

        if level not in PERTURBATION_INTENSITIES:
            raise ValueError(
                f"Invalid level: {level}"
            )

        logger.info(
            "Applying %s perturbation to %d rows",
            level.upper(),
            len(df),
        )

        results = []

        for idx, value in tqdm(
            df[text_column].items(),
            total=len(df),
            desc=f"Perturbing {level.upper()}",
        ):

            text = (
                ""
                if pd.isna(value)
                else str(value)
            )

            result = (
                self.apply_perturbation_with_metadata(
                    text=text,
                    level=level,
                    intensity=intensity,
                )
            )

            results.append(
                {
                    "original_index": idx,
                    **result,
                }
            )

        result_df = pd.DataFrame(results)

        perturbed_df = df.copy()

        # Keep original dataframe columns intact
        perturbed_df[text_column] = (
            result_df["perturbed_text"].values
        )

        # Add audit metadata
        metadata_columns = [
            "original_text",
            "perturbed_text",
            "perturbation_level",
            "perturbation_rule",
            "target_intensity",

            "direct_lookup_words",
            "reverse_parent_lookup_words",
            "stemmed_lookup_words",
            "failed_lookup_words",

            "total_words",
            "eligible_words",
            "target_words",
            "words_changed",

            "actual_ratio_all_words",
            "actual_ratio_eligible_words",

            "word_cosine_similarity_mean",
            "word_cosine_similarity_min",
            "word_cosine_similarity_max",

            "is_same_as_original",
            "similarity_in_range",
            "perturbation_in_range",

            "replacements",
        ]

        for column in metadata_columns:

            if column in result_df.columns:

                perturbed_df[column] = (
                    result_df[column].values
                )

        # -------------------------------------------------------------------------
        # SUCCESS COUNT
        # -------------------------------------------------------------------------

        successful = (
            result_df["perturbation_in_range"]
            & ~result_df["is_same_as_original"]
        )

        successful_count = int(
            successful.sum()
        )

        total_count = len(result_df)

        success_percentage = (
            successful_count / total_count * 100
            if total_count
            else 0.0
        )

        # -------------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------------

        logger.info(
            "%s successful perturbations: %d/%d (%.2f%%)",
            level.upper(),
            successful_count,
            total_count,
            success_percentage,
        )

        print(
            f"\n{level.upper()} SUCCESS: "
            f"{successful_count}/{total_count} "
            f"({success_percentage:.2f}%)"
        )

        # -------------------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------------------

        logger.info(
            "%s target intensity: %.2f%%",
            level.upper(),
            (
                result_df["target_intensity"].iloc[0] * 100
                if len(result_df)
                else 0.0
            ),
        )

        logger.info(
            "%s actual ratio over all words: %.4f",
            level.upper(),
            result_df[
                "actual_ratio_all_words"
            ].mean(),
        )

        logger.info(
            "%s actual ratio over eligible words: %.4f",
            level.upper(),
            result_df[
                "actual_ratio_eligible_words"
            ].mean(),
        )

        logger.info(
            "%s mean word cosine similarity: %.4f",
            level.upper(),
            result_df[
                "word_cosine_similarity_mean"
            ].mean(),
        )

        # -------------------------------------------------------------------------
        # Debug samples
        # -------------------------------------------------------------------------

        dbg_perturbation_samples(
            level=level,
            domain="dataframe",
            originals=result_df[
                "original_text"
            ].tolist(),
            perturbed=result_df[
                "perturbed_text"
            ].tolist(),
        )

        dbg_perturbation_stats(
            level=level,
            domain="dataframe",
            n_texts=len(perturbed_df),
            char_change_mean=0.0,
            word_change_mean=float(
                result_df[
                    "actual_ratio_all_words"
                ].mean()
            ),
        )

        # -------------------------------------------------------------------------
        # SAVE
        # -------------------------------------------------------------------------

        if output_csv:

            output_dir = os.path.dirname(
                os.path.abspath(output_csv)
            )

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            perturbed_df.to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
            )

            logger.info(
                "Saved perturbation dataset: %s",
                output_csv,
            )

            print(
                f"Saved → {output_csv}"
            )

        return perturbed_df

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_perturbation_stats(
        self,
        original: str,
        perturbed: str,
    ) -> Dict[str, float]:

        original_words = tokenize_words(
            original
        )

        perturbed_words = tokenize_words(
            perturbed
        )

        if len(original_words) != len(
            perturbed_words
        ):
            raise ValueError(
                "Word substitution should preserve "
                "the number of word tokens."
            )

        words_changed = sum(
            a.lower() != b.lower()
            for a, b in zip(
                original_words,
                perturbed_words,
            )
        )

        ratio = (
            words_changed
            / len(original_words)
            if original_words
            else 0.0
        )

        return {
            "original_word_count": len(
                original_words
            ),
            "perturbed_word_count": len(
                perturbed_words
            ),
            "words_changed": words_changed,
            "word_change_ratio": ratio,
            "is_same_as_original": (
                original == perturbed
            ),
        } 