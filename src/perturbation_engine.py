"""
Perturbation Engine Module for IndoBERT Clickbait Detection System

This module implements three levels of text perturbations for robustness testing:
- Low-level: Character-level typos (5-10% intensity)
- Medium-level: Informal language injection (15-25% intensity)
- High-level: Synonym replacement and paraphrasing (40-60% intensity)

Vocabulary is loaded from JSON files in ``dataset/perturbation_vocab/``.
If a file is missing the module silently falls back to the built-in defaults,
so the pipeline never breaks on a fresh clone without the vocab files.
"""

import json
import os
import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from debug_logger import (
    dbg_perturbation_samples,
    dbg_perturbation_stats,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── vocabulary loader ──────────────────────────────────────────────────────────

def _find_vocab_dir() -> Optional[str]:
    """
    Locate the ``dataset/perturbation_vocab/`` directory relative to this
    source file.  Works whether the script is run from ``src/``, from the
    repo root, or from any other working directory.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # Walk up from src/ to find the dataset/ sibling
    for base in (src_dir, os.path.dirname(src_dir)):
        candidate = os.path.join(base, "dataset", "perturbation_vocab")
        if os.path.isdir(candidate):
            return candidate
    return None


def _load_vocab_json(filename: str, default: any) -> any:
    """
    Load a JSON vocabulary file from the perturbation_vocab directory.

    If the directory or file does not exist, logs a warning and returns
    *default* so the pipeline continues with the built-in fallback.

    The special ``_comment`` key and section-header keys (those whose value
    is an empty dict ``{}``) are stripped automatically.
    """
    vocab_dir = _find_vocab_dir()
    if vocab_dir is None:
        logger.warning(
            "perturbation_vocab/ directory not found — using built-in defaults."
        )
        return default

    filepath = os.path.join(vocab_dir, filename)
    if not os.path.exists(filepath):
        logger.warning(
            f"Vocab file not found: {filepath} — using built-in defaults."
        )
        return default

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Strip comment / section-header keys so callers get clean data
        if isinstance(data, dict):
            data = {
                k: v for k, v in data.items()
                if not k.startswith("_") and not k.startswith("──") and v != {}
            }
        logger.info(f"Loaded vocab from {filepath}")
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to load {filepath}: {exc} — using built-in defaults.")
        return default


class PerturbationEngine:
    """
    Main engine for applying various levels of perturbations to Indonesian text.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the perturbation engine.

        Each sub-class receives a *different* derived seed so that they use
        independent random streams. This avoids the earlier bug where each
        class reset the global random.seed(), meaning Low and Medium
        perturbations were not truly reproducible because their seeds were
        immediately overwritten by the next class's __init__.

        Args:
            random_seed: Base random seed for reproducibility
        """
        self.random_seed = random_seed
        # NOTE: We do NOT call np.random.seed() here to avoid mutating global
        # NumPy state as a constructor side-effect. Each sub-engine uses its
        # own private random.Random instance for full isolation.

        # Each class gets a distinct derived seed so their streams are
        # independent and do not interfere with each other.
        self.low_level    = LowLevelPerturbation(random_seed)
        self.medium_level = MediumLevelPerturbation(random_seed + 1)
        self.high_level   = HighLevelPerturbation(random_seed + 2)

        logger.info(f"PerturbationEngine initialized with seed={random_seed}")
    
    def apply_perturbation(
        self,
        text: str,
        level: str,
        intensity: Optional[float] = None
    ) -> str:
        """
        Apply perturbation to text based on specified level.
        
        Args:
            text: Input text to perturb
            level: Perturbation level ('low', 'medium', 'high')
            intensity: Optional custom intensity (overrides default)
            
        Returns:
            Perturbed text
            
        Raises:
            ValueError: If level is invalid
        """
        if level.lower() == 'low':
            return self.low_level.perturb(text, intensity)
        elif level.lower() == 'medium':
            return self.medium_level.perturb(text, intensity)
        elif level.lower() == 'high':
            return self.high_level.perturb(text, intensity)
        else:
            raise ValueError(f"Invalid perturbation level: {level}. Must be 'low', 'medium', or 'high'")
    
    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        level: str = 'low',
        intensity: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Apply perturbations to all texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            level: Perturbation level
            intensity: Optional custom intensity
            
        Returns:
            DataFrame with perturbed texts
        """
        logger.info(f"Applying {level}-level perturbations to {len(df)} texts")
        
        original_texts = df[text_column].tolist()
        perturbed_df = df.copy()
        perturbed_df[text_column] = perturbed_df[text_column].apply(
            lambda x: self.apply_perturbation(x, level, intensity)
        )
        perturbed_texts = perturbed_df[text_column].tolist()

        # ── debug: show sample before/after pairs ─────────────────────────
        dbg_perturbation_samples(
            level=level,
            domain="dataframe",
            originals=original_texts,
            perturbed=perturbed_texts,
        )
        # ── debug: aggregate change statistics ────────────────────────────
        char_changes = [
            sum(1 for a, b in zip(o, p) if a != b) / max(len(o), 1)
            for o, p in zip(original_texts, perturbed_texts)
        ]
        word_changes = [
            len(set(o.split()).symmetric_difference(set(p.split()))) / max(len(o.split()), 1)
            for o, p in zip(original_texts, perturbed_texts)
        ]
        dbg_perturbation_stats(
            level=level,
            domain="dataframe",
            n_texts=len(perturbed_df),
            char_change_mean=float(np.mean(char_changes)) if char_changes else 0.0,
            word_change_mean=float(np.mean(word_changes)) if word_changes else 0.0,
        )
        
        logger.info("Perturbation complete")
        return perturbed_df
    
    def get_perturbation_stats(
        self,
        original: str,
        perturbed: str
    ) -> Dict[str, float]:
        """
        Calculate statistics about the perturbation applied.
        
        Args:
            original: Original text
            perturbed: Perturbed text
            
        Returns:
            Dictionary with perturbation statistics
        """
        original_chars = len(original)
        perturbed_chars = len(perturbed)
        
        original_words = len(original.split())
        perturbed_words = len(perturbed.split())
        
        # Calculate character-level changes
        char_changes = sum(1 for a, b in zip(original, perturbed) if a != b)
        char_change_ratio = char_changes / original_chars if original_chars > 0 else 0
        
        # Calculate word-level changes
        original_word_set = set(original.split())
        perturbed_word_set = set(perturbed.split())
        word_changes = len(original_word_set.symmetric_difference(perturbed_word_set))
        word_change_ratio = word_changes / original_words if original_words > 0 else 0
        
        return {
            'char_change_ratio': char_change_ratio,
            'word_change_ratio': word_change_ratio,
            'length_change': perturbed_chars - original_chars,
            'original_length': original_chars,
            'perturbed_length': perturbed_chars
        }


class BasePerturbation(ABC):
    """
    Abstract base class that defines the common interface for all perturbation
    intensity levels. Subclasses must implement :meth:`perturb`.

    This makes the Strategy pattern explicit: :class:`PerturbationEngine` holds
    three concrete strategies (low / medium / high) that are interchangeable
    at the ``perturb(text, intensity)`` boundary.
    """

    @abstractmethod
    def perturb(self, text: str, intensity: Optional[float] = None) -> str:
        """
        Apply perturbations to *text*.

        Args:
            text:      Input string to perturb
            intensity: Fraction of characters/words to affect.
                       Uses a level-specific default when ``None``.

        Returns:
            Perturbed text string
        """


class LowLevelPerturbation(BasePerturbation):
    """
    Low-level perturbations: Character-level typos.
    Intensity: 5-10% of characters affected.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize low-level perturbation handler.

        Uses a private random.Random instance so this class's state is
        fully isolated from the global random module and from other classes.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

        # Indonesian keyboard layout for realistic typos
        self.keyboard_neighbors = {
            'a': ['s', 'q', 'w', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 'r', 'd', 's'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l', 'k'],
            'p': ['o', 'l'],
            'q': ['w', 'a'],
            'r': ['e', 't', 'f', 'd'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'y', 'g', 'f'],
            'u': ['y', 'i', 'j', 'h'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'e', 's', 'a'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'u', 'h', 'g'],
            'z': ['a', 's', 'x']
        }
        
        logger.info("LowLevelPerturbation initialized")
    
    def perturb(self, text: str, intensity: Optional[float] = None) -> str:
        """
        Apply low-level perturbations to text.
        
        Args:
            text: Input text
            intensity: Proportion of characters to affect (default: 0.05-0.10)
            
        Returns:
            Perturbed text
        """
        if not text or len(text) == 0:
            return text
        
        # Default intensity: 5-10%
        if intensity is None:
            intensity = self._rng.uniform(0.05, 0.10)

        chars = list(text)
        num_perturbations = max(1, int(len(chars) * intensity))

        # Get indices of alphabetic characters only
        alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]

        if not alpha_indices:
            return text

        # Randomly select characters to perturb
        num_perturbations = min(num_perturbations, len(alpha_indices))
        perturb_indices = self._rng.sample(alpha_indices, num_perturbations)

        # 'swap' operates at word level: collect swaps, apply after char loop
        swap_positions: List[int] = []
        for idx in perturb_indices:
            result = self._apply_typo(chars, idx)
            if result is None:
                # 'swap' requested — defer to word-level swap below
                swap_positions.append(idx)
            else:
                chars[idx] = result

        # Word-level adjacent-character swap for the deferred positions
        for idx in swap_positions:
            if idx + 1 < len(chars):
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        return ''.join(chars)

    def _apply_typo(self, chars: List[str], idx: int):
        """
        Apply a random typo at position *idx* of the character list.

        Returns the replacement string, or None to signal a word-level swap
        (caller handles the swap to avoid indexing complexity inside here).

        Args:
            chars: Full character list of the text being perturbed
            idx:   Index of the character to modify

        Returns:
            str  — replacement character(s), or '' for deletion
            None — caller should perform adjacent-character swap at idx
        """
        char = chars[idx]
        char_lower = char.lower()
        typo_type = self._rng.choice(['substitute', 'delete', 'insert', 'swap'])

        if typo_type == 'substitute' and char_lower in self.keyboard_neighbors:
            new_char = self._rng.choice(self.keyboard_neighbors[char_lower])
            return new_char.upper() if char.isupper() else new_char

        elif typo_type == 'delete':
            return ''

        elif typo_type == 'insert' and char_lower in self.keyboard_neighbors:
            insert_char = self._rng.choice(self.keyboard_neighbors[char_lower])
            return char + insert_char

        elif typo_type == 'swap':
            # Signal the caller to do a word-level adjacent swap
            return None

        return char


class MediumLevelPerturbation(BasePerturbation):
    """
    Medium-level perturbations: Informal language injection.
    Intensity: 15-25% of words affected.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize medium-level perturbation handler.

        Uses a private random.Random instance so this class's state is
        fully isolated from the global random module and from other classes.

        Vocabulary is loaded from ``dataset/perturbation_vocab/`` JSON files
        at init time.  If a file cannot be found the built-in minimal defaults
        are used so the pipeline never fails on a fresh environment.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

        # ── built-in minimal defaults (used when JSON file is absent) ─────
        _default_formal_to_informal = {
            'tidak': ['gak', 'nggak', 'ga', 'ngga'],
            'apa': ['apaan', 'apa sih'],
            'saya': ['gue', 'gw', 'aku'],
            'kamu': ['lu', 'loe', 'elu'],
            'sudah': ['udah', 'udh'],
            'belum': ['blm', 'blom'],
            'dengan': ['sama', 'ama'],
            'untuk': ['buat', 'utk'],
            'yang': ['yg'],
            'ini': ['nih'],
            'itu': ['tuh'],
            'bagaimana': ['gimana', 'gmn'],
            'kenapa': ['knp', 'napa'],
            'kapan': ['kapan sih'],
            'dimana': ['dmn', 'mana'],
            'siapa': ['siapa sih'],
            'akan': ['bakal'],
            'hanya': ['cuma', 'cm'],
            'juga': ['jg', 'juga sih'],
            'sangat': ['banget', 'bgt'],
            'sekali': ['banget'],
            'bisa': ['bs', 'bisa kok'],
            'mau': ['mw', 'pengen'],
            'ingin': ['pengen', 'pgn'],
            'seperti': ['kayak', 'kyk'],
            'tetapi': ['tapi', 'tp'],
            'atau': ['ato'],
            'karena': ['soalnya', 'krn'],
            'memang': ['emang', 'emg'],
            'sekarang': ['skrg', 'sekarang nih'],
            'nanti': ['ntar'],
            'tahu': ['tau'],
            'banyak': ['byk', 'banyak banget'],
        }
        _default_slang_particles = [
            'sih', 'nih', 'dong', 'deh', 'lah', 'kok', 'kan'
        ]
        _default_abbreviations = {
            'dan': 'n',
            'di': 'd',
            'ke': 'k',
            'dari': 'dr',
            'sama': 'sm',
        }

        # ── load from JSON, fall back to defaults if absent ───────────────
        loaded_f2i = _load_vocab_json(
            "formal_to_informal.json", _default_formal_to_informal
        )
        # JSON values must be lists; guard against stray string values
        self.formal_to_informal: Dict[str, List[str]] = {
            k: (v if isinstance(v, list) else [v])
            for k, v in loaded_f2i.items()
        }

        particles_data = _load_vocab_json(
            "slang_particles.json",
            {"particles": _default_slang_particles},
        )
        raw_particles = particles_data.get("particles", _default_slang_particles)
        self.slang_additions: List[str] = (
            raw_particles if isinstance(raw_particles, list) else _default_slang_particles
        )

        self.abbreviations: Dict[str, str] = _load_vocab_json(
            "abbreviations.json", _default_abbreviations
        )

        logger.info(
            f"MediumLevelPerturbation initialized — "
            f"{len(self.formal_to_informal)} formal→informal entries, "
            f"{len(self.slang_additions)} particles, "
            f"{len(self.abbreviations)} abbreviations"
        )
    
    def perturb(self, text: str, intensity: Optional[float] = None) -> str:
        """
        Apply medium-level perturbations to text.
        
        Args:
            text: Input text
            intensity: Proportion of words to affect (default: 0.15-0.25)
            
        Returns:
            Perturbed text
        """
        if not text or len(text) == 0:
            return text
        
        # Default intensity: 15-25%
        if intensity is None:
            intensity = self._rng.uniform(0.15, 0.25)

        words = text.split()
        num_perturbations = max(1, int(len(words) * intensity))

        # Prefer content words (length > 2, not purely numeric/symbolic).
        # This avoids wasting perturbation budget on tokens like "1", "Rp",
        # "&" which cannot be meaningfully informalized.
        content_indices = [
            i for i, w in enumerate(words)
            if len(re.sub(r'[^\w]', '', w)) > 2
            and not re.sub(r'[^\w]', '', w).isdigit()
        ]
        # Fall back to all positions if not enough content words available
        candidate_pool = content_indices if len(content_indices) >= num_perturbations else list(range(len(words)))
        perturb_indices = self._rng.sample(candidate_pool, min(num_perturbations, len(candidate_pool)))

        for idx in perturb_indices:
            words[idx] = self._informalize_word(words[idx])

        return ' '.join(words)
    
    def _informalize_word(self, word: str) -> str:
        """
        Convert a word to informal Indonesian.

        Priority order:
          1. Exact match in formal_to_informal mapping (highest confidence)
          2. Exact match in abbreviations dict (50% chance to abbreviate)
          3. Append a slang particle to the word (70% chance — raised from
             the previous 30% so that out-of-vocabulary news nouns/verbs are
             reliably perturbed rather than silently passed through unchanged)
          4. Drop one interior character as a last-resort OOV fallback
             (50% chance, only for words longer than 4 characters)

        Args:
            word: Word to informalize

        Returns:
            Informalized word (guaranteed to differ from input on paths 1–4
            when the word has enough characters)
        """
        word_lower = word.lower()

        # Remove punctuation for matching
        word_clean = re.sub(r'[^\w\s]', '', word_lower)

        # 1. Formal → informal mapping
        if word_clean in self.formal_to_informal:
            informal = self._rng.choice(self.formal_to_informal[word_clean])
            # Preserve capitalisation: if the original word started with an
            # uppercase letter, capitalise the replacement too.
            # NOTE: do NOT use word.replace(word_clean, informal) here —
            # that fails when the word has a leading capital (e.g. "Pemerintah"
            # does not contain the lowercase substring "pemerintah").
            # Preserve any trailing punctuation (e.g. comma, period).
            trailing = ''.join(c for c in word if not c.isalnum() and c not in ("'", "\u2019"))
            if word[0].isupper():
                informal = informal.capitalize()
            return informal + trailing

        # 2. Abbreviation (50% chance)
        if word_clean in self.abbreviations and self._rng.random() < 0.5:
            return self.abbreviations[word_clean]

        # 3. Slang particle appended — raised to 70% so OOV news words
        #    (names, tech terms, domain nouns) reliably get perturbed
        if len(word_clean) > 3 and self._rng.random() < 0.7:
            particle = self._rng.choice(self.slang_additions)
            return f"{word} {particle}"

        # 4. Guaranteed OOV fallback: drop one interior character.
        #    No probability gate — if all three paths above missed, this
        #    ensures the word is *always* changed so medium perturbation
        #    never silently passes through a content word unchanged.
        #    Threshold ≥ 3 (was > 3) so tokens like "17T" (len=3) are
        #    also covered — randint(1, len-2) requires len >= 3.
        if len(word_clean) >= 3 and len(word) >= 3:
            drop_idx = self._rng.randint(1, len(word) - 2)
            return word[:drop_idx] + word[drop_idx + 1:]

        return word


class HighLevelPerturbation(BasePerturbation):
    """
    High-level perturbations: Synonym replacement and paraphrasing.
    Intensity: 40-60% of content altered.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize high-level perturbation handler.

        Uses a private random.Random instance so this class's state is
        fully isolated from the global random module and from other classes.

        Vocabulary is loaded from ``dataset/perturbation_vocab/synonyms.json``
        at init time.  If the file is absent the built-in adjective-only
        defaults are used so the pipeline never fails.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

        # ── built-in minimal defaults (adjectives only, used when JSON absent)
        _default_synonyms = {
            'besar': ['raksasa', 'jumbo', 'gede', 'luas'],
            'kecil': ['mungil', 'mini', 'cilik'],
            'bagus': ['baik', 'oke', 'mantap', 'keren'],
            'buruk': ['jelek', 'tidak baik', 'payah'],
            'cepat': ['kilat', 'gesit', 'laju'],
            'lambat': ['pelan', 'lelet'],
            'tinggi': ['jangkung', 'menjulang'],
            'rendah': ['pendek'],
            'penting': ['krusial', 'vital', 'esensial'],
            'mudah': ['gampang', 'simpel'],
            'sulit': ['susah', 'rumit', 'kompleks'],
            'baru': ['anyar', 'fresh'],
            'lama': ['lawas', 'usang'],
            'menarik': ['seru', 'asyik', 'keren'],
            'membosankan': ['ngebosenin', 'monoton'],
            'senang': ['gembira', 'bahagia', 'happy'],
            'sedih': ['duka', 'galau'],
            'marah': ['kesal', 'jengkel', 'dongkol'],
            'takut': ['ngeri', 'seram'],
            'berani': ['pemberani', 'gagah'],
            'pintar': ['cerdas', 'pandai', 'jenius'],
            'bodoh': ['dungu', 'tolol'],
            'cantik': ['indah', 'ayu', 'elok'],
            'jelek': ['buruk rupa'],
            'kaya': ['tajir', 'berada', 'mampu'],
            'miskin': ['papa', 'melarat'],
            'ramai': ['rame', 'hiruk pikuk'],
            'sepi': ['sunyi', 'lengang'],
            'panas': ['gerah', 'hangat'],
            'dingin': ['sejuk', 'adem'],
            'terang': ['cerah', 'jelas'],
            'gelap': ['remang', 'kelam'],
            'keras': ['kuat', 'solid'],
            'lembut': ['halus', 'soft'],
            'mahal': ['pricey', 'selangit'],
            'murah': ['terjangkau', 'ekonomis'],
        }

        # ── load from JSON, fall back to defaults if absent ───────────────
        loaded_synonyms = _load_vocab_json("synonyms.json", _default_synonyms)
        # Ensure every value is a non-empty list and strip the key itself
        # from the options list so we always substitute a *different* word.
        self.synonyms: Dict[str, List[str]] = {}
        for k, v in loaded_synonyms.items():
            options = v if isinstance(v, list) else [v]
            options = [s for s in options if s != k]   # exclude identity
            if options:
                self.synonyms[k] = options

        # Sentence structure variations
        self.structure_patterns = [
            'passive_to_active',
            'active_to_passive',
            'reorder_clauses'
        ]

        logger.info(
            f"HighLevelPerturbation initialized — "
            f"{len(self.synonyms)} synonym entries loaded"
        )
    
    def perturb(self, text: str, intensity: Optional[float] = None) -> str:
        """
        Apply high-level perturbations to text.
        
        Args:
            text: Input text
            intensity: Proportion of content to alter (default: 0.40-0.60)
            
        Returns:
            Perturbed text
        """
        if not text or len(text) == 0:
            return text
        
        # Default intensity: 40-60%
        if intensity is None:
            intensity = self._rng.uniform(0.40, 0.60)

        # Apply synonym replacement
        text = self._replace_synonyms(text, intensity)

        # Apply sentence structure modification (with lower probability)
        if self._rng.random() < 0.3:
            text = self._modify_structure(text)
        
        return text
    
    def _replace_synonyms(self, text: str, intensity: float) -> str:
        """
        Replace words with their synonyms.
        
        Args:
            text: Input text
            intensity: Proportion of words to replace
            
        Returns:
            Text with synonyms replaced
        """
        words = text.split()
        num_replacements = max(1, int(len(words) * intensity))
        
        # Find replaceable words
        replaceable_indices = []
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w\s]', '', word.lower())
            if word_clean in self.synonyms:
                replaceable_indices.append(i)
        
        if not replaceable_indices:
            return text

        # Randomly select words to replace
        num_replacements = min(num_replacements, len(replaceable_indices))
        replace_indices = self._rng.sample(replaceable_indices, num_replacements)

        for idx in replace_indices:
            word = words[idx]
            word_clean = re.sub(r'[^\w\s]', '', word.lower())

            if word_clean in self.synonyms:
                # Get synonym (excluding the original word)
                synonym_options = [s for s in self.synonyms[word_clean] if s != word_clean]
                if synonym_options:
                    synonym = self._rng.choice(synonym_options)
                    
                    # Preserve capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    
                    # Preserve punctuation
                    punctuation = ''.join(c for c in word if not c.isalnum())
                    words[idx] = synonym + punctuation
        
        return ' '.join(words)
    
    def _modify_structure(self, text: str) -> str:
        """
        Modify sentence structure while preserving meaning.
        
        Args:
            text: Input text
            
        Returns:
            Text with modified structure
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return text

        # Randomly reorder some clauses or sentences
        if self._rng.random() < 0.5 and len(sentences) >= 2:
            # Swap two adjacent sentences
            idx = self._rng.randint(0, len(sentences) - 2)
            sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
        
        # Reconstruct text
        return '. '.join(sentences) + '.'


class PerturbationBatch:
    """
    Utility class for batch perturbation operations.
    """
    
    def __init__(self, engine: PerturbationEngine):
        """
        Initialize batch perturbation handler.
        
        Args:
            engine: PerturbationEngine instance
        """
        self.engine = engine
        logger.info("PerturbationBatch initialized")
    
    def create_perturbation_variants(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        levels: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create multiple perturbation variants of a dataset.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            levels: List of perturbation levels (default: all three)
            
        Returns:
            Dictionary mapping level names to perturbed DataFrames
        """
        if levels is None:
            levels = ['low', 'medium', 'high']
        
        variants = {}
        
        for level in levels:
            logger.info(f"Creating {level}-level perturbation variant")
            variants[level] = self.engine.apply_to_dataframe(df, text_column, level)
        
        return variants
    
    def analyze_perturbation_impact(
        self,
        original_df: pd.DataFrame,
        perturbed_dfs: Dict[str, pd.DataFrame],
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Analyze the impact of perturbations across levels.
        
        Args:
            original_df: Original DataFrame
            perturbed_dfs: Dictionary of perturbed DataFrames by level
            text_column: Name of text column
            
        Returns:
            DataFrame with perturbation statistics
        """
        stats_list = []
        
        for level, perturbed_df in perturbed_dfs.items():
            level_stats = []
            
            for idx in range(len(original_df)):
                original_text = original_df.iloc[idx][text_column]
                perturbed_text = perturbed_df.iloc[idx][text_column]
                
                stats = self.engine.get_perturbation_stats(original_text, perturbed_text)
                stats['level'] = level
                stats['index'] = idx
                level_stats.append(stats)
            
            stats_df = pd.DataFrame(level_stats)
            stats_list.append(stats_df)
        
        combined_stats = pd.concat(stats_list, ignore_index=True)
        
        # Log summary statistics
        logger.info("\nPerturbation Impact Summary:")
        for level in perturbed_dfs.keys():
            level_data = combined_stats[combined_stats['level'] == level]
            logger.info(f"\n{level.upper()} Level:")
            logger.info(f"  Avg char change ratio: {level_data['char_change_ratio'].mean():.2%}")
            logger.info(f"  Avg word change ratio: {level_data['word_change_ratio'].mean():.2%}")
            logger.info(f"  Avg length change: {level_data['length_change'].mean():.2f}")
        
        return combined_stats
