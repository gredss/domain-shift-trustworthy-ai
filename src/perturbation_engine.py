"""
Perturbation Engine Module for IndoBERT Clickbait Detection System

This module implements three levels of text perturbations for robustness testing:
- Low-level: Character-level typos (5-10% intensity)
- Medium-level: Informal language injection (15-25% intensity)
- High-level: Synonym replacement and paraphrasing (40-60% intensity)
"""

import random
import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerturbationEngine:
    """
    Main engine for applying various levels of perturbations to Indonesian text.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the perturbation engine.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize perturbation handlers
        self.low_level = LowLevelPerturbation(random_seed)
        self.medium_level = MediumLevelPerturbation(random_seed)
        self.high_level = HighLevelPerturbation(random_seed)
        
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
        
        perturbed_df = df.copy()
        perturbed_df[text_column] = perturbed_df[text_column].apply(
            lambda x: self.apply_perturbation(x, level, intensity)
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


class LowLevelPerturbation:
    """
    Low-level perturbations: Character-level typos.
    Intensity: 5-10% of characters affected.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize low-level perturbation handler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        
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
            intensity = random.uniform(0.05, 0.10)
        
        chars = list(text)
        num_perturbations = max(1, int(len(chars) * intensity))
        
        # Get indices of alphabetic characters only
        alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
        
        if not alpha_indices:
            return text
        
        # Randomly select characters to perturb
        num_perturbations = min(num_perturbations, len(alpha_indices))
        perturb_indices = random.sample(alpha_indices, num_perturbations)
        
        for idx in perturb_indices:
            chars[idx] = self._apply_typo(chars[idx])
        
        return ''.join(chars)
    
    def _apply_typo(self, char: str) -> str:
        """
        Apply a random typo to a character.
        
        Args:
            char: Character to perturb
            
        Returns:
            Perturbed character
        """
        char_lower = char.lower()
        typo_type = random.choice(['substitute', 'delete', 'insert', 'swap'])
        
        if typo_type == 'substitute' and char_lower in self.keyboard_neighbors:
            # Substitute with keyboard neighbor
            new_char = random.choice(self.keyboard_neighbors[char_lower])
            return new_char.upper() if char.isupper() else new_char
        
        elif typo_type == 'delete':
            # Delete character (return empty string)
            return ''
        
        elif typo_type == 'insert' and char_lower in self.keyboard_neighbors:
            # Insert a neighbor character
            insert_char = random.choice(self.keyboard_neighbors[char_lower])
            return char + insert_char
        
        elif typo_type == 'swap':
            # This will be handled at word level, return original
            return char
        
        return char


class MediumLevelPerturbation:
    """
    Medium-level perturbations: Informal language injection.
    Intensity: 15-25% of words affected.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize medium-level perturbation handler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Indonesian informal language mappings
        self.formal_to_informal = {
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
            'banyak': ['byk', 'banyak banget']
        }
        
        # Common Indonesian slang additions
        self.slang_additions = [
            'sih', 'nih', 'dong', 'deh', 'lah', 'kok', 'kan'
        ]
        
        # Abbreviations
        self.abbreviations = {
            'dan': 'n',
            'di': 'd',
            'ke': 'k',
            'dari': 'dr',
            'sama': 'sm'
        }
        
        logger.info("MediumLevelPerturbation initialized")
    
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
            intensity = random.uniform(0.15, 0.25)
        
        words = text.split()
        num_perturbations = max(1, int(len(words) * intensity))
        
        # Randomly select words to perturb
        perturb_indices = random.sample(range(len(words)), min(num_perturbations, len(words)))
        
        for idx in perturb_indices:
            words[idx] = self._informalize_word(words[idx])
        
        return ' '.join(words)
    
    def _informalize_word(self, word: str) -> str:
        """
        Convert a word to informal Indonesian.
        
        Args:
            word: Word to informalize
            
        Returns:
            Informalized word
        """
        word_lower = word.lower()
        
        # Remove punctuation for matching
        word_clean = re.sub(r'[^\w\s]', '', word_lower)
        
        # Try formal to informal mapping
        if word_clean in self.formal_to_informal:
            informal = random.choice(self.formal_to_informal[word_clean])
            # Preserve original punctuation
            if word != word_lower:
                return word.replace(word_clean, informal)
            return informal
        
        # Try abbreviation
        if word_clean in self.abbreviations and random.random() < 0.5:
            return self.abbreviations[word_clean]
        
        # Add slang particle
        if len(word_clean) > 3 and random.random() < 0.3:
            particle = random.choice(self.slang_additions)
            return f"{word} {particle}"
        
        return word


class HighLevelPerturbation:
    """
    High-level perturbations: Synonym replacement and paraphrasing.
    Intensity: 40-60% of content altered.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize high-level perturbation handler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Indonesian synonym dictionary
        self.synonyms = {
            'besar': ['besar', 'raksasa', 'jumbo', 'gede', 'luas'],
            'kecil': ['kecil', 'mungil', 'mini', 'cilik'],
            'bagus': ['bagus', 'baik', 'oke', 'mantap', 'keren'],
            'buruk': ['buruk', 'jelek', 'tidak baik', 'payah'],
            'cepat': ['cepat', 'kilat', 'gesit', 'laju'],
            'lambat': ['lambat', 'pelan', 'lelet'],
            'tinggi': ['tinggi', 'jangkung', 'menjulang'],
            'rendah': ['rendah', 'pendek'],
            'penting': ['penting', 'krusial', 'vital', 'esensial'],
            'mudah': ['mudah', 'gampang', 'simpel'],
            'sulit': ['sulit', 'susah', 'rumit', 'kompleks'],
            'baru': ['baru', 'anyar', 'fresh'],
            'lama': ['lama', 'lawas', 'usang'],
            'menarik': ['menarik', 'seru', 'asyik', 'keren'],
            'membosankan': ['membosankan', 'ngebosenin', 'monoton'],
            'senang': ['senang', 'gembira', 'bahagia', 'happy'],
            'sedih': ['sedih', 'duka', 'galau'],
            'marah': ['marah', 'kesal', 'jengkel', 'dongkol'],
            'takut': ['takut', 'ngeri', 'seram'],
            'berani': ['berani', 'pemberani', 'gagah'],
            'pintar': ['pintar', 'cerdas', 'pandai', 'jenius'],
            'bodoh': ['bodoh', 'dungu', 'tolol'],
            'cantik': ['cantik', 'indah', 'ayu', 'elok'],
            'jelek': ['jelek', 'buruk rupa'],
            'kaya': ['kaya', 'tajir', 'berada', 'mampu'],
            'miskin': ['miskin', 'papa', 'melarat'],
            'ramai': ['ramai', 'rame', 'hiruk pikuk'],
            'sepi': ['sepi', 'sunyi', 'lengang'],
            'panas': ['panas', 'gerah', 'hangat'],
            'dingin': ['dingin', 'sejuk', 'adem'],
            'terang': ['terang', 'cerah', 'jelas'],
            'gelap': ['gelap', 'remang', 'kelam'],
            'keras': ['keras', 'kuat', 'solid'],
            'lembut': ['lembut', 'halus', 'soft'],
            'mahal': ['mahal', 'pricey', 'selangit'],
            'murah': ['murah', 'terjangkau', 'ekonomis']
        }
        
        # Sentence structure variations
        self.structure_patterns = [
            'passive_to_active',
            'active_to_passive',
            'reorder_clauses'
        ]
        
        logger.info("HighLevelPerturbation initialized")
    
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
            intensity = random.uniform(0.40, 0.60)
        
        # Apply synonym replacement
        text = self._replace_synonyms(text, intensity)
        
        # Apply sentence structure modification (with lower probability)
        if random.random() < 0.3:
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
        replace_indices = random.sample(replaceable_indices, num_replacements)
        
        for idx in replace_indices:
            word = words[idx]
            word_clean = re.sub(r'[^\w\s]', '', word.lower())
            
            if word_clean in self.synonyms:
                # Get synonym (excluding the original word)
                synonym_options = [s for s in self.synonyms[word_clean] if s != word_clean]
                if synonym_options:
                    synonym = random.choice(synonym_options)
                    
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
        if random.random() < 0.5 and len(sentences) >= 2:
            # Swap two adjacent sentences
            idx = random.randint(0, len(sentences) - 2)
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
