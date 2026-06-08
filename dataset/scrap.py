import io
import re
import time
import gzip
import random
import urllib.parse
from collections import deque
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import pandas as pd

class UniversalNewsEngine:
    def __init__(self, target_count=1000):
        self.target_count = target_count
        self.user_agents = [
          # Customize based on your host / browser
        ]

    def _get_source_name(self, url):
        url_lower = url.lower()
        if "cnnindonesia.com" in url_lower: return "CNN Indonesia"
        if "kompas.com" in url_lower: return "Kompas"
        if "tribunnews.com" in url_lower: return "Tribun News"
        if "detik.com" in url_lower: return "Detik"
        if "tempo.co" in url_lower: return "Tempo"
        return None

    def _extract_date(self, url, lastmod_text, pub_date_text):
        for text in [pub_date_text, lastmod_text]:
            if text:
                match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
                if match:
                    year, month, day = match.groups()
                    if 2022 <= int(year) <= 2026:
                        return f"{year}-{month}-{day}"

        match_slash = re.search(r"/(\d{4})/(\d{2})/(\d{2})", url)
        if match_slash:
            year, month, day = match_slash.groups()
            if 2022 <= int(year) <= 2026:
                return f"{year}-{month}-{day}"

        match_cnn = re.search(r"/(\d{4})(\d{2})(\d{2})\d*-", url)
        if match_cnn:
            year, month, day = match_cnn.groups()
            if 2022 <= int(year) <= 2026:
                return f"{year}-{month}-{day}"

        return None

    def _extract_title(self, url, xml_title, source):
        """ALWAYS fetch title from article page for maximum accuracy"""
        # ALWAYS fetch from actual article page - XML titles are unreliable
        # They often contain image captions, gallery titles, or outdated content
        fetched_title = self._fetch_title_from_page(url, source)
        
        if fetched_title and len(fetched_title) > 15:
            return fetched_title
        
        # Only use XML as absolute last resort if page fetch fails
        if xml_title and len(xml_title.strip()) > 20:
            xml_lower = xml_title.lower().strip()
            # Skip if it's clearly a caption
            if not any(xml_lower.startswith(prefix) for prefix in ["ilustrasi", "foto", "gambar", "image", "kumpulan"]):
                cleaned = re.sub(r'\s+\d{6,}$', '', xml_title.strip())
                if len(cleaned) > 15:
                    return cleaned
        
        # Final fallback to URL parsing
        return self._fallback_title_from_url(url)
    
    def _fetch_title_from_page(self, url, source):
        """Fetch actual title from article page - most reliable method"""
        try:
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "id,en-US;q=0.7,en;q=0.3"
            }
            time.sleep(random.uniform(0.5, 1.0))
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, "html.parser")
            title = None
            
            # Method 1: h1 tag (article headline - MOST ACCURATE for actual content)
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text().strip()
                # Validate it's substantial
                if len(title) > 15:
                    title_lower = title.lower()
                    # Make sure it's not a caption
                    if not any(title_lower.startswith(prefix) for prefix in ["ilustrasi", "foto", "gambar", "kumpulan foto", "kumpulan gambar"]):
                        return self._clean_title(title)
            
            # Method 2: og:title meta tag (reliable for social sharing)
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
                if len(title) > 15:
                    title_lower = title.lower()
                    if not any(title_lower.startswith(prefix) for prefix in ["ilustrasi", "foto", "gambar", "kumpulan"]):
                        return self._clean_title(title)
            
            # Method 3: twitter:title meta tag
            twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
            if twitter_title and twitter_title.get("content"):
                title = twitter_title["content"].strip()
                if len(title) > 15:
                    title_lower = title.lower()
                    if not any(title_lower.startswith(prefix) for prefix in ["ilustrasi", "foto", "gambar", "kumpulan"]):
                        return self._clean_title(title)
            
            # Method 4: <title> tag (with cleanup)
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()
                # Remove site name suffixes
                for suffix in [" - CNN Indonesia", " - Kompas.com", " - detikcom", " - Tempo.co",
                               " - Tribunnews.com", " - Tribunnews", " | Kompas.com", " | CNN Indonesia",
                               " - Tekno Kompas.com", " - Health Kompas.com"]:
                    if title.endswith(suffix):
                        title = title[:-len(suffix)].strip()
                
                if len(title) > 15:
                    title_lower = title.lower()
                    if not any(title_lower.startswith(prefix) for prefix in ["ilustrasi", "foto", "gambar", "kumpulan"]):
                        return self._clean_title(title)
            
            return None
            
        except Exception:
            return None
    
    def _clean_title(self, title):
        """Clean and validate title"""
        if not title:
            return None
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove trailing article IDs
        title = re.sub(r'\s+\d{6,}$', '', title)
        
        # Remove common prefixes that indicate non-article content
        title = re.sub(r'^(FOTO:|VIDEO:|INFOGRAFIS:|BREAKING NEWS:|Ilustrasi|Gambar|Kumpulan Foto|Kumpulan Gambar)\s*[:\-]?\s*', '', title, flags=re.IGNORECASE)
        
        # Final validation
        title_lower = title.lower()
        if (len(title) > 15 and
            not title_lower.startswith("ilustrasi") and
            not title_lower.startswith("foto") and
            not title_lower.startswith("gambar") and
            not title_lower.startswith("kumpulan")):
            return title
        
        return None
    
    def _fallback_title_from_url(self, url):
        """Extract title from URL as last resort"""
        try:
            path = urllib.parse.urlparse(url).path
            parts = [p for p in path.split("/") if p]
            if not parts:
                return "News Article"
            
            last_part = parts[-1]
            # Remove file extensions
            last_part = re.sub(r"\.(html|htm|php|asp)$", "", last_part)
            # Remove date patterns
            last_part = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", last_part)
            last_part = re.sub(r"^\d{8,}-", "", last_part)
            # Remove trailing numbers
            last_part = re.sub(r"-\d{6,}$", "", last_part)
            
            # Convert to readable title
            title_cleaned = last_part.replace("-", " ").replace("_", " ").strip().title()
            return title_cleaned if len(title_cleaned) > 10 else "News Article"
        except Exception:
            return "News Article"


    def run_harvest(self, config):
        """Executes targeted collection using specific domain criteria parameters."""
        sitemap_queue = deque(config["seed_sitemaps"])
        visited_sitemaps = set()
        collected_articles = []
        seen_urls = set()
        seen_titles = set()
        
        headers = {"User-Agent": self.user_agents[0], "Accept-Encoding": "gzip, deflate"}
        print(f"\n[ENGINE] Starting collection profile for task: {config['domain_label'].upper()}")
        
        progress_bar = tqdm(total=self.target_count, desc=f"Harvesting {config['domain_label']}", unit="rows")
        
        while sitemap_queue and len(collected_articles) < self.target_count:
            current_sitemap = sitemap_queue.popleft()
            if current_sitemap in visited_sitemaps: continue
            visited_sitemaps.add(current_sitemap)
            
            progress_bar.set_postfix({"Sitemap": current_sitemap.split('/')[-1][:20]}, refresh=True)
            
            try:
                time.sleep(0.4)
                response = requests.get(current_sitemap, headers=headers, timeout=12)
                if response.status_code != 200: continue
                    
                content = response.content
                if current_sitemap.endswith(".gz") or response.headers.get("Content-Encoding") == "gzip":
                    try:
                        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f: xml_content = f.read()
                    except Exception: xml_content = content
                else:
                    xml_content = content
                    
                try:
                    root = ET.fromstring(xml_content)
                except ET.ParseError: continue
                    
                for child in root:
                    tag_name = child.tag.split("}")[-1]
                    
                    if tag_name == "sitemap":
                        loc_elem = child.find("{*}loc")
                        if loc_elem is not None and loc_elem.text:
                            sub_url = loc_elem.text.strip().lower()
                            
                            has_year = any(yr in sub_url for yr in ["2022", "2023", "2024", "2025", "2026"])
                            has_structural_match = any(sec in sub_url for sec in config["sitemap_filters"])
                            
                            if has_year or has_structural_match or len(visited_sitemaps) < 15:
                                if loc_elem.text.strip() not in visited_sitemaps and loc_elem.text.strip() not in sitemap_queue:
                                    sitemap_queue.append(loc_elem.text.strip())
                                    
                    elif tag_name == "url":
                        loc_elem = child.find("{*}loc")
                        if loc_elem is not None and loc_elem.text:
                            url = loc_elem.text.strip()
                            if url in seen_urls: continue
                            
                            # URL validation - must be a proper article URL
                            parsed_url = urllib.parse.urlparse(url)
                            path_parts = [p for p in parsed_url.path.split('/') if p]
                            
                            # Skip homepage, category pages, and invalid URLs
                            if (len(path_parts) < 2 or  # Too short to be an article
                                url.endswith('/') or  # Directory/category page
                                not any(char.isalnum() for char in parsed_url.path) or  # No content in path
                                len(parsed_url.path) < 10):  # Path too short
                                continue
                                
                            source = self._get_source_name(url)
                            if not source: continue
                                
                            lastmod_text = child.find("{*}lastmod").text.strip() if child.find("{*}lastmod") is not None else ""
                            pub_date_text = child.find(".//{*}publication_date").text.strip() if child.find(".//{*}publication_date") is not None else ""
                            xml_title = child.find(".//{*}title").text.strip() if child.find(".//{*}title") is not None else ""
                            
                            date_str = self._extract_date(url, lastmod_text, pub_date_text)
                            if not date_str: continue
                                
                            title_str = self._extract_title(url, xml_title, source)
                            
                            # Skip if title extraction failed or is too short
                            if not title_str or len(title_str) < 20:
                                continue
                            
                            # Content filter evaluation
                            combined_text = f"{url.lower()} {title_str.lower()}"
                            if not any(keyword in combined_text for keyword in config["content_keywords"]): continue
                            
                            # Exclusion filter (if configured)
                            if "exclusion_keywords" in config:
                                if any(keyword in combined_text for keyword in config["exclusion_keywords"]): continue
                            
                            if title_str.lower() in seen_titles: continue
                                
                            seen_urls.add(url)
                            seen_titles.add(title_str.lower())
                            
                            collected_articles.append({
                                "id": len(collected_articles) + 1,
                                "source": source,
                                "date": date_str,
                                "title": title_str,
                                "url": url,
                                "label": ""
                            })
                            
                            progress_bar.update(1)
                            if len(collected_articles) >= self.target_count: break
            except Exception:
                continue
                
        progress_bar.close()
        
        # Save output to Excel
        try:
            df = pd.DataFrame(collected_articles)
            df = df[["id", "source", "date", "title", "url", "label"]]  # Reorder columns
            df.to_excel(config["output_file"], index=False, engine='openpyxl')
            print(f"[SUCCESS] Saved {len(collected_articles)} rows to {config['output_file']}\n")
        except Exception as e:
            print(f"[ERROR] Failed to save Excel output: {e}")

# ==========================================
# HOW TO RUN THE CONFIGURATIONS
# ==========================================
if __name__ == "__main__":
    # Initialize the engine once
    scraper = UniversalNewsEngine(target_count=1000)

    # Profile 1: Technology News (Strict filtering)
    tech_config = {
        "domain_label": "Technology",
        "output_file": "indonesian_tech_dataset.xlsx",
        "seed_sitemaps": [
            "https://www.cnnindonesia.com/teknologi/sitemap_web.xml",
            "https://tekno.kompas.com/sitemap.xml",
            "https://inet.detik.com/sitemap.xml",
            "https://www.tempo.co/sitemap_index.xml",
            "https://www.tribunnews.com/sitemap.xml",
            # Additional tech sources
            "https://www.cnnindonesia.com/teknologi/teknologi-informasi/sitemap_web.xml"
        ],
        "sitemap_filters": ["teknologi", "tekno", "inet", "gadget", "gawai", "digital", "startup", "techno"],
        "content_keywords": [
            # Core tech terms
            "teknologi", "tekno", "gadget", "gawai", "perangkat", "smartphone", "ponsel", "tablet",
            # Software & Apps
            "aplikasi", "software", "perangkat lunak", "program", "platform", "sistem operasi", "android", "ios", "windows",
            # Internet & Digital
            "internet", "digital", "website", "situs", "media sosial", "medsos", "facebook", "instagram", "twitter", "tiktok", "youtube",
            # AI & Advanced Tech
            "artificial intelligence", "kecerdasan buatan", "machine learning", "robot", "otomasi", "automasi",
            # Security & Cyber
            "hacker", "siber", "cyber", "keamanan", "security", "privasi", "enkripsi", "malware", "virus",
            # Business & Startup
            "startup", "unicorn", "decacorn", "teknologi finansial", "fintech", "e-commerce", "tokopedia", "shopee", "gojek", "grab",
            # Crypto & Blockchain
            "kripto", "cryptocurrency", "bitcoin", "blockchain", "nft", "metaverse",
            # Telecom & Network
            "telkomsel", "indosat", "smartfren", "5g", "4g", "jaringan", "provider", "operator",
            # Government & Regulation (tech-specific)
            "kominfo", "kementerian komunikasi", "regulasi digital", "literasi digital",
            # Gaming
            "game", "gaming", "esport", "mobile legends", "pubg", "free fire", "steam", "playstation", "xbox", "nintendo",
            # Innovation & Products
            "inovasi", "teknologi terbaru", "peluncuran", "rilis", "update", "fitur baru", "spesifikasi", "review"
        ],
        "exclusion_keywords": [
            # Weather & Climate (NOT technology)
            "cuaca", "iklim", "hujan", "kemarau", "el nino", "la nina", "bmkg", "badan meteorologi",
            "suhu", "temperatur", "angin", "badai", "topan", "siklon", "banjir", "longsor",
            # Natural Disasters
            "gempa", "tsunami", "gunung meletus", "erupsi", "bencana alam",
            # Agriculture & Environment (unless tech-related)
            "pertanian", "panen", "petani", "sawah", "ladang", "tanaman pangan",
            # Politics
            "pemilu", "pilpres", "capres", "partai politik", "dpr", "mpr",
            # Sports
            "sepak bola", "liga", "pertandingan", "turnamen", "piala",
            # Entertainment/Celebrity
            "artis", "selebriti", "konser", "film", "sinetron",
            # Health/Medical (unless health tech)
            "penyakit", "virus corona", "covid", "vaksin", "rumah sakit", "dokter"
        ]
    }

    # Profile 2: Political News (Strict filtering)
    politics_config = {
        "domain_label": "Politics",
        "output_file": "indonesian_politics_dataset.xlsx",
        "seed_sitemaps": [
            "https://www.cnnindonesia.com/nasional/sitemap_web.xml",
            "https://www.kompas.com/sitemap.xml",
            "https://news.detik.com/berita/sitemap_web.xml",
            "https://www.tempo.co/sitemap_index.xml",
            "https://www.tribunnews.com/sitemap.xml",
            # Additional politics sources
            "https://www.cnnindonesia.com/nasional/politik/sitemap_web.xml",
            "https://nasional.kompas.com/sitemap.xml"
        ],
        "sitemap_filters": ["nasional", "politik", "pemilu", "tag"],
        "content_keywords": [
            # Core political terms
            "politik", "nasional", "pemerintah", "pemerintahan", "negara", "kebijakan", "regulasi",
            # Elections
            "pemilu", "pemilihan umum", "pilpres", "pilkada", "pilgub", "pilkada serentak", "pemilihan", "kampanye",
            # Candidates & Leaders
            "capres", "cawapres", "calon presiden", "calon wakil presiden", "jokowi", "prabowo", "gibran", "anies", "ganjar", "ridwan kamil", "ahok", "megawati", "sby",
            # Government Institutions
            "dpr", "mpr", "dprd", "presiden", "wakil presiden", "menteri", "kementerian", "kabinet", "istana", "senayan",
            # Political Parties
            "partai", "parpol", "pdip", "golkar", "gerindra", "demokrat", "pks", "pan", "nasdem", "ppp", "pkb", "perindo",
            # Electoral Bodies
            "kpu", "bawaslu", "komisi pemilihan umum", "pengawas pemilu",
            # Political Issues
            "koalisi", "oposisi", "demokrasi", "reformasi", "korupsi", "kpk", "komisi pemberantasan korupsi",
            # Parliament & Legislation
            "rapat paripurna", "sidang", "undang-undang", "ruu", "peraturan", "legislasi",
            # Regional Politics
            "gubernur", "bupati", "walikota", "provinsi", "kabupaten",
            # Political Events
            "demonstrasi", "unjuk rasa", "aksi", "orasi", "deklarasi", "konvensi", "kongres"
        ],
        "exclusion_keywords": [
            # Technology
            "smartphone", "gadget", "aplikasi", "software", "game", "gaming",
            # Sports
            "sepak bola", "liga", "pertandingan", "turnamen", "piala", "atlet",
            # Entertainment
            "artis", "selebriti", "konser", "film", "sinetron",
            # Weather
            "cuaca", "hujan", "kemarau", "bmkg"
        ]
    }

    # Profile 3: Health News (Enriched keywords for better coverage)
    health_config = {
        "domain_label": "Health",
        "output_file": "indonesian_health_dataset.xlsx",
        "seed_sitemaps": [
            "https://www.cnnindonesia.com/gaya-hidup/sitemap_web.xml",
            "https://health.kompas.com/sitemap.xml",
            "https://health.detik.com/sitemap.xml",
            "https://www.tempo.co/sitemap_index.xml",
            "https://www.tribunnews.com/sitemap.xml",
            # Additional health sources
            "https://www.tvonenews.com/lifestyle/kesehatan/sitemap.xml",
            "https://www.tribunnews.com/kesehatan/sitemap.xml"
        ],
        "sitemap_filters": ["kesehatan", "health", "gaya-hidup", "lifestyle", "medis", "tvonenews"],
        "content_keywords": [
            # Core health terms (EXPANDED)
            "kesehatan", "health", "medis", "kedokteran", "rumah sakit", "klinik", "puskesmas",
            "sehat", "sakit", "penyakit", "gejala", "diagnosis", "kondisi medis",
            
            # Diseases & Conditions (EXPANDED)
            "covid", "corona", "virus", "flu", "demam", "batuk", "pilek", "diare", "mual", "muntah",
            "diabetes", "hipertensi", "darah tinggi", "jantung", "stroke", "kanker", "tumor",
            "hepatitis", "tuberkulosis", "tbc", "malaria", "dengue", "demam berdarah", "dbd",
            "asma", "alergi", "migrain", "sakit kepala", "vertigo", "maag", "asam lambung",
            "kolesterol", "obesitas", "kegemukan", "kurus", "berat badan",
            "infeksi", "radang", "peradangan", "luka", "cedera", "patah tulang",
            
            # Medical Professionals (EXPANDED)
            "dokter", "dr", "perawat", "bidan", "apoteker", "tenaga medis", "nakes", "tenaga kesehatan",
            "spesialis", "ahli", "konsultan", "praktisi", "medis",
            
            # Treatment & Medicine (EXPANDED)
            "obat", "vaksin", "vaksinasi", "imunisasi", "suntik", "injeksi",
            "terapi", "pengobatan", "perawatan", "operasi", "bedah", "pembedahan",
            "resep", "dosis", "antibiotik", "vitamin", "suplemen", "herbal", "jamu", "tradisional",
            "tablet", "kapsul", "sirup", "salep", "krim",
            
            # Healthcare System (EXPANDED)
            "bpjs", "bpjs kesehatan", "asuransi kesehatan", "kemenkes", "kementerian kesehatan", "dinkes",
            "faskes", "fasilitas kesehatan", "pelayanan kesehatan", "layanan kesehatan",
            "rawat inap", "rawat jalan", "igd", "ugd", "emergency",
            
            # Public Health (EXPANDED)
            "pandemi", "epidemi", "wabah", "outbreak", "karantina", "isolasi", "lockdown",
            "protokol kesehatan", "prokes", "masker", "hand sanitizer", "social distancing",
            "vaksinasi massal", "imunisasi", "screening", "tes kesehatan", "pemeriksaan",
            
            # Nutrition & Wellness (EXPANDED)
            "gizi", "nutrisi", "diet", "makanan sehat", "pola makan", "kalori", "protein", "karbohidrat",
            "lemak", "serat", "mineral", "air putih", "hidrasi", "dehidrasi",
            "menu sehat", "resep sehat", "tips sehat", "pola hidup sehat",
            "olahraga", "aktivitas fisik", "fitness", "yoga", "meditasi", "relaksasi",
            
            # Mental Health (EXPANDED)
            "kesehatan mental", "mental health", "psikologi", "psikiater", "psikolog",
            "depresi", "stress", "stres", "anxiety", "cemas", "gangguan jiwa", "gangguan mental",
            "kesehatan jiwa", "konseling", "terapi mental", "burnout",
            
            # Maternal & Child Health (EXPANDED)
            "ibu hamil", "kehamilan", "persalinan", "melahirkan", "bayi", "balita", "anak",
            "posyandu", "stunting", "tumbuh kembang", "imunisasi anak", "asi", "menyusui",
            "ibu dan anak", "kesehatan ibu", "kesehatan anak", "pediatri",
            
            # Body Parts & Systems (NEW)
            "mata", "telinga", "hidung", "tenggorokan", "mulut", "gigi", "kulit",
            "rambut", "kuku", "tulang", "sendi", "otot", "darah", "pembuluh darah",
            "paru", "paru-paru", "ginjal", "hati", "liver", "lambung", "usus",
            "otak", "syaraf", "saraf", "jantung", "pankreas",
            
            # Symptoms & Signs (NEW)
            "nyeri", "sakit", "pusing", "lemas", "lelah", "letih", "sesak nafas",
            "sesak napas", "bengkak", "gatal", "ruam", "bintik merah",
            
            # Medical Procedures (NEW)
            "rontgen", "ct scan", "mri", "usg", "ultrasonografi", "endoskopi",
            "biopsi", "transfusi", "cuci darah", "dialisis", "kemoterapi", "radioterapi",
            
            # Lifestyle & Prevention (NEW)
            "pencegahan", "preventif", "deteksi dini", "check up", "medical check up",
            "gaya hidup", "pola hidup", "kebiasaan sehat", "tips kesehatan",
            "bahaya", "risiko", "faktor risiko", "komplikasi",
            
            # Medical Research (EXPANDED)
            "penelitian medis", "riset kesehatan", "studi", "uji klinis", "clinical trial",
            "temuan", "penemuan", "inovasi medis", "teknologi medis"
        ],
        "exclusion_keywords": [
            # Politics (keep strict)
            "pemilu", "pilpres", "capres", "partai", "dpr", "mpr", "kpu", "politik",
            # Technology (unless health tech - be selective)
            "smartphone", "gadget", "game", "gaming",
            # Sports (unless health/fitness related - be selective)
            "sepak bola", "liga", "pertandingan", "turnamen", "piala", "timnas",
            # Entertainment
            "artis", "selebriti", "konser", "film", "sinetron", "musik",
            # Weather (unless health impact)
            "cuaca", "bmkg", "el nino"
        ]
    }

    # Profile 4: Sports News (Strict filtering)
    sport_config = {
        "domain_label": "Sport",
        "output_file": "indonesian_sport_dataset.xlsx",
        "seed_sitemaps": [
            "https://www.cnnindonesia.com/olahraga/sitemap_web.xml",
            "https://bola.kompas.com/sitemap.xml",
            "https://sport.detik.com/sitemap.xml",
            "https://www.tempo.co/sitemap_index.xml",
            "https://www.tribunnews.com/sitemap.xml",
            # Additional sport sources
            "https://www.kompas.com/sports/sitemap.xml",
            "https://www.tvonenews.com/sport/sitemap.xml",
            "https://www.tribunnews.com/sport/sitemap.xml"
        ],
        "sitemap_filters": ["olahraga", "sport", "bola", "sepakbola", "sports"],
        "content_keywords": [
            # Core sports terms
            "olahraga", "sport", "atlet", "olahragawan", "olahragawati", "pelatih", "coach",
            # Football/Soccer
            "sepak bola", "sepakbola", "bola", "football", "soccer", "pemain", "striker", "kiper", "gelandang", "bek",
            "liga", "pertandingan", "laga", "tanding", "turnamen", "kompetisi", "piala",
            "timnas", "tim nasional", "garuda", "pssi", "persib", "persija", "arema", "persebaya",
            "premier league", "liga inggris", "la liga", "serie a", "bundesliga", "liga champions", "champions league",
            "piala dunia", "world cup", "piala asia", "sea games", "asian games",
            # Badminton
            "bulutangkis", "badminton", "pbsi", "all england", "bwf", "thomas cup", "uber cup",
            "ganda", "tunggal", "smash", "rally",
            # Basketball
            "basket", "basketball", "nba", "perbasi", "ibl", "dunk", "three point",
            # Volleyball
            "voli", "volleyball", "proliga", "pbvsi", "spike", "block", "libero",
            # Other Sports
            "tenis", "tennis", "renang", "swimming", "atletik", "lari", "marathon", "balap", "racing",
            "tinju", "boxing", "mma", "pencak silat", "silat", "karate", "taekwondo", "judo",
            "angkat besi", "weightlifting", "senam", "gymnastics", "panahan", "archery",
            "esport", "e-sport", "mobile legends", "dota", "pubg",
            # Events & Competitions
            "olimpiade", "olympic", "pon", "pekan olahraga nasional", "sea games", "asian games", "asean games",
            # Results & Achievements
            "juara", "champion", "menang", "kalah", "skor", "score", "gol", "goal", "medali", "emas", "perak", "perunggu",
            "rekor", "record", "prestasi", "kemenangan", "kekalahan", "hasil pertandingan",
            # Transfers & Contracts
            "transfer", "kontrak", "rekrut", "datangkan", "lepas", "pinjam", "loan"
        ],
        "exclusion_keywords": [
            # Politics
            "pemilu", "pilpres", "capres", "partai", "dpr", "mpr", "kpu", "politik",
            # Technology (unless sports tech)
            "smartphone", "gadget", "aplikasi", "software",
            # Entertainment
            "artis", "selebriti", "konser", "film", "sinetron",
            # Weather
            "cuaca", "hujan", "kemarau", "bmkg", "el nino",
            # Health (unless sports health)
            "penyakit", "virus", "covid", "vaksin", "rumah sakit"
        ]
    }

    # Profile 5: Education News (Strict filtering to exclude politics)
    education_config = {
        "domain_label": "Education",
        "output_file": "indonesian_education_dataset.xlsx",
        "seed_sitemaps": [
            # Additional education sources
            "https://www.detik.com/edu/sitemap.xml",
            "https://www.kompas.com/edu/sitemap.xml",
            "https://www.cnnindonesia.com/nasional/sitemap_web.xml",
            "https://www.kompas.com/sitemap.xml",
            "https://news.detik.com/berita/sitemap_web.xml",
            "https://www.tempo.co/sitemap_index.xml",
            "https://www.tribunnews.com/sitemap.xml"
        ],
        "sitemap_filters": ["pendidikan", "edukasi", "kampus", "sekolah", "universitas", "edu"],
        "content_keywords": [
            # Core education terms (MUST HAVE)
            "pendidikan", "edukasi", "education", "belajar", "pembelajaran", "pengajaran", "mengajar",
            # Educational Institutions
            "sekolah", "sd", "smp", "sma", "smk", "madrasah", "pesantren", "pondok",
            "universitas", "kampus", "perguruan tinggi", "institut", "politeknik", "akademi",
            "ui", "ugm", "itb", "ipb", "its", "unair", "undip", "unpad", "bina nusantara", "binus",
            # Students & Teachers
            "siswa", "murid", "pelajar", "mahasiswa", "mahasiswi", "anak didik",
            "guru", "dosen", "pengajar", "pendidik", "tenaga pendidik", "kepala sekolah", "rektor", "dekan",
            # Academic Programs
            "kurikulum", "silabus", "mata pelajaran", "mapel", "mata kuliah", "jurusan", "prodi", "program studi",
            "fakultas", "sarjana", "magister", "doktor", "diploma",
            # Exams & Assessment
            "ujian", "ujian nasional", "utbk", "sbmptn", "snmptn", "snbp", "snbt",
            "nilai", "rapor", "ijazah", "sertifikat", "akreditasi", "evaluasi",
            # Scholarships & Funding
            "beasiswa", "scholarship", "bantuan pendidikan", "pip", "kip", "bidikmisi",
            # Government & Policy (Education specific)
            "kemendikbud", "kemdikbud", "kementerian pendidikan", "nadiem makarim", "mendikbud", "menteri pendidikan",
            "dinas pendidikan", "diknas", "kebijakan pendidikan",
            # School Activities
            "ppdb", "penerimaan peserta didik baru", "pendaftaran", "seleksi", "penerimaan mahasiswa baru", "pmb",
            "wisuda", "graduation", "kelulusan", "lulus", "alumni",
            "ospek", "orientasi", "mpls", "masa pengenalan lingkungan sekolah",
            # Learning Methods
            "daring", "luring", "hybrid", "blended learning", "e-learning",
            "ptm", "pembelajaran tatap muka", "kelas online", "zoom", "google classroom",
            # Educational Issues
            "literasi", "numerasi", "asesmen", "kompetensi", "karakter", "pendidikan karakter",
            "bullying", "perundungan",
            # Vocational & Skills
            "vokasi", "kejuruan", "pelatihan", "keterampilan", "skill",
            # Research & Innovation
            "penelitian", "riset", "karya ilmiah", "skripsi", "tesis", "disertasi", "jurnal", "publikasi"
        ],
        "exclusion_keywords": [
            # Political terms to exclude
            "pemilu", "pilpres", "pilkada", "pilgub", "capres", "cawapres", "kampanye",
            "partai", "parpol", "pdip", "golkar", "gerindra", "demokrat", "pks", "pan", "nasdem",
            "koalisi", "oposisi", "dpr", "mpr", "dprd", "kpu", "bawaslu",
            "jokowi", "prabowo", "gibran", "anies", "ganjar", "megawati",
            "demonstrasi", "unjuk rasa", "aksi massa", "orasi politik",
            "pemilihan umum", "calon presiden", "calon wakil presiden",
            "rapat paripurna", "sidang dpr", "fraksi", "legislasi politik",
            # Crime/Violence (non-educational)
            "korupsi", "kpk", "penangkapan", "tersangka", "terdakwa", "pengadilan",
            "pembunuhan", "pencurian", "perampokan", "penipuan",
            # Sports (to avoid confusion)
            "sepak bola", "sepakbola", "liga", "pertandingan", "turnamen", "piala",
            "timnas", "atlet", "pelatih", "juara", "medali",
            # Entertainment/Celebrity
            "artis", "selebriti", "konser", "film", "sinetron"
        ]
    }

    # Execute all 5 profiles sequentially
    scraper.run_harvest(tech_config)
    scraper.run_harvest(politics_config)
    scraper.run_harvest(health_config)
    scraper.run_harvest(sport_config)
    scraper.run_harvest(education_config)
