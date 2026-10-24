# CLAUDE.md

Bu dosya Claude Code'un bu repoda nasıl çalışacağını tanımlar. Görev listesi ve öncelikler için `.claude/PLAN.md`'ye bak.

## Proje

`rmlab` (Robust ML Lab): dengesiz/sınırlı veri, anomali tespiti, OOD ve güvenilir yapay zekâ üzerine alandan bağımsız bir deney ve pipeline kütüphanesi. Birincil alan telekom (RF sinyal, KPI zaman serisi), ikincil alan bankacılık (tablo verisi, sahtekârlık).

Bu aynı zamanda bir **öğrenme projesi**. Kullanıcı *Understanding Deep Learning* kitabını çalışıyor ve modelleri kitaptaki ilerlemesine paralel ekliyor.

## İletişim

- Kullanıcıyla Türkçe konuş.
- Kod, değişken adları, docstring ve commit mesajları İngilizce.
- Kod yazmadan önce yaklaşımı ve geçen kavramları kısaca açıkla. Neden bu seçimi yaptığını belirt. Ayrıntılar için aşağıdaki "Öğrenme modu"na bak.

## Öğrenme modu

Kullanıcı bu projeyi kod yapısını ve yöntemleri öğrenmek için yapıyor. Amaç sadece çalışan kod değil, kullanıcının her parçayı anlatabilmesi ve değiştirebilmesi. Kodun türüne göre üç farklı çalışma şekli var:

| Katman | Kapsam | Kim yazar | Önerilen output style |
|---|---|---|---|
| **A. Altyapı** | `pyproject.toml`, paket düzeni, config yükleme, logging, seed, CI, Docker, CLI | Claude yazar, kullanıcı okur | `Explanatory` |
| **B. Yardımcı katman** | `data/` adaptörleri ve dönüşümler, bölme, `eval/` metrik ve raporlama, `train/` Trainer iskeleti | Claude yazar, kullanıcı üstünde küçük bir değişiklik yapar | `Explanatory` |
| **C. ML çekirdeği** | `models/` mimarileri, kayıp fonksiyonları, eğitim adımı (`train_step`), `scoring/` OOD ve anomali yöntemleri, `augment/` üretken modeller, sentetik sinyal üretecinin fizik kısmı | Kullanıcı yazar | `Learning` |

### Kurallar

1. **Önce anlat.** Her madde için kod yazmadan önce şunları söyle: ne yapacağız, hangi kavramlar geçiyor (her biri 1-2 cümle), hangi dosyalara dokunulacak. Kullanıcı onaylamadan büyük bir parça yazma.
2. **Küçük adımlar.** Bir PLAN maddesini tek seferde bitirme. Her adım bir oturuşta okunabilecek büyüklükte olsun (yaklaşık 100-150 satırdan az değişiklik). Adım sonunda durup kullanıcının okumasını bekle.
3. **Katman C'de kodu kullanıcı yazar.** Fonksiyon imzasını, docstring'i ve testi sen hazırla, gövdeye `TODO(human)` bırak. Ne yazılacağını, girdi/çıktı şekillerini ve hangi tasarım kararlarının tartılması gerektiğini anlat, sonra bekle. Kullanıcı yazınca kodu gözden geçir: doğruluk, şekil hataları, sayısal kararlılık, verimlilik. Hatayı doğrudan düzeltmek yerine önce ipucu ver.
4. **Kullanıcı açıkça "sen yaz" derse** katman C'de de yazabilirsin, ama sonra kodu satır satır açıkla.
5. **Pekiştirme önerisi.** Katman A ve B'de bir adım bitince kullanıcıya küçük bir alıştırma öner: bir parametreyi değiştirmek, yeni bir metrik eklemek, bir testi bilerek bozup hatayı okumak gibi.
6. **Öğrenme notu.** Bir madde bitince kullanıcıdan `docs/learning/<madde-no>-<konu>.md` dosyasına kendi cümleleriyle 3-5 satırlık bir not yazmasını hatırlat. Notu onun yerine yazma. İstenirse okuyup eksik veya yanlış anlaşılan noktayı belirt.
7. **Bitti ölçütü.** Bir madde, kullanıcı (a) ne işe yaradığını anlatabildiğinde ve (b) küçük bir değişikliği yardımsız yapabildiğinde bitmiş sayılır. Emin değilsen kısa bir soru sorarak kontrol et.
8. **Kitapla bağlantı.** Katman C'de ilgili *Understanding Deep Learning* bölümüne atıf yap (örn. "Bölüm 11.4, batch normalization"). Kullanıcı o bölüme henüz gelmediyse bunu belirt ve ya basit bir sürümle ilerlemeyi ya da maddeyi ertelemeyi öner.

### Altyapı (Bölüm 0) için sıralama

Bölüm 0'ı tek seferde kurma. Her biri ayrı bir adım:

1. Klasör yapısı + `pyproject.toml` + sanal ortam, paketin `import rmlab` ile yüklenmesi
2. `pytest` ve ilk test
3. `ruff` (lint ve format)
4. Config ve ilk koşu, iki adımda:
   - 4a. YAML config yükleme (OmegaConf seçilirse): varsayılanlar + deney dosyası + komut satırı
   - 4b. Seed, koşu klasörü (config kopyası, `metrics.json`, git hash) ve sahte veriyle uçtan uca `train` komutu
5. GitHub Actions (CI) ve Docker: ilk gerçek deneyden sonra

Makefile yok: `make` Windows'ta varsayılan olarak kurulu değil. Komutlar aşağıdaki "Komutlar" bölümünde ve README'de.

Bölüm 0'dan sonra 1b (sentetik sinyal üreteci) ile ML'ye geçilir.

## Mimari

Çekirdek alandan bağımsızdır. Alanlar (sinyal, tablo, zaman serisi) adaptör olarak eklenir.

```
src/rmlab/
  data/        # DatasetAdapter'lar, sentetik üreteçler, dönüşümler, bölme
  models/      # model kayıt defteri ve mimariler
  train/       # genel Trainer, callback'ler
  eval/        # metrikler, Report, grafikler
  scoring/     # OOD ve anomali skorlayıcıları (Scorer arayüzü)
  augment/     # klasik ve üretken augmentation (Augmenter arayüzü)
  explain/     # açıklanabilirlik, model kartı
  utils/       # seed, logging, io
configs/       # deney YAML'ları
experiments/   # koşu çıktıları (git'e girmez)
splits/        # kilitli bölme indeksleri (git'e girer)
data/          # ham ve işlenmiş veri (git'e girmez)
docs/data_cards/
tests/
```

### Temel arayüzler

```python
@dataclass
class Sample:
    x: np.ndarray | torch.Tensor
    y: int | None
    meta: dict          # örn. {"snr": 10, "source": "radioml"}

class DatasetAdapter(Protocol):
    def load(self, split: str) -> Sequence[Sample]: ...
    @property
    def num_classes(self) -> int: ...

def get_dataset(name: str, split: str, **cfg) -> Dataset: ...
def build_model(name: str, **cfg) -> nn.Module: ...          # kayıt defteri

class Scorer(Protocol):                                       # OOD ve anomali
    def fit(self, model, train_data) -> "Scorer": ...
    def score(self, data) -> np.ndarray: ...                   # yüksek = daha olağandışı

class Augmenter(Protocol):
    def fit(self, train_data) -> "Augmenter": ...
    def generate(self, n: int, cls: int | None = None) -> Sequence[Sample]: ...

def evaluate(model, dataset, metrics: list[str]) -> Report: ...
```

Yeni bir yöntem eklerken mevcut arayüzün altına ekle, yeni paralel bir yapı kurma. Yeni bileşenler kayıt defterine (`@register("name")`) kaydedilir ve config'ten adıyla çağrılır.

## Kesin kurallar

1. **Test bölmesi kilitlidir.** `splits/` altındaki dosyaları kullanıcının açık onayı olmadan yeniden üretme veya değiştirme.
2. **Veri sızıntısı yok.** Normalizasyon istatistikleri, augmentation ve üretken modeller **yalnızca train bölmesiyle** eğitilir. Test verisi hiçbir eğitim veya seçim adımına girmez; model ve eşik seçimi val bölmesiyle yapılır.
3. **Her deney bir config.** Kodda gömülü hiperparametre yok. Her koşu `experiments/<YYYYMMDD-HHMM>_<ad>/` altına config kopyası, `metrics.json`, git commit hash ve log yazar.
4. **Tekrar üretilebilirlik.** Rastgelelik `utils.seed.set_seed()` üzerinden. Sonuçlar en az 3 seed ile raporlanır.
5. **Metriklere dokunma.** Sonucu iyi göstermek için metrik tanımını, bölmeyi veya değerlendirme kodunu değiştirme. Olumsuz sonuçlar da raporlanır.
6. **Gömülü yol yok.** Yollar config veya ortam değişkeninden gelir.
7. **Veri git'e girmez.** `data/` ve `experiments/` `.gitignore`'da.
8. **Yeni bağımlılık** eklemeden önce kullanıcıya sor.

## Kod standartları

- Python 3.11+, type hint'ler, kısa docstring'ler.
- Formatlama ve lint: `ruff`.
- Her yeni modül için en az bir birim testi (şekil kontrolü, küçük sabit girdide beklenen çıktı).
- Fonksiyonlar küçük ve test edilebilir. Gereğinden fazla soyutlama yapma, ihtiyaç doğunca soyutla.

## Komutlar

```bash
python -m pip install -e ".[dev]"   # paket + geliştirme araçları
python -m pytest                    # testler
ruff check .                        # lint
ruff format --check .               # format kontrolü (düzeltmek için: ruff format .)
python -m rmlab.train --config configs/<deney>.yaml
python -m rmlab.eval  --run experiments/<koşu>
```

## Çalışma şekli

- `.claude/PLAN.md`'den bir seferde **tek madde** üzerinde çalış.
- Başlamadan önce maddenin kapsamını ve dokunacağın dosyaları kısaca söyle.
- Bitince: testleri çalıştır, `.claude/PLAN.md`'de maddeyi işaretle, "İlerleme günlüğü"ne bir satır ekle.
- Küçük, anlamlı commit'ler. Bir ana başlık bitince sürüm etiketi öner.
- Bir karar `.claude/PLAN.md`'deki "Açık kararlar" listesindeyse, kendin karar verme; seçenekleri ve önerini sun.
- İki kopya düzeni: Claude bulut kopyasında dosyaları yazar, testleri ve ruff'ı çalıştırır, commit ve push eder. Kullanıcı yerelde (`D:\projeler\Robust_ML_Lab`) `git pull` ile alır ve VS Code'da okur.
  - Claude her adıma başlamadan önce `git pull` yapar. Kullanıcı kendi tarafında bir şey push ettiğinde haber verir.
  - İki taraf aynı dosyayı aynı anda değiştirmez.
  - Windows'a özgü bir şey (kurulum, yol, PowerShell) gerektiğinde Claude komutu ve ne işe yaradığını söyler, kullanıcı çalıştırıp çıktıyı paylaşır.
- Her adımın sonunda bir okuma rehberi ver: yeni ve değişen dosyalar; her dosyadaki fonksiyon ve class'lar (birer cümleyle); bunları kimin çağırdığı ve neyi kullandıkları; önerilen okuma sırası.
- Referans repo: `HeliumNitrate/NN_training` (yerelde `../NN_training`). Yalnızca okunur, hiçbir dosyası değiştirilmez. Bir PLAN maddesine başlarken orada benzer bir şey yapılmış mı bak (özellikle `docs/FINDINGS.md`, `docs/CHANGELOG.md`, `docs/RESULTS.md`). Örnek alınacak veya kaçınılacak bir şey bulursan söyle. Kodu kopyalama, rmlab'ın mimarisine uyarlayarak yeniden yaz.
