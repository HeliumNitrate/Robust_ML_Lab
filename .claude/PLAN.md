# Proje Planı: Robust ML Lab (`rmlab`)

Telekom ve bankacılığın kesişimindeki problemlere (dengesiz/sınırlı veri, anomali, OOD, güvenilir yapay zekâ) odaklanan, alandan bağımsız ve yeniden kullanılabilir bir makine öğrenmesi pipeline'ı.

- **Birincil alan:** Telekom (RF sinyal, ağ KPI zaman serileri)
- **Kesişim:** Telekom ∩ bankacılık (dengesiz veri, anomali, OOD, açıklanabilirlik)
- **İkincil alan:** Bankacılık (tablo verisi, sahtekârlık)
- **Kural:** Çekirdek kod alandan bağımsız; alanlar veri adaptörü olarak eklenir.

> Proje adı (`rmlab`) geçici, istenirse değiştirilir.

---

## Öncelik etiketleri

| Etiket | Anlamı |
|---|---|
| `[TEMEL]` | Her şeyin üzerine kurulduğu altyapı |
| `[KESİŞİM]` | Telekom ve bankacılıkta ortak |
| `[TELEKOM]` | Telekoma özgü |
| `[GENEL]` | Sektörden bağımsız, sunum ve paketleme |

## Kitap eşlemesi (Understanding Deep Learning)

| Bölüm | Plandaki karşılığı |
|---|---|
| 10 CNN | 3.2 |
| 11 ResNet | 3.3 |
| 12 Transformer | 3.4 |
| 14.3 Üretken model değerlendirme | 4.4 |
| 15 GAN | 4.3 |
| 17 VAE | 4.3, 6.3 |
| 18 Diffusion | 4.3 |
| 13 GNN, 19 RL | Plan dışı, ileride |

---

## Yürütme sırası

1. **Faz A, İskelet:** 0 → 1a (RadioML) → 2
2. **Faz B, İlk modeller:** 3.1, 3.2
3. **Faz C, Kitap gerektirmeyen klasik yöntemler:** 4.1, 4.2, 5.1, 5.4, 6.2
4. **Faz D, Mimari ilerleme:** 3.3, 3.4 (kitapta ilerledikçe)
5. **Faz E, Üretken modeller:** 4.3, 4.4, 6.3 (Bölüm 15-18 ile birlikte)
6. **Faz F, Kesişim deneyleri:** 4.7, 6.1 tablo verisi
7. **Faz G:** 5.2, 5.3, 5.5, 6.4-6.6, 7
8. **Faz H:** 8
9. **Sürekli:** 9 (her faz sonunda README ve tag)

---

## 0. Altyapı ve çalışma düzeni `[TEMEL]`

- [ ] **0.1** Repo iskeleti: `src/rmlab/{data,models,train,eval,scoring,augment,explain}`, `configs/`, `experiments/`, `tests/`, `splits/`
- [ ] **0.2** Ortam: `pyproject.toml`, CPU Dockerfile, README'de komut listesi (Makefile yok: `make` Windows'ta varsayılan değil)
- [ ] **0.3** Config sistemi: her deney bir YAML dosyası (karar: Hydra mı sade YAML mı)
- [ ] **0.4** Deney takibi: koşu klasörü = config kopyası + `metrics.json` + git hash + log
- [ ] **0.5** Tekrar üretilebilirlik: merkezi `set_seed()`, deterministik ayarlar
- [ ] **0.6** Test ve CI: pytest, GitHub Actions (lint + test)
- [ ] **0.7** `CLAUDE.md` güncel tutulur

**Çıktı:** Sahte veriyle uçtan uca çalışan pipeline: `python -m rmlab.train --config configs/smoke.yaml`

## 1. Veri katmanı `[KESİŞİM]`

### 1a. Hazır veri setleri
- [ ] **1a.1** Sinyal: RadioML; sürüm, lisans ve indirme koşullarını doğrula
- [ ] **1a.2** Tablo: kredi kartı sahtekârlık verisi (dengesiz, anonim); lisansı doğrula
- [ ] **1a.3** Zaman serisi anomali: telekom KPI'ya yakın açık veri seti araştır (NAB benzeri)
- [ ] **1a.4** Her veri seti için `docs/data_cards/<ad>.md`: kaynak, lisans, boyut, sınıf dağılımı, bilinen sorunlar

### 1b. Sentetik sinyal üreteci `[TELEKOM]`
- [ ] **1b.1** Modülasyonlar: BPSK, QPSK, 8PSK, 16/64-QAM, FSK, ...
- [ ] **1b.2** Kanal etkileri: AWGN, sönümleme, frekans/faz kayması, zamanlama hatası
- [ ] **1b.3** Parametreler: SNR, örnekleme hızı, örnek uzunluğu
- [ ] **1b.4** Mod 1, tek örnek (sınıflandırma); Mod 2, uzun kayıt + başlangıç/bitiş etiketleri (bölüm 8 için)
- [ ] **1b.5** Doğrulama: RadioML ile spektrum ve istatistik karşılaştırması

### 1c. Ortak veri arayüzü
- [ ] **1c.1** `DatasetAdapter` arayüzü: `Sample(x, y, meta)` döndürür
- [ ] **1c.2** Dönüşümler: I/Q, genlik-faz, STFT; normalizasyon
- [ ] **1c.3** Veri analizi: sınıf/SNR dağılımı, intrinsic boyut, redundancy

**Çıktı:** Her kaynak için `get_dataset(name, split)`

## 2. Değerlendirme protokolü `[TEMEL]`

- [ ] **2.1** Bölme: sabit train/val/test, sınıf + SNR'a göre katmanlı; indeksler `splits/` altına yazılır ve kilitlenir
- [ ] **2.2** Sızıntı kontrolleri (tekrarlı örnek, bölmeler arası çakışma testi)
- [ ] **2.3** Metrikler
  - Sınıflandırma: accuracy, macro-F1, sınıf bazlı recall, karışıklık matrisi, SNR'a göre eğriler
  - Dengesiz veri: PR-AUC
  - OOD / anomali: AUROC, AUPR, FPR@95TPR
  - Kalibrasyon: ECE
- [ ] **2.4** Çoklu seed (en az 3), ortalama ± std
- [ ] **2.5** Raporlama: tüm koşuları tek tabloda toplayan `results.parquet` + standart grafikler

**Çıktı:** `evaluate(model, dataset) -> Report`

## 3. Temel modeller ve eğitim `[KESİŞİM]`

- [ ] **3.1** Klasik baseline: sinyalde elle öznitelik + lojistik regresyon / random forest; tabloda gradient boosting
- [ ] **3.2** 1D CNN (Bölüm 10)
- [ ] **3.3** 1D ResNet + batch norm; derinlik ve residual ablation (Bölüm 11)
- [ ] **3.4** Transformer encoder, sinyali parçalara bölüp token yapma (Bölüm 12)
- [ ] **3.5** Genel trainer: optimizer/scheduler, early stopping, checkpoint, karma hassasiyet
- [ ] **3.6** Hiperparametre araması (Optuna, küçük ölçekli)
- [ ] **3.7** Karşılaştırma raporu: doğruluk, parametre sayısı, çıkarım süresi

**Çıktı:** Model kayıt defteri (`build_model(name, **cfg)`) + genel `Trainer`

## 4. Sınırlı ve dengesiz veri `[KESİŞİM]`

- [ ] **4.1** Deney düzeneği: seçilen sınıf(lar)ı %50 / %10 / %1'e indirme, öğrenme eğrileri
- [ ] **4.2** Klasik çözümler: oversampling, sınıf ağırlıkları, focal loss, sinyal augmentation (gürültü, faz döndürme, zaman kaydırma); tabloda SMOTE
- [ ] **4.3** Üretken augmentation: koşullu VAE → koşullu GAN → diffusion
  - Sadece azınlık sınıfıyla eğitim ve tüm sınıflarla koşullu eğitim karşılaştırması
  - Üretici **yalnızca train bölmesiyle** eğitilir
- [ ] **4.4** Üretim kalitesi: TSTR, dağılım mesafeleri, ezberleme kontrolü (en yakın gerçek örneğe uzaklık)
- [ ] **4.5** Self-supervised ön eğitim (contrastive) + az etiketle ince ayar
- [ ] **4.6** Sonuç: kapanma oranı = (aug − azaltılmış) / (tam − azaltılmış), rapor
- [ ] **4.7** Aynı düzeneği tablo sahtekârlık verisinde tekrarla

**Karşılaştırma kolları (sabit):** azaltılmış · azaltılmış + klasik aug · + VAE · + GAN · + diffusion · tam veri

**Çıktı:** `Augmenter` modülleri + yeniden kullanılabilir deney şablonu

## 5. Belirsizlik, kalibrasyon ve OOD `[KESİŞİM]`

- [ ] **5.1** Kalibrasyon: reliability diyagramı, ECE, temperature scaling
- [ ] **5.2** Belirsizlik: MC dropout, deep ensembles
- [ ] **5.3** OOD senaryoları: yeni sınıf (eğitimden çıkarılan modülasyonlar); dağılım kayması (farklı kanal/SNR); tabloda zamansal kayma
- [ ] **5.4** OOD skorları: MSP, energy, Mahalanobis, ODIN, kNN
- [ ] **5.5** Reddetme seçenekli sınıflandırma: kapsama-risk eğrisi

**Çıktı:** `Scorer` arayüzü (`fit` / `score`); tüm OOD yöntemleri bunun altında

## 6. Anomali tespiti `[KESİŞİM]`

- [ ] **6.1** Problemler: sinyalde bozuk/bilinmeyen yayın; KPI zaman serisi anomalisi; tabloda sahtekârlık (etiket sadece değerlendirmede)
- [ ] **6.2** Klasik: Isolation Forest, One-Class SVM, LOF, istatistiksel eşikler
- [ ] **6.3** Derin: autoencoder / VAE yeniden yapılandırma hatası, Deep SVDD
- [ ] **6.4** Zaman serisi: tahmin hatası tabanlı, pencere tabanlı
- [ ] **6.5** Eşik seçimi: sabit yanlış alarm oranında tespit, maliyete dayalı eşik
- [ ] **6.6** OOD ile karşılaştırma (aynı `Scorer` arayüzü)

**Çıktı:** Anomali skorlayıcıları + eşik seçim aracı

## 7. Açıklanabilirlik `[KESİŞİM]`

- [ ] **7.1** Tablo: SHAP (genel ve tekil), öznitelik önemi
- [ ] **7.2** Sinyal: saliency, 1D Grad-CAM, integrated gradients
- [ ] **7.3** Hata analizi: karışan sınıf ve SNR'lar, hatalı örnek galerisi
- [ ] **7.4** Model kartı: amaç, veri, metrikler, bilinen sınırlar

**Çıktı:** Açıklama ve model kartı üreten script

## 8. Tespit ve bölütleme (1D) `[TELEKOM]`

- [ ] **8.1** Veri: 1b uzun kayıt modu
- [ ] **8.2** Bölütleme: 1D U-Net, örnek bazlı etiket
- [ ] **8.3** Tespit: başlangıç/bitiş tahmini veya kayan pencere + NMS
- [ ] **8.4** Alternatif: spektrogram üzerinde 2D tespit, karşılaştırma
- [ ] **8.5** Metrikler: IoU, olay bazlı precision/recall

**Çıktı:** Tespit modülü

## 9. Paketleme ve sunum `[GENEL]` (sürekli)

- [ ] **9.1** pip ile kurulabilir paket + CLI (`rmlab train | eval | score`)
- [ ] **9.2** Çıkarım servisi: FastAPI, ONNX'e aktarma
- [ ] **9.3** Docker imajı
- [ ] **9.4** README, modül kullanım örnekleri, notebook'lar
- [ ] **9.5** Bulgular raporu / blog yazısı
- [ ] **9.6** Her ana başlık bitince sürüm etiketi (`v0.1-infra`, `v0.2-baselines`, ...)

---

## Açık kararlar

- [x] Config: **OmegaConf** (2026-10-26). Birleştirme, komut satırından ezme ve katı mod hazır geliyor; Hydra'nın çalışma dizinini değiştirmesi ve otomatik koşu klasörü gerekmiyor. Config dosyaları Hydra ile uyumlu, gerekirse geçilebilir
- [ ] Deney takibi: dosya tabanlı mı MLflow mu (başlangıç önerisi: dosya tabanlı)
- [ ] RadioML hangi sürüm (lisans ve boyuta göre)
- [ ] Zaman serisi anomali veri seti seçimi
- [ ] GPU erişimi (yerel / bulut), Docker GPU desteği ne zaman
  - Not: Yerel makinede NVIDIA GPU yok (AMD Vega 3). Küçük deneyler CPU'da, ağır deneyler bulut GPU'da (Colab/Kaggle).
- [ ] Lisans: şimdilik yok (kod görünür, kullanım izni yok). Proje olgunlaşınca yeniden değerlendir: PolyForm Noncommercial, AGPL veya MIT. Not: lisans sonradan eklenebilir, ama verilen bir lisans o sürüm için geri alınamaz.

## Plan dışı, ileride

- SQL (sonuç tablosu üzerinden pratik, DuckDB)
- Graf sinir ağları (Bölüm 13): ağ topolojisi, para akışı grafı
- Pekiştirmeli öğrenme (Bölüm 19)
- Bankacılık genişlemesi: kredi skorlama, model risk yönetimi, drift izleme

## İlerleme günlüğü

| Tarih | Madde | Not |
|---|---|---|
| 2026-10-14 | 0.1 (kısmen) | Klasör iskeleti, `pyproject.toml`, `.gitignore`; `import rmlab` çalışıyor |
| 2026-10-21 | 0.6 (kısmen) | pytest `dev` extra olarak eklendi, `tests/test_import.py` (9 test geçiyor); CI sonraya |
| 2026-10-22 | 0.6 (kısmen) | ruff `dev` extra olarak eklendi (line-length 100, py311, E/F/I/UP/B); `.claude/` hariç; `.gitattributes` ile LF |
| 2026-10-24 | 0 (düzen) | Çalışma düzeni değişti: Claude bulut kopyasında yazar ve push eder, kullanıcı `git pull` ile alır. Bölüm 0 sıralaması sadeleşti (Makefile yok, 4a/4b) |
| 2026-10-26 | 0.3 (kısmen) | OmegaConf ile config yükleme: `configs/base.yaml` + `smoke.yaml`, `utils/config.py` (`load_config`, `save_config`), 8 test; `device: auto` |
| 2026-10-26 | — | `docs/design_principles.md`: baştan uyulacak 15 mühendislik ilkesi, PLAN maddelerine bağlı |
