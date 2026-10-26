# Tasarım ilkeleri

rmlab'ın baştan uyduğu mühendislik ilkeleri. Her biri sinyal ve ML projelerinde sık yaşanan,
sonradan düzeltmesi pahalı bir hatayı önlemek için var. Her ilkenin karşısında, ilgili PLAN maddesi yazıyor.

---

## Veri

**1. Örnek örnek veri, parça parça okuma.**
- Büyük bir veri setini tek seferde belleğe almak, üstüne her dönüşümde (normalizasyon, STFT) yeni bir tam kopya üretmek, güçlü bir makinede bile belleği tüketir.
- Bu yüzden:
  - Adaptörler veriyi örnek örnek verir. Dönüşümler `Dataset.__getitem__` içinde çalışır.
  - Diziler `float32` olarak tutulur.
  - Büyük dosyalar `h5py` ya da memmap ile parça parça okunur.
- *PLAN 1c.1, 1c.2*

**2. İstatistikler yalnız train'den çıkarılır.**
- Normalizasyon istatistikleri, augmentation ve üretken modeller sadece train bölmesini görür.
- İstatistik için alt-örneklem gerekiyorsa, config'teki seed ile seçilir.
- *CLAUDE.md kesin kural 2*

**3. Fiziksel birimleri koru.**
- Spektral dönüşümlerde gerçek örnekleme frekansı kullanılır.
- Mel ölçeği Hz cinsinden tanımlıdır, normalize frekansta (0-0.5) anlamını yitirir ve pratikte lineer bantlara dönüşür. Normalize frekansta çalışılıyorsa lineer bantlar bilinçli olarak seçilir.
- *PLAN 1c.2*

## Bölme ve değerlendirme

**4. Üç ayrı bölme: train / val / test.**
- Model seçimi, erken durdurma ve eşik kalibrasyonu **val**'de yapılır. Sonuç yalnızca **test**'te raporlanır.
- Seçim ile raporlama aynı veride yapılırsa sonuç iyimser çıkar. OOD eşiği de aynı örneklerde hem kalibre edilip hem ölçülürse, hedef yanlış alarm oranı kurgu gereği tutar.
- *PLAN 2.1, 5.4, 6.5*

**5. Grup bazlı bölme.**
- Aynı kaynaktan türetilmiş örnekler tek bir bölmede kalmalı. Bunlar aynı kaydın parçaları ya da aynı temel sinyalin farklı bozulmalarla çoğaltılmış kopyaları olabilir.
- Aksi hâlde model kaynağı ezberler ve val'de "tanır". Doğruluk gerçekte olduğundan yüksek görünür.
- Bu yüzden:
  - Her örnek `meta["source_id"]` taşır.
  - Bölme bu kimliğe göre yapılır.
  - Sızıntı testi, bölmeler arasında birebir **ve** yakın kopyalara (cosine) bakar.
- *PLAN 1b.4, 2.1, 2.2*

**6. Karıştırma: train'de evet, değerlendirmede hayır.**
- Train loader her epoch karıştırır. Karıştırmamak, modelin sıraya bağlı örüntüler öğrenmesine yol açabilir.
- Val ve test sabit sıradadır.
- Alt-örneklemle ölçülecekse, sabit tohumlu bir indeks listesi kullanılır. Böylece aynı ağırlıklar her zaman aynı skoru alır.
- *PLAN 3.5*

**7. Seçim ölçütü göreve uygun olmalı.**
- En iyi checkpoint'i seçen ölçüt config'te açıkça yazılır.
- Kapalı-küme doğruluğu tavana ulaştığında, OOD veya kalibrasyon kalitesi daha erken checkpoint'leri tercih edebilir. O zaman val üzerinde bu ölçütler de izlenir.
- *PLAN 3.5, 5*

**8. Eşikten bağımsız metrikler ve en az 3 seed.**
- OOD ve anomali tespitinde AUROC, AP ve FPR@95TPR esas alınır. Pozitif sınıfın taban oranı düşükken precision yanıltıcıdır.
- Kararlar tek bir eğitimle değil, en az 3 seed ile, ortalama ± std olarak verilir.
- *PLAN 2.3, 2.4*

## Tekrar üretilebilirlik

**9. Rastgelelik tek bir kaynaktan gelir.**
- Tek bir `seed` var. Bütün rastgelelik `utils.seed.set_seed()` ile ve ondan türetilen `np.random.Generator` / `torch.Generator` nesneleriyle üretilir:
  - bölme
  - model başlatma
  - augmentation
  - alt-örneklem
- DataLoader worker'ları `worker_init_fn` ile farklı ama belirli tohumlar alır.
- Kodun içinde sabit bir tohum (`42`) bulunmaz.
- *PLAN 0.5*

**10. Kaydedilen config, kullanılan config'tir.**
- Config çalışma sırasında değiştirilmez.
- Türetilen değerler (sınıf sayısı, girdi boyutu, çözülmüş cihaz) ayrı bir `resolved` kaydına yazılır.
- Koşu klasörüne konan kopya, bütün türetmeler bittikten sonra, eğitim başlamadan yazılır.
- *PLAN 0.4*

**11. Varsayılanlar tek bir yerde.**
- Bütün varsayılanlar `configs/base.yaml`'da.
- Kodun içinde `cfg.get(k, varsayılan)` ile ikinci bir varsayılan tanımlanmaz.
- Bilinmeyen anahtar hata verir.
- *Uygulandı (0.3)*

## Kod

**12. Hızlı ve yüksek sesle hata ver.**
- Beklenmeyen bir hata eğitimi durdurur. Hata yakalanıp batch atlanarak devam edilmez.
- NaN/Inf sayısı metriklere yazılır, eşiği aşarsa koşu başarısız sayılır.
- Config ve veri uyumsuzlukları (boş val, sınıf sayısı ile çıktı boyutu farkı, hedef dağılım uzunluğu) en başta kontrol edilir.
- *PLAN 3.5*

**13. Tek yol, kayıt defteri.**
- Her bileşen türü (veri, model, kayıp, skorlayıcı, augmenter) için tek bir kayıt defteri (`@register`) ve tek bir kurulum yolu var.
- Kullanılmayan kod silinir.

**14. Davranış testlerle belgelenir.**
- Belge az ve kısa tutulur. Uzun belgeler kodla ayrışır.
- Önemli davranışlar (karıştırma, sızıntı, şekiller, tohum) testlerle kilitlenir.
- Mimari değişince PLAN, README ve CLAUDE.md aynı commit'te güncellenir.

**15. Okunabilir geçmiş.**
- Küçük ve anlamlı commit'ler, açıklayıcı mesajlar.
- Otomatik "snapshot" commit'leri kullanılmaz.

## Taşınmaya değer pratikler

| Pratik | Ne işe yarıyor | PLAN |
|---|---|---|
| `device: auto` + gerçek CUDA testi | Küçük bir tensörle dener, hata alırsa CPU'ya düşer | 0.4 (4b) |
| AMP yalnız CUDA'da; grad clip'ten önce `unscale_` | Karma hassasiyetin doğru kullanımı | 3.5 |
| Scheduler'ın türüne göre `step` | `ReduceLROnPlateau` metrik ister, diğerleri istemez | 3.5 |
| Katmanlı bölme (her sınıf her bölmede) | Küçük sınıflar bölmelerden kaybolmaz | 2.1 |
| Çıkarım paketi (fit edilmiş istatistik + çözülmüş config) | Çıkarım, eğitimdeki ön işlemeyi birebir tekrarlar | 9.2 |
| Export'ta zorunlu parite testi | Grafa gömülen ön işlemenin Python yoluyla aynı sonucu verdiği kanıtlanır. Eşik aşılırsa export iptal edilir | 9.2 |
| Kendi checkpoint'inden warm-start | İsim ve şekil eşleşen ağırlıklar yüklenir | 3.5 |
| Dış checkpoint'lerde `weights_only=True` | Güvenli yükleme | 3.x |
| Kontrollü bozulma taraması (kazanç, DC, gürültü, bant sınırlama) | Ölçüm donanımı farkına dayanıklılığı, sahadan veri gerekmeden ölçer | 5.3 |
| Augmentation ham sinyale, normalizasyondan önce, yalnız train'de | Donanım etkileri fiziksel olarak ham sinyalde oluşur. Augmentation sadece gördüğü bozulma türünü öğretir | 4.2 |
| Veri analizi: birebir ve yakın kopyalar, sınıflar arası çakışma, efektif rank | Veriyi eğitimden önce tanımak | 1c.3 |
