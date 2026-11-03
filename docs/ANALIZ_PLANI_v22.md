# Revizyon v22 — Analiz planı (sonuçlar görülmeden önce yazıldı)

Tarih: 2 Kasım 2026.

> Not (3 Kasım 2026): Bu yayımlanan kopyada, önceki bir dergi gönderimiyle ilgili iki bağlam satırı çıkarıldı. Tasarım, kohort, bölme, hedef, seçim ve karşılaştırma bölümleri değişmedi. Özgün dosyanın SHA-256 değeri: 389e6dc6c018c78965611f14950f9bfda40db6575d10a6979686d821df101aa2. Özgün dosya yazarda saklanıyor.

## Değişmeyenler
- Araştırma sorusu: Eşleştirilmiş eğitim prosedürü altında TabPFN öğretmen hedefleri, sert etiketlere ve tek tip etiket yumuşatmaya göre öğrenci MLP'lerin performansını artırıyor mu; sonuç kohort tanımına bağlı mı?
- Birincil uç nokta: Hasta bazında birleştirilmiş OOF tahminlerinden Macro F1; birincil karşılaştırmalar damıtma − sert etiket ve damıtma − yumuşatma (eşleştirilmiş hasta-bloklu bootstrap).
- Bölmeler, kohortlar ve etiket kuralları (PTB-XL) aynıdır. Test verisinin daha önce görüldüğü açıkça belirtilecek; yeni analizler bu yüzden keşifsel/ikincil sayılır.

## WP4 — Yeni eğitim gerektirmeyen ikincil analizler (bu turda)
Girdi: kayıtlı OOF olasılıkları (`phase3d_full_cohort_oof_predictions.npz`, `phase3b_repeated_oof_predictions.npz`). Hiçbir model yeniden eğitilmez, hiçbir eşik veya ayar test verisine göre seçilmez.

1. Ayırt edicilik: Makro AUROC (one-vs-rest, eşit ağırlıklı) ve makro AUPRC (sınıf bazında average precision, eşit ağırlıklı); sınıf bazında değerler.
2. Kalibrasyon: Çok sınıflı negatif log-olabilirlik (NLL), çok sınıflı Brier skoru, top-label ECE (15 eşit genişlikte kutu; tanım: Σ_b (n_b/N)·|acc_b − conf_b|) ve sınıf bazında ECE (her sınıf için one-vs-rest, 15 kutu, sınıflar üzerinden ortalama). Güvenilirlik diyagramı (top-label).
3. Belirsizlik: Phase 3D için hasta-bloklu eşleştirilmiş bootstrap (B = 2000, tohum 2026) ile %95 yüzdelik aralıklar; model farkları için aynı indeks vektörü. Phase 3B için tohum bazında değerler ve 5 tohum ortalaması ± SD.
4. Alt gruplar (SAGER/TRIPOD+AI, keşifsel): Phase 3D'de cinsiyet (PTB-XL kodu 0 = erkek, 1 = kadın; PTB-XL belgesindeki %52/%48 oranından çıkarıldı) ve yaş (< 65 / ≥ 65). Her alt grupta Macro F1, makro AUROC, ikili duyarlılık/özgüllük ve HYP recall; alt grup içi hasta-bloklu bootstrap %95 aralık. Alt grup farkları için çoklu karşılaştırma düzeltmesi yapılmaz; sonuçlar betimseldir ve test gücü açısından değerlendirilmez.
5. Doğrulama: AUROC'u Mann–Whitney formülüyle ikinci kez hesaplayıp scikit-learn ile karşılaştırma; ECE uygulamasını sentetik, kalibre olduğu bilinen veride test etme; sınıf sayıları ve paydaların kontrolü.

## Yazarın kararları (2 Kasım 2026)
- Hedef dergi: Biomedical Signal Processing and Control (BSPC).
- Onaylanan yeni deneyler: (1) iç doğrulama kümesinde α/ε/T taraması + entropi eşleştirmeli yumuşatma kontrolü + 8 estimator'lı iç öğretmen; (2) Natural-5000 faktöriyel kohortu; (3) Chapman-Shaoxing'de prosedür-eşleştirilmiş replikasyon (CPSC ve legacy deneyler makaleden çıkarılır); (4) XGBoost öğrenci.
- WP2: önce yalnızca ölçüm (özellik yeniden çıkarılmaz); düzeltme kararı ölçüm sonuçlarına göre verilecek.

## WP2 ölçüm bulguları (2 Kasım 2026; `results/wp2/`)
- QRS 146 ms tavanı: kayıtların %1.1–1.5'i (lead II/I); CD sınıfında %5.0–7.0. 12SL referans QRS ≥ 146 ms olan 378 kayıtta (%2.5) bizim ölçümümüz kısa kalıyor (medyan 120–140 ms).
- Genlikler (R, S, T): 12SL/Uni-G ile r = 0.89–0.99, medyan fark ≤ 0.06 mV.
- QRS ve QT: r ≈ 0.57–0.73; bizim değerler sistematik olarak kısa (QRS 4–20 ms, QT 22–34 ms).
- PR aralığı: referanslarla ilişkisiz (r ≈ 0.0–0.1); bizim PR dağılımı dar (IQR 138–152 ms). Bu, FEAT_50'nin önemli bir ölçüm kusuru; düzeltme kararı yazara sunulacak.

## FEAT_50 v2 düzeltmesi (yazar kararı: 2 Kasım 2026, seçenek a)
Kusurlar (kod incelemesi): (1) ortanca atım R − 200 ms'de kesiliyor, uzun PR'de P başlangıcı pencere dışında kalıyor; (2) P başlangıcı pencerenin başından ileriye doğru aranıyor ve çoğunlukla pencerenin ilk örneğini buluyor (PR ≈ sabit); (3) QRS sınırları R'den dışarı değil sabit pencere içinde aranıyor, 146 ms tavanı oluşuyor.
Düzeltme (`feature_extraction/feat50_v2.py`; arşiv betiği değiştirilmez): ortanca atım R − 350 / + 450 ms; QRS başlangıç/bitişi R'den dışarı doğru (R − 90 / + 120 ms arama sınırı); P tepesi QRS başlangıcından önceki 300 ms içinde, P başlangıcı tepeden geriye doğru; T sonu araması R + 390 ms yerine R + 440 ms'ye kadar (v1'de QT pratikte ~430 ms ile sınırlıydı; ölçülen QT'nin referanslardan 22–34 ms kısa çıkmasının olası nedeni). Diğer özelliklerin tanımı aynı kalır; J noktası QRS bitişine bağlı olduğu için J/ST özellikleri yeni QRS bitişiyle değişir.
Protokol: parametreler yalnızca rastgele 1.000 kayıtlık geliştirme alt kümesinde (tohum 2026) PTB-XL+ referanslarıyla (Uni-G derivasyon bazında QRS/QT, 12SL/Uni-G global PR) ayarlanır; tanı etiketleri kullanılmaz. Değerlendirme kalan 14.160 kayıtta yapılır. Önceden tanımlı kabul ölçütleri: PR–referans Pearson r ≥ 0.6; QRS–Uni-G r v1'den düşük değil; 146 ms tavanında kayıt kalmaz; QT–Uni-G r v1'den düşük değil ve medyan fark küçülür; genlik uyumu (R, S, T) korunur (r farkı ≤ 0.02).
Sonra: 15.160 kayıt v2 ile yeniden çıkarılır; 5.000'lik kohort aynı ecg_id'lerden alınır; ana deneyler (3B, 3C, 3D) ve WP3 v2 özellikleriyle çalıştırılır. v1 sonuçları korunur ve karşılaştırılır.

## Onay bekleyen yeni deneyler (yazarın kararı gerekiyor)
- WP2: QRS arama penceresinin genişletilmesi (özellik çıkarımı ve bütün deneylerin yeniden çalıştırılması) ve PTB-XL+ (12SL, Uni-G, ECGDeli) ile ölçüm uyumu analizi.
- WP3: İç doğrulama kümesinde α/ε/T taraması, entropi eşleştirmeli yumuşatma kontrolü, 8 estimator'lı iç öğretmen, Natural-5000 faktöriyel kohortu, GBDT öğrencileri, Chapman-Shaoxing'de prosedür-eşleştirilmiş replikasyon.
- WP1/WP5: Kaynakçanın doğrulanmış kaynaklarla genişletilmesi ve metnin yeniden yazımı (her kaynak PubMed/scite ile doğrulanacak; doğrulanamayan kaynak kaynakçaya girmez).

## WP3 tasarımı (sonuçlar görülmeden önce yazıldı, 2 Kasım 2026)
Bütün WP3 deneyleri FEAT_50 v2 özellikleriyle yapılır. WP3 analizleri ikincil/keşifseldir; PTB-XL test verisi önceki turlarda görüldüğü için "bağımsız onay" olarak sunulmaz. Chapman-Shaoxing kohortunun öğretmen/öğrenci deneyleri yeni yapılır; ancak aynı protokol PTB-XL'de sonuçlar görülerek geliştirildi.

Kohortlar (hepsinde bir kayıt = bir hasta; bölmeler hasta kimliğiyle gruplanır):
- PTB-XL Quota-5000: Phase 3B kohortu; bölme tohumları 42–46.
- PTB-XL Natural-5000: 15.160'lık uygun kohorttan sınıf oranlarını koruyan tabakalı rastgele 5.000 hasta (örnekleme tohumu 2026, örnekleme etiket dışında hiçbir bilgiye bakmaz); bölme tohumları 42–46. Quota-5000 ile karşılaştırma sınıf dağılımının, Natural-15160 ile karşılaştırma örneklem büyüklüğünün etkisini gösterir (test bileşimi de değiştiği için bu ayrıştırma tam değildir; makalede belirtilecek).
- PTB-XL Natural-15160: Phase 3D kohortu; bölme tohumu 42.
- Chapman Quota-5000: v1 Chapman örneklemindeki aynı 5.000 kayıt (kayıt kimlikleri `data/chapman_54features.csv`); bölme tohumları 42–46.
- Chapman Natural (tam uygun kohort): v1 etiket kurallarıyla (değiştirilmeden) ≥ 18 yaş ve tek üst sınıfa atanabilen 9.315 kayıt; bölme tohumu 42. Not: v1 kuralı sinüs bradikardisi/taşikardisini (SB, ST, SA, SI) ek bulgu yoksa NORM sayar; bu bir sınırlılık olarak raporlanır.

Bölme protokolü (her kohort ve tohum için, 3B ile aynı): dış 5 katlı StratifiedGroupKFold (random_state = tohum); dış eğitim içinden öğrenci eğitim/doğrulama %80/%20 (StratifiedGroupKFold, 5 kat, ilk bölme); StandardScaler yalnızca öğrenci eğitiminde fit edilir. Öğretmen hedefleri öğrenci eğitimi içinde 3 katlı gruplu iç OOF ile üretilir; TabPFN n_estimators = 8 (önceki 4 yerine; bağımsız öğretmen referansıyla eşitlendi). Bağımsız TabPFN referansı tam öğrenci eğitiminde n_estimators = 8. Sıcaklık: q_T ∝ p^(1/T) (TabPFN olasılıklarından).

Öğrenci kolları (her dış katta aynı başlangıç ağırlıkları; MLP 50–128–64–5, 3B/3D eğitim ayarları aynen):
- H: sert etiket.
- S(ε): tek tip yumuşatma, ε ∈ {0,05; 0,10; 0,15; 0,20; 0,25; 0,30}.
- D(α,T): damıtma hedefi (1−α)·onehot + α·q_T, α ∈ {0,25; 0,50; 0,75; 1,00}, T ∈ {1; 2; 4}.
- EM(α,T): örnek bazlı entropi eşleştirmeli yumuşatma. Her eğitim örneği için gerçek sınıfa c_i, diğer sınıfların her birine (1−c_i)/(K−1) verilir; c_i ∈ [1/K, 1], H(hedef) = H(D(α,T) hedefi) olacak şekilde ikiye bölme yöntemiyle bulunur. Böylece hedef entropisi birebir aynı kalır, öğretmenin sınıflar arası benzerlik bilgisi (ve yanlış sınıfa verdiği ağırlık) kaldırılır.
- XGBoost öğrenci: aynı kollar (H, S, D, EM). Yumuşak hedefler, her örneğin K kez çoğaltılıp k. kopyasına q_ik ağırlığı verilmesiyle uygulanır. Sabit ayarlar: max_depth 4, learning_rate 0,05, subsample 0,8, colsample_bytree 0,8, en fazla 2.000 ağaç, doğrulama kümesinde gerçek etiketlerle mlogloss'a göre 50 turluk erken durdurma, tree_method "hist".

Seçim (iç içe; test verisi kullanılmaz): Her kol ailesinde (S, D, EM) her dış katta öğrenci doğrulama kümesinde Macro F1'i en yüksek ayar seçilir (eşitlikte doğrulama NLL'si düşük olan, sonra ızgara sırası). Seçilen ayarın dış test tahminleri "ayarlı" OOF'u oluşturur. Ayrıca önceki çalışmayla süreklilik için sabit D(0,5;1) ve S(0,15) raporlanır.

Karşılaştırmalar (ikincil/keşifsel): ayarlı D − ayarlı S; ayarlı D − ayarlı EM; ayarlı D − H; her biri MLP ve XGBoost için ayrı. Ölçüt Macro F1 (birincil); makro AUROC, NLL, top-label ECE (ikincil). Tek tohumlu kohortlarda eşleştirilmiş hasta-bloklu bootstrap (B = 2000, tohum 2026) %95 aralık; beş tohumlu kohortlarda tohum başına fark, ortalama ± SD ve tohum başına bootstrap aralıkları. Ayrıca her (α,T) noktasında D − EM farkı betimsel olarak verilir (ızgara üzerinde çoklu karşılaştırma; çıkarım için kullanılmaz). Hedeflerin ortalama entropisi her kol için kaydedilir.

Hesaplanmayacaklar / sınırlar: 5 katlı iç OOF öğretmen varyantı yapılmaz (yazar kararı yalnızca 8 estimator'ı onayladı); CatBoost eklenmez (yazar XGBoost'u onayladı).
