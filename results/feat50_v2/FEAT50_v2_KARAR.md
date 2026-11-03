# FEAT_50 v2 — değerlendirme ve karar (2 Kasım 2026)

Kaynak: `heldout_evaluation.csv`, `heldout_extra.json` (ayarlamada kullanılmayan 14.160 kayıt); ayarlama: `dev_grid.csv` ve P/QRS keşif denemeleri (1.000 kayıtlık geliştirme alt kümesi, tohum 2026). Tanı etiketleri hiçbir aşamada kullanılmadı.

## Önceden tanımlı kabul ölçütleri ve sonuçlar
| Ölçüt | v1 | v2 | Durum |
|---|---|---|---|
| PR–referans r ≥ 0.6 (AF dışı) | L2: 0.07 (12SL), 0.08 (Uni-G); L1: 0.08 | L2: **0.65** (12SL), 0.60 (Uni-G); L1: 0.53 / 0.47 | Lead II'de sağlandı; lead I'de sağlanmadı |
| 146 ms tavanında kayıt kalmasın | %1.5 (L1), %1.1 (L2) | %0.18 / %0.19 (tavan değil, gerçek 146 ms değerleri) | Sağlandı |
| QRS–Uni-G r v1'den düşük olmasın | 0.612 / 0.661 | 0.578 / 0.647 | **Sağlanmadı** (Δr −0.034 / −0.014) |
| QT–Uni-G r düşük olmasın, medyan fark küçülsün | r 0.660 / 0.594; fark −26 / −22 ms | r 0.648 / 0.577; fark −26 / −18 ms | **Sağlanmadı** (r'de −0.01/−0.02; L2 farkı küçüldü) |
| Genlik uyumu korunsun (Δr ≤ 0.02) | R/S/T r 0.89–0.98 | aynı (Δr ≤ 0.001) | Sağlandı |

Ek (önceden tanımlı değil, tavanın etkilediği alt grup): 12SL QRS ≥ 146 ms olan 357 kayıtta medyan fark v1 −24/−32 ms → v2 −6/−14 ms; QRS ≥ 120 ms olan 890 kayıtta −20/−24 → −10/−14 ms. Uni-G ile korelasyon bu alt gruplarda benzer.

Değişmeyen özellikler: RR, spektral ve istatistiksel 18×2 özellik v1 ile birebir aynı (geliştirme kümesinde fark 0); kimlik/etiket/demografi sütunları birebir kopyalandı.

## Karar
v2 benimsendi. Gerekçe: PR ilk kez referanslarla anlamlı biçimde ilişkili (lead II r ≈ 0.6–0.65, v1'de ~0.07); QRS tavanı kalktı ve geniş QRS'lerdeki sistematik kısaltma yaklaşık yarıya indi. Bedel: normal aralıktaki QRS ve QT korelasyonunda küçük düşüş (0.01–0.03). Karşılanmayan ölçütler (lead I PR, QRS ve QT korelasyonu) makalede açıkça raporlanacak; ölçütler sonradan değiştirilmedi.

Bağlam: iki ticari algoritmanın kendi arasındaki uyum PR için r = 0.81, QRS için r ≈ 0.79 (geliştirme kümesi). Bizim tek/iki derivasyonlu, ortanca atıma dayalı yöntemimiz bu düzeyin altında kalıyor; bu bir sınırlılık olarak raporlanacak.

## Dış kontrol: Chapman-Shaoxing (sonradan eklendi; kabul ölçütü değildir)
v2 çıkarıcısı Chapman verisinde hiç ayarlanmadı. Kaynak: `data_v2/chapman_eligible_features_v2.csv` (9.315 kayıt), v1 5.000'lik örneklemde GE MUSE (12 derivasyon, global) ölçümleriyle karşılaştırma:
| | v1 r / medyan fark | v2 r / medyan fark |
|---|---|---|
| QRS lead I | 0.479 / −16 ms | 0.398 / −16 ms |
| QRS lead II | 0.523 / −12 ms | 0.475 / −12 ms |
| QT lead I | 0.571 / −32 ms | 0.570 / −30 ms |
| QT lead II | 0.523 / −28 ms | 0.498 / −22 ms |
Chapman'da PR referansı yok. QRS korelasyonundaki düşüş PTB-XL'dekinden (Δr −0.01/−0.03) büyük (Δr −0.05/−0.08); makalede sınırlılık olarak raporlanacak. v1 etiket kuralı 5.000 kayıtta birebir yeniden üretildi; RR ve spektral özellikler v1 ile aynı.
