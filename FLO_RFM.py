
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
import pandas as pd
from pandas.core.methods import describe

df = pd.read_csv("flo_data_20k.csv")

print(df.head(10))

  # 2. Veri setinde
                     # a. İlk 10 gözlem,
                    print(df.head(10))

                     # b. Değişken isimleri,
                    print(df.shape)
                    print(df.columns)
                     # c. Betimsel istatistik,
                     print(df.describe().T)

                     # d. Boş değer,
                     print(df.isnull().sum())
                     # e. Değişken tipleri, incelemesi yapınız.
                     df.info()

           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
#  Toplam alışveriş sayısı

df["order_num_total"] = (
    df["order_num_total_ever_online"] +
    df["order_num_total_ever_offline"])

# Toplam harcama

df["customer_value_total"] = (
    df["customer_value_total_ever_online"] +
    df["customer_value_total_ever_offline"])
print(df.head(10))


           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = [
    "first_order_date",
    "last_order_date",
    "last_order_date_online",
    "last_order_date_offline"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col])
df.info()

           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({
    "master_id": "count",
    "order_num_total": "mean",
    "customer_value_total": "mean"
})
channel_analysis = (
    df.groupby("order_channel")
      .agg({
          "master_id": "count",
          "order_num_total": "mean",
          "customer_value_total": "mean"
      })
      .rename(columns={
          "master_id": "Musteri_Sayisi",
          "order_num_total": "Ortalama_Alisveris_Sayisi",
          "customer_value_total": "Ortalama_Harcama"
      })
)

print(channel_analysis)
 # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.





# En çok harcama yapan ilk 10 müşteri

top10_customer_value = df.sort_values(
    by="customer_value_total",
    ascending=False
).head(10)

print(top10_customer_value)

           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

top10_order = df.sort_values(
    by="order_num_total",
    ascending=False
).head(10)

print(top10_order)

           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_prep(dataframe):

    dataframe["order_num_total"] = (
        dataframe["order_num_total_ever_online"] +
        dataframe["order_num_total_ever_offline"]
    )

    dataframe["customer_value_total"] = (
        dataframe["customer_value_total_ever_online"] +
        dataframe["customer_value_total_ever_offline"]
    )

    date_columns = [
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline"
    ]

    for col in date_columns:
        dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe


df = data_prep(df)

# GÖREV 2: RFM Metriklerinin Hesaplanması
df["last_order_date"].max()
analysis_date = pd.Timestamp("2021-06-01")
rfm = df[["master_id"]].copy()
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]
print(rfm.head())

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
rfm["recency_score"] = pd.qcut(
    rfm["recency"],
    5,
    labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(
    rfm["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(
    rfm["monetary"],
    5,
    labels=[1, 2, 3, 4, 5])

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
rfm["RF_SCORE"] = (
    rfm["recency_score"].astype(str) +
    rfm["frequency_score"].astype(str))
rfm[[
    "recency_score",
    "frequency_score",
    "RF_SCORE"
]].head()
seg_map = {

    r"[1-2][1-2]": "hibernating",

    r"[1-2][3-4]": "at_Risk",

    r"[1-2]5": "cant_loose",

    r"3[1-2]": "about_to_sleep",

    r"33": "need_attention",

    r"[3-4][4-5]": "loyal_customers",

    r"41": "promising",

    r"51": "new_customers",

    r"[4-5][2-3]": "potential_loyalists",

    r"5[4-5]": "champions"

}
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
print(rfm[[
    "RF_SCORE",
    "segment"
]].head())

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
segment_analysis = rfm.groupby("segment").agg({
    "recency": "mean",
    "frequency": "mean",
    "monetary": "mean"})
print(segment_analysis)


           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.

                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
new_brand = rfm[
    ( rfm["segment"].isin(["champions", "loyal_customers"])) &
    (rfm["monetary"] > 250) &
    ( df["interested_in_categories_12"].str.contains("KADIN"))]
new_brand[["master_id"]].to_csv(
    "yeni_marka_hedef_musteri_id.csv",
    index=False)

                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.
discount_target = rfm[
    (rfm["segment"].isin([
            "cant_loose",
            "hibernating",
            "new_customers"])) &
    ( df["interested_in_categories_12"].str.contains(
            "ERKEK|COCUK",
            regex=True))]
discount_target[["master_id"]].to_csv(
    "indirim_hedef_musteri_ids.csv",
    index=False)
print(discount_target.shape)
print(discount_target.head())

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.
####################################################
# GÖREV 6: Tüm Süreci Fonksiyonlaştırma
####################################################

def create_rfm(dataframe, csv=False):

    dataframe = data_prep(dataframe)

    analysis_date = pd.Timestamp("2021-06-01")

    rfm = dataframe[["master_id"]].copy()

    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).dt.days
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    rfm["recency_score"] = pd.qcut(
        rfm["recency"],
        5,
        labels=[5, 4, 3, 2, 1])

    rfm["frequency_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5])

    rfm["monetary_score"] = pd.qcut(
        rfm["monetary"],
        5,
        labels=[1, 2, 3, 4, 5])

    rfm["RF_SCORE"] = (
        rfm["recency_score"].astype(str) +
        rfm["frequency_score"].astype(str))

    seg_map = {
        r"[1-2][1-2]": "hibernating",
        r"[1-2][3-4]": "at_Risk",
        r"[1-2]5": "cant_loose",
        r"3[1-2]": "about_to_sleep",
        r"33": "need_attention",
        r"[3-4][4-5]": "loyal_customers",
        r"41": "promising",
        r"51": "new_customers",
        r"[4-5][2-3]": "potential_loyalists",
        r"5[4-5]": "champions"}

    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

    if csv:
        rfm.to_csv("rfm.csv", index=False)

    return rfm

rfm = create_rfm(df, csv=True)
print(rfm.head())


###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
df = pd.read_csv("flo_data_20k.csv")

# 2. Veri setinde
        # a. İlk 10 gözlem,
        print(df.head(10))
        # b. Değişken isimleri,
        print(df.columns)
        # c. Boyut,
        print(df.shape)
        # d. Betimsel istatistik,
        print(df.describe().T)
        # e. Boş değer,
        print(df.isnull().sum())
        # f. Değişken tipleri, incelemesi yapınız.
       df.info()

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["customer_value_total"] = (
    df["customer_value_total_ever_online"] +
    df["customer_value_total_ever_offline"])
df.head()

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns=["first_order_date", "last_order_date","last_order_date_online" ,"last_order_date_offline"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col])
df.info()

# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)



# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız. 
df["order_num_total"] = (
    df["order_num_total_ever_online"] +
    df["order_num_total_ever_offline"]
)

df["customer_value_total"] = (
    df["customer_value_total_ever_online"] +
    df["customer_value_total_ever_offline"]
)

channel_analysis = df.groupby("order_channel").agg({
    "master_id": "count",
    "order_num_total": "sum",
    "customer_value_total": "sum"
})
print(df.columns)

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
top10_customer_value = df.sort_values(
    by="customer_value_total",
    ascending=False
).head(10)

print(top10_customer_value)



# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
top10_order = df.sort_values(
    by="order_num_total",
    ascending=False
).head(10)

print(top10_order)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_prep(dataframe):

    dataframe["order_num_total"] = (
        dataframe["order_num_total_ever_online"] +
        dataframe["order_num_total_ever_offline"])

    dataframe["customer_value_total"] = (
        dataframe["customer_value_total_ever_online"] +
        dataframe["customer_value_total_ever_offline"])

    date_columns = [
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline"]

    for col in date_columns:
        dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe


df = data_prep(df)

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi
analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)
print(analysis_date)

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe
analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)

# RFM DataFrame

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
rfm = df[["master_id"]].copy()

rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

print(rfm.head())
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi
rfm["recency_score"] = pd.qcut(
    rfm["recency"],
    5,
    labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(
    rfm["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(
    rfm["monetary"],
    5,
    labels=[1, 2, 3, 4, 5])
rfm["RF_SCORE"] = (
    rfm["recency_score"].astype(str) +
    rfm["frequency_score"].astype(str))

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = (
    rfm["recency_score"].astype(str) +
    rfm["frequency_score"].astype(str))

###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme
seg_map = {r"[1-2][1-2]": "hibernating",

    r"[1-2][3-4]": "at_Risk",

    r"[1-2]5": "cant_loose",

    r"3[1-2]": "about_to_sleep",

    r"33": "need_attention",

    r"[3-4][4-5]": "loyal_customers",

    r"41": "promising",

    r"51": "new_customers",

    r"[4-5][2-3]": "potential_loyalists",

    r"5[4-5]": "champions"}

###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
print(rfm.head())
segment_analysis = rfm.groupby("segment").agg({
    "recency": "mean",
    "frequency": "mean",
    "monetary": "mean"})
print(segment_analysis)
# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.
new_brand = rfm[
    (rfm["segment"].isin([
            "champions",
            "loyal_customers"])) &( df["interested_in_categories_12"].str.contains("KADIN") )]

new_brand[["master_id"]].to_csv(
    "yeni_marka_hedef_musteri_id.csv",
    index=False)


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
discount_target = rfm[
    (rfm["segment"].isin([
            "cant_loose",
            "hibernating",
            "new_customers"])) &(df["interested_in_categories_12"].str.contains(
            "ERKEK|COCUK",
            regex=True))]

discount_target[["master_id"]].to_csv(
    "indirim_hedef_musteri_ids.csv",
    index=False)