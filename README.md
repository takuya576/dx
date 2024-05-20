# 建物AIプログラム

建物AIのプログラムは、基本的にGPU1,2を使います。
gpu1,2へのユーザ登録はZhuさんにお願いしてください。
とりあえず、GPU2という前提で以下の説明を書きます。

## singularity(仮想環境、コンテナ)使用方法

1. defファイル(コンテナの設計書)からsifファイル(コンテナのイメージファイル)をビルド

```
singularity build --fakeroot g2_dx.sif  g2_dx.def
```

2. 以下のコマンドでコンテナ起動、コンテナに入る

```
bash gpu2_exec.sh
```

bashファイルの内容

```
#!bin/bash/
singularity shell --nv --bind /mnt/data-raid/{gpu2の自分のユーザ名}/:/mnt/data-raid/{gpu2の自分のユーザ名}/ g2_dx.sif
# --nvはGPUを使うためのおまじない
# —bindはマウントのためのオプション
```

 3. コンテナから抜ける

```
exit
```

4. 今後、コンテナに入るときは、２番を行う

## データセット分割

```
python utils/make_dataset.py
```

`~/dx/data/{config.which_data}`内に`train`(訓練データ), `val`(検証データ)ディレクトリ作成

## プログラム実行方法

```
# 通常の実行
python main.py
# or
# バックグラウンド実行(sshが途中で切れても、GPU側で処理が継続)
nohup python main.py &
```

バックグラウンド実行を行うと、標準出力がnohup.outに出力される
実行時、`~/dx/result`以下に検証結果が出力される(一例)

+ confusion_matrix(本当のラベルと予測ラベルの割合比較)
+ latent_space(潜在空間における入力データ分類)
+ abst.txt
+ ~.png(精度グラフ、損失グラフ、検証データのサンプル画像)

## 実行パラメータ設定

`~/dx/config/config.json`に実行時のパラメータを設定

```
# ディープラーニングに用いるパラメータ
"net": "vit_b_16", # 用いる学習器
"pretrained": true, # 事前学習の有無
"transfer": false, # ファインチューニング：True、転移学習：False
"lr": 0.001,
"momentum": 0.9,
"num_epochs": 1,
"batch_size": 10,
"nvidia": 0, # 使用するGPUは 0 or 1　が指定できる
"num_val": 8, # `~/dx/utils/make_dataset.py`を実行するとき用いる(検証データの比率)

# どのデータセットを用いるかを指定
"which_data": "data2", # どのデータを用いるか
"train_data": "train", # which_dataのうちどれを訓練データにするか
"test_data": "val", # which_dataのうちどれを検証データにするか

# 以下は坂本の個人研究用パラメータ(関係ないやつ)
"generated": false,
"alpha": 0.5,
"beta": 10,
"a": 10,
"sigmoid": false
```
