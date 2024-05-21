# 建物AIプログラム

このレポジトリをforkするなりして、使ってみてください。
建物AIのプログラムは、基本的にGPU1,2を使います。
とりあえず、GPU2という前提で以下の説明を書きます。

## 画像データ(部屋の画像)の保管場所

gpu2内の`/mnt/data-raid/{ユーザ名}`内に画像データは保管しましょう。

プログラムのコード自体は、`/home/{ユーザ名}/`内に保管するので、`/home/{ユーザ名}/`以下に保管ディレクトリのシンボリックリンクを作ると便利です。

坂本の場合
`/mnt/data-raid/sakamoto/dx/coins_data`を`/home/sakamoto/dx/data`に紐づけてます。

## 重要なファイル、ディレクトリ

トップの階層を整備しました。
以下のファイル、ディレクトリは流用出来ると思います。

+ config
+ data(シンボリックリンク)
+ pythonlibs
+ result
+ utils
+ .gitignore
+ main.py
+ g1_dx.def
+ g2_dx.def
+ gpu1_exec.sh
+ gpu2_exec.sh
+ main.py
+ README.md

その他のディレクトリ、ファイルに関しては、今回整備していないので動くか不明です。消しちゃってもらって大丈夫です。
ちなみに、坂本の卒業研究ではreduce_vibrationのディレクトリを使っていました。

## singularity(仮想環境、コンテナ)使用方法

1. defファイル(コンテナの設計書)からsifファイル(コンテナのイメージファイル)をビルド

defファイルがある階層にて以下を実行

```
singularity build --fakeroot g2_dx.sif  g2_dx.def
```

だだし、defファイルは以下の部分を個人用に書き換えてください。

```
%files
    /mnt/data-raid/{ユーザ名}/ /mnt
```

2. 以下のコマンドでコンテナ起動、コンテナに入る

```
bash gpu2_exec.sh
```

bashファイルの内容(個人用に書き換えてください)。

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

## データセット配置、分割

坂本がslackに貼ったgoogledriveのリンク内のデータ(例えば、`坂本画像/data2`など)を、`/home/{ユーザ名}/dx/data/`に配置してみてください。

data2のように、画像のラベルごとに0000~1111のディレクトリに分けられた画像データを配置します。

以下のコマンドでデータを分割できます。

```
python utils/make_dataset.py
```

`/home/{ユーザ名}/dx/data/{config.which_data}`内に`train`(訓練データ), `val`(検証データ)ディレクトリ作成されます。

## プログラム実行方法

```
# 通常の実行
python main.py
# or
# バックグラウンド実行(sshが途中で切れても、GPU側で処理が継続)
nohup python main.py &
```

バックグラウンド実行を行うと、標準出力がnohup.outに出力される。

実行時には、`/home/{ユーザ名}/dx/result/{config.which_data}`以下に実行時刻のディレクトリが作成され、検証結果が出力される(以下一例)。

+ confusion_matrix(本当のラベルと予測ラベルの割合比較)
+ latent_space(潜在空間における入力データ分類)
+ abst.txt
+ ~.png(精度グラフ、損失グラフ、検証データのサンプル画像)

## 実行パラメータ設定

`/home/{ユーザ名}/dx/config/config.json`に実行時のパラメータを設定

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
