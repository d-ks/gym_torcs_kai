# Gym-TORCS 改

[English](README.md)

**Gym-TORCS 改** は、強化学習でドライビングシミュレータ「TORCS」を利用するためのOpenAI Gymライクな環境である「Gym-TORCS」の改造版です。

鋭意修正中。

## インストール
[オリジナルリポジトリ](https://github.com/ugo-nama-kun/gym_torcs)の手順に従ってください。

## 環境要件
Ubuntu 16.04 LTS環境において、以下のパッケージがインストールされていること：
* Python 3
* xautomation (http://linux.die.net/man/7/xautomation)
* OpenAI-Gym (https://github.com/openai/gym)
* numpy
* vtorcs-RL-color (vtorcs-RL-colorディレクトリにインストール説明があります)

# 使い方
**gym_torcs_kai** フォルダをRLのコードのある場所と同じ階層に置き、環境名として次の名前を指定します：
```
gym_torcs_kai:GymTorcsKai-v0
```

# Acknowledgement
- **gym_torcs** : developed by Naoto Yoshida.

