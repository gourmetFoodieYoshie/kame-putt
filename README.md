# ニューラルネットワークを用いた亀のパターゴルフ

ROS 2 と TurtleSim を使い、試行データから目標距離に応じた初速度を推定するプロジェクトです。亀を目標方向へ滑らせ、目標付近で停止させることを目指します。

## 概要

- 練習モードで初速度を変えてショットし、初速度と実際の移動距離を CSV に記録します。
- 記録したデータを使って、距離から初速度を推定する全結合型ニューラルネットワークを学習します。
- 推論モードでは学習済みモデルを読み込み、目標距離に対する初速度を出力します。
- NNへの入力は距離のみです。減速モデルの係数やフィールド特性はNNに入力しません。

## リポジトリの構成

```text
.
├── src/
│   └── kame_putt/          # ROS 2 パッケージ
├── putt_controller_baseline.py
├── shot_speed_nn.pt        # 学習済みモデル
├── shot_trials.csv         # 試行データ
├── shot_trials_auto.csv    # 自動収集した試行データ
└── shot_trials_model.csv   # 学習用に整理したデータ
```

`build/`、`install/`、`log/` はビルド時に生成されるため、リポジトリには含めていません。


## 必要な環境

- ROS 2
- turtlesim
- Python 3
- PyTorch
- パッケージの `package.xml` に記載された ROS 2 依存パッケージ

## ビルド

リポジトリのルートで実行します。

```bash
source /opt/ros/<ROS_DISTRO>/setup.bash
colcon build --packages-select kame_putt
source install/setup.bash
```

`<ROS_DISTRO>` は使用する ROS 2 のディストリビューション名に置き換えてください。

## 実行

### 1. TurtleSimを起動

```bash
ros2 run turtlesim turtlesim_node
```

### 2. コントローラを起動

別の端末でワークスペースを読み込み、ノードを起動します。

```bash
source /opt/ros/<ROS_DISTRO>/setup.bash
cd <ワークスペース>
source install/setup.bash
ros2 run kame_putt putt_controller
```

`setup.py` で登録されている実行名は `putt_controller` です。ノード名も `putt_controller` です。既定の `mode` は `infer` です。

練習モードでは、起動時に次のように指定します。

```bash
ros2 run kame_putt putt_controller --ros-args -p mode:=train -p practice_shots:=30
```

目標位置は `target_x` と `target_y` パラメータで指定できます。既定値はそれぞれ `8.0`、`5.5` です。


