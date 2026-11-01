# TurtleSim上で床の抵抗を簡易的に再現するモデル

BREAKAWAY_SPEED = 0.20  # 動き出すための最低初速
STICK_SPEED = 0.04      # これ以下になったら停止とみなす
COULOMB_DECEL = 0.12    # 一定の抵抗による減速度
VISCOUS_COEFF = 0.08    # 速度に比例する抵抗の係数


def launch_speed(speed):
    """静止摩擦を簡易的に表す。最低初速に届かなければ動き出さない。"""
    if speed < BREAKAWAY_SPEED:
        return 0.0
    return speed


def update_speed(speed, dt):
    """一定抵抗と粘性抵抗で、時間 dt 後の速度を計算する。"""
    if speed <= STICK_SPEED:
        return 0.0

    deceleration = COULOMB_DECEL + VISCOUS_COEFF * speed
    new_speed = max(0.0, speed - deceleration * dt)

    if new_speed <= STICK_SPEED:
        return 0.0
    return new_speed
